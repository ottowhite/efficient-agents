import asyncio
import aiohttp
import time
from datasets import load_dataset, Dataset
from tqdm import tqdm

from src.config import Config
from src.sal.score import score
from src.utils import dataset_name
from src.examples.beam_search.scoring import calculate_and_print_accuracies, create_dataset_from_results


class WorkDispatcher:
    def __init__(self, server_url: str = "http://localhost:5000", max_concurrent_requests: int = 100):
        self.server_url = server_url
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def send_beam_search_request(self, problem: str, search_width: int = 4, select_top_k: int = 1, max_iterations: int = 40) -> dict:
        """Send a beam search request to the agentic server"""
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "problem": problem,
                    "search_width": search_width,
                    "select_top_k": select_top_k,
                    "max_iterations": max_iterations
                }
                
                async with session.post(f"{self.server_url}/beam_search", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Server error {response.status}: {error_text}")

    async def process_dataset(self, dataset_slice: list, search_width: int = 4, select_top_k: int = 1, max_iterations: int = 40) -> list:
        """Process a dataset slice by sending requests to the agentic server"""
        
        # Create async tasks for all problems
        tasks = [
            asyncio.create_task(self.send_beam_search_request(
                sample["problem"], 
                search_width, 
                select_top_k, 
                max_iterations
            ))
            for sample in dataset_slice
        ]
        
        # Process tasks with progress bar
        for future in tqdm(
            asyncio.as_completed(tasks), 
            total=len(tasks), 
            desc="Processing problems"
        ):
            await future
        
        return await asyncio.gather(*tasks)

async def check_server_health(server_url: str) -> bool:
    """Check if the agentic server is healthy and ready"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return health_data.get("models_initialized", False)
                return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

async def main():
    """Main function that orchestrates the work dispatcher"""
    
    # Configuration
    server_url = "http://localhost:5000"
    max_concurrent_requests = 100
    search_width = 4
    select_top_k = 1
    max_iterations = 40
    num_problems = 10

    print("="*80)
    print("WORK DISPATCHER - BEAM SEARCH EVALUATION")
    print("="*80)
    
    # Check server health
    print("Checking server health...")
    if not await check_server_health(server_url):
        print(f"❌ Server at {server_url} is not healthy or models not initialized")
        print("Please start the agentic server first: python src/examples/agentic_server.py")
        return
    
    print("✅ Server is healthy and ready")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    dataset_list = list(dataset)[:num_problems]
    print(f"Loaded {len(dataset_list)} problems")
    
    # Initialize work dispatcher
    dispatcher = WorkDispatcher(server_url, max_concurrent_requests)
    
    # Start timing
    time_start = time.time()
    print(f"\nStarting beam search evaluation with {max_concurrent_requests} concurrent requests...")
    print(f"Search parameters: width={search_width}, top_k={select_top_k}, max_iter={max_iterations}")
    
    # Process dataset
    try:
        beam_search_results = await dispatcher.process_dataset(
            dataset_list, 
            search_width, 
            select_top_k, 
            max_iterations
        )
        
        time_end = time.time()
        total_time = time_end - time_start
        print(f"\n✅ Completed processing in {total_time:.2f} seconds")
        print(f"Average time per problem: {total_time/len(dataset_list):.2f} seconds")
        
        # Transform results into dataset format
        print("\nTransforming results into dataset format...")
        result_dataset = create_dataset_from_results(dataset_list, beam_search_results)
        
        # Create config for scoring
        print("Creating configuration for scoring...")
        config = Config(
            search_width=search_width,
            num_proc=1,  # Keep simple for this example
            search_algorithm="beam_search"
        )
        
        # Apply scoring using the same function as beam_search_async
        print("Applying scoring algorithms...")
        scored_dataset = score(result_dataset, config)
        
        print(f"✅ Scoring completed. Dataset now contains prediction fields for analysis.")
        print(f"Total problems processed: {len(scored_dataset)}")
        
        # Calculate and print all accuracy metrics
        calculate_and_print_accuracies(scored_dataset)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())
