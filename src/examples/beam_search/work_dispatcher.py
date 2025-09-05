import asyncio
import aiohttp
import os
import time
from itertools import cycle
from datasets import load_dataset, Dataset
from tqdm import tqdm

from src.config import Config
from src.sal.score import score
from src.utils import dataset_name
from src.examples.beam_search.scoring import calculate_and_print_accuracies, create_dataset_from_results


class WorkDispatcher:
    def __init__(self, location: str, base_port: int, num_replicas: int, max_concurrent_requests: int = 100):
        self.location = location
        self.base_port = base_port
        self.num_replicas = num_replicas
        self.server_urls = [f"{location}:{base_port + i}" for i in range(num_replicas)]
        self.url_cycle = cycle(self.server_urls)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def send_beam_search_request(self, problem: str, search_width: int = 4, select_top_k: int = 1, max_iterations: int = 40) -> dict:
        """Send a beam search request to the agentic server using round-robin load balancing"""
        async with self.semaphore:
            server_url = next(self.url_cycle)
            async with aiohttp.ClientSession() as session:
                payload = {
                    "problem": problem,
                    "search_width": search_width,
                    "select_top_k": select_top_k,
                    "max_iterations": max_iterations
                }
                
                async with session.post(f"{server_url}/beam_search", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"⚠️  Server error from {server_url}: {response.status} - {error_text}")
                        raise Exception(f"Server error {response.status} from {server_url}: {error_text}")

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

async def main():
    """Main function that orchestrates the work dispatcher"""
    
    # Configuration
    base_server_port = os.environ.get("BASE_SERVER_PORT", None)
    num_replicas = os.environ.get("NUM_REPLICAS", None)

    assert base_server_port is not None, "BASE_SERVER_PORT must be set"
    assert num_replicas is not None, "NUM_REPLICAS must be set"

    base_server_port = int(base_server_port)
    num_replicas = int(num_replicas)
    max_concurrent_requests = 100
    search_width = 4
    select_top_k = 1
    max_iterations = 40
    num_problems = 10

    print("="*80)
    print("WORK DISPATCHER - BEAM SEARCH EVALUATION")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    dataset_list = list(dataset)[:num_problems]
    print(f"Loaded {len(dataset_list)} problems")
    
    # Initialize work dispatcher
    dispatcher = WorkDispatcher("http://localhost", base_server_port, num_replicas, max_concurrent_requests)
    print(f"Generated server URLs: {dispatcher.server_urls}")
    
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
        print(f"Used round-robin load balancing across {num_replicas} servers")
        
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
