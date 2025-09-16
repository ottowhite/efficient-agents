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


class BatchWorkDispatcher:
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
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60 * 24)) as session:
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


class ArrivalWorkDispatcher:
    def __init__(self, location: str, base_port: int, num_replicas: int, dataset: list, 
                 timeout: float, endpoint: str, arrival_rate: float, metric_printing_frequency: int):
        """
        Initialize ArrivalWorkDispatcher for load testing with arrival rate control.
        
        Args:
            location: Base URL location (e.g., "http://localhost")
            base_port: Starting port number
            num_replicas: Number of server replicas
            dataset: Dataset to cycle through
            timeout: HTTP client timeout in seconds
            endpoint: API endpoint to hit (e.g., "/beam_search")
            arrival_rate: Requests per second
            metric_printing_frequency: Print metrics every N requests
        """
        self.location = location
        self.base_port = base_port
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.timeout = timeout
        self.endpoint = endpoint
        self.arrival_rate = arrival_rate
        self.metric_printing_frequency = metric_printing_frequency
        self.requests_in_flight = 0
        
        self.server_urls = [f"{location}:{base_port + i}" for i in range(num_replicas)]
        self.url_cycle = cycle(self.server_urls)
        self.dataset_cycle = cycle(dataset)
        self.last_request_sent_time: float = 0.0
        
        self.latencies: list[float] = []
        self.requests_completed_count = 0
        self.start_time: float | None = None
    
    async def send_request(self, **kwargs) -> dict:
        """Send a request to the server using round-robin load balancing"""
        server_url = next(self.url_cycle)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:

            self.requests_in_flight += 1
            request_start_time = time.time()
            self.last_request_sent_time = request_start_time
            async with session.post(f"{server_url}{self.endpoint}", json=kwargs) as response:
                request_end_time = time.time()

                latency = request_end_time - request_start_time
                self.latencies.append(latency)
                self.requests_completed_count += 1
                self.requests_in_flight -= 1
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"⚠️  Server error from {server_url}: {response.status} - {error_text}")
                    raise Exception(f"Server error {response.status} from {server_url}: {error_text}")
    
    def print_metrics_dashboard(self):
        """Print metrics dashboard with current statistics"""

        metrics_dashboard_start_time = time.time()
        if not self.latencies:
            return
        
        # Clear the terminal
        os.system("clear")
        
        avg_latency = sum(self.latencies) / len(self.latencies)
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        requests_completed_rate = self.requests_completed_count / elapsed_time if elapsed_time > 0 else 0
        
        print("=" * 60)
        print("📊 ARRIVAL WORK DISPATCHER METRICS DASHBOARD")
        print("=" * 60)
        print(f"📈 Total Requests: {self.requests_completed_count}")
        print(f"⏱️  Average Latency: {avg_latency:.3f}s")
        print(f"🎯 Configured Arrival Rate: {self.arrival_rate:.2f} req/s")
        print(f"🔄 Requests Completed Rate: {requests_completed_rate:.2f} req/s")
        print(f"🔄 Requests In Flight: {self.requests_in_flight}")
        print(f"⏰ Elapsed Time: {elapsed_time:.1f}s")
        print(f"🌐 Active Servers: {len(self.server_urls)}")
        print("=" * 60)

        # self.latencies = []
        # self.request_count = 0

        metrics_dashboard_end_time = time.time()
        metrics_dashboard_latency = metrics_dashboard_end_time - metrics_dashboard_start_time
        print(f"🔄 Metrics Dashboard Latency: {metrics_dashboard_latency:.3f}s, once every {self.metric_printing_frequency} requests, {metrics_dashboard_latency * self.metric_printing_frequency:.3f}s per second")
    
    async def start(self, search_width: int, select_top_k: int, max_iterations: int):
        """Start the arrival-based load testing"""
        self.start_time = time.time()
        sleep_interval = 1.0 / self.arrival_rate  # Time between requests
        
        print(f"🚀 Starting ArrivalWorkDispatcher")
        print(f"📊 Arrival rate: {self.arrival_rate} req/s (interval: {sleep_interval:.3f}s)")
        print(f"📈 Metrics frequency: every {self.metric_printing_frequency} requests")
        print(f"🔗 Endpoints: {self.server_urls}")
        
        try:
            while True:
                # Get next sample from dataset
                sample = next(self.dataset_cycle)

                payload = {
                    "problem": sample["problem"],
                    "search_width": search_width,
                    "select_top_k": select_top_k,
                    "max_iterations": max_iterations
                }
                
                # Create async task for the request
                asyncio.create_task(self.send_request(**payload))
                
                # Print metrics dashboard at specified frequency
                if self.requests_completed_count % self.metric_printing_frequency == 0:
                    self.print_metrics_dashboard()
                
                time_since_last_request = time.time() - self.last_request_sent_time
                time_to_sleep = max(0, sleep_interval - time_since_last_request)
                # Wait for the arrival rate sleep interval
                await asyncio.sleep(time_to_sleep)

                
        except KeyboardInterrupt:
            print("\n🛑 Stopping ArrivalWorkDispatcher...")
            self.print_metrics_dashboard()
        except Exception as e:
            print(f"❌ Error in ArrivalWorkDispatcher: {e}")
            self.print_metrics_dashboard()


async def main():
    """Main function that orchestrates the work dispatcher"""
    
    # Configuration
    base_server_port = os.environ.get("BASE_SERVER_PORT", None)
    num_replicas = os.environ.get("NUM_REPLICAS", None)
    num_problems = os.environ.get("NUM_PROBLEMS", None)
    concurrent_problems = os.environ.get("CONCURRENT_PROBLEMS", None)

    assert base_server_port is not None, "BASE_SERVER_PORT must be set"
    assert num_replicas is not None, "NUM_REPLICAS must be set"


    base_server_port = int(base_server_port)
    num_replicas = int(num_replicas)
    search_width = 4
    select_top_k = 1
    max_iterations = 40

    print("="*80)
    print("WORK DISPATCHER - BEAM SEARCH EVALUATION")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    arrival_dispatcher = True

    if arrival_dispatcher:
        dataset_list = list(dataset)
        print(f"Loaded {len(dataset_list)} problems for arrival dispatcher")
        real_dispatcher = ArrivalWorkDispatcher(
            "http://localhost",
            base_server_port,
            num_replicas,
            dataset_list,
            timeout=5 * 60,
            endpoint="/beam_search",
            arrival_rate=1,
            metric_printing_frequency=5)

        await real_dispatcher.start(search_width, select_top_k, max_iterations)

        return

    assert num_problems is not None, "NUM_PROBLEMS must be set"
    num_problems = int(num_problems)
    assert concurrent_problems is not None, "CONCURRENT_PROBLEMS must be set"
    max_concurrent_requests = int(concurrent_problems)
    dataset_list = list(dataset)[:num_problems]
    print(f"Loaded {len(dataset_list)} problems for batch dispatcher")
    # Initialize work dispatcher
    dispatcher = BatchWorkDispatcher("http://localhost", base_server_port, num_replicas, max_concurrent_requests)

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
