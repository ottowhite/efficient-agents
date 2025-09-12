import asyncio
import os
import time
import string
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer
from langchain_openai import OpenAI
from dotenv import load_dotenv
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    latency: float
    tokens_generated: int


@dataclass
class ExperimentResults:
    """Results for an entire experiment"""
    individual_latencies: List[float]
    end_to_end_latency: float
    throughput: float  # requests per second
    tokens_per_second: float


class ModelWrapper:
    """Base class for model wrappers"""
    async def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class LangChainWrapper(ModelWrapper):
    """Wrapper for LangChain-based model communication over API"""
    
    def __init__(self, model_name: str, base_url: str, temperature: float = 0.0):
        self.model = OpenAI(
            model=model_name,
            base_url=base_url,
            api_key="EMPTY",  # type: ignore
            streaming=False,
            temperature=temperature
        )
    
    async def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text using LangChain OpenAI API"""
        time.sleep(0.1)
        return await self.model.ainvoke(prompt, max_tokens=max_tokens, stop=None)

next_request_id = 0

class VLLMWrapper(ModelWrapper):
    """Wrapper for direct vLLM engine"""
    
    def __init__(self, model_name: str, temperature: float = 0.0):
        # TODO: Try with both log requests enabled and disabled
        engine_args = AsyncEngineArgs(
            model=model_name,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            tensor_parallel_size=1,
            enable_log_requests=False
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._SamplingParams = SamplingParams
        self.req_id = 0
    
    async def generate(self, prompt: str, max_tokens: int) -> str:
        req_id = self.req_id
        self.req_id += 1

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=max_tokens,
            ignore_eos=True  # Ignore EOS as requested
        )

        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=str(req_id),
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        assert final_output is not None

        return final_output.outputs[0].text

def generate_random_string(length: int) -> str:
    """Generate a random string of specified length"""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

def generate_random_strings(length: int, group_size: int) -> List[str]:
    """Generate a list of random strings of specified length and group size"""
    return [generate_random_string(length) for _ in range(group_size)]

def generate_random_string_groups(length: int, number_of_groups: int, group_size: int) -> List[List[str]]:
    """Generate a list of random strings of specified length and number of groups"""
    return [generate_random_strings(length, group_size) for _ in range(number_of_groups)]


async def run_single_request(model: ModelWrapper, prompt: str, max_tokens: int) -> RequestMetrics:
    """Run a single request and measure its latency"""
    start_time = time.time()
    response = await model.generate(prompt, max_tokens)
    end_time = time.time()
    
    latency = end_time - start_time
    tokens_generated = len(response.split())  # Approximate token count
    
    return RequestMetrics(latency=latency, tokens_generated=tokens_generated)

async def run_concurrent_requests(model: ModelWrapper, prompts: List[str], max_tokens: int) -> RequestMetrics:
    """Run multiple requests concurrently and measure their latencies"""
    start_time = time.time()
    tasks = [model.generate(prompt, max_tokens) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    end_time = time.time()

    latency = end_time - start_time
    tokens_generated = sum(len(response.split()) for response in responses)
    return RequestMetrics(latency=latency, tokens_generated=tokens_generated)


async def run_experiment(model: ModelWrapper, model_name: str, num_requests: int = 20, 
                        input_length: int = 100, output_length: int = 50) -> ExperimentResults:
    """Run the full experiment for a single model"""
    print(f"\n🔄 Running experiment for {model_name}...")
    
    # Generate test prompts
    prompts = [generate_random_string(input_length) for _ in range(num_requests)]
    
    # Record start time for end-to-end measurement
    experiment_start = time.time()
    
    # Run all requests concurrently
    tasks = [run_single_request(model, prompt, output_length) for prompt in prompts]
    metrics = await asyncio.gather(*tasks)
    
    # Record end time
    experiment_end = time.time()
    
    # Calculate results
    individual_latencies = [m.latency for m in metrics]
    end_to_end_latency = experiment_end - experiment_start
    throughput = num_requests / end_to_end_latency
    total_tokens = sum(m.tokens_generated for m in metrics)
    tokens_per_second = total_tokens / end_to_end_latency
    
    return ExperimentResults(
        individual_latencies=individual_latencies,
        end_to_end_latency=end_to_end_latency,
        throughput=throughput,
        tokens_per_second=tokens_per_second
    )

async def run_concurrent_experiment(model: ModelWrapper, model_name: str, experiment_size: int = 20, 
                        input_length: int = 100, output_length: int = 50, concurrent_requests: int = 10) -> ExperimentResults:
    """Run the full experiment for a single model with concurrent requests"""
    print(f"\n🔄 Running experiment for {model_name}...")
    
    # Generate test prompts
    prompt_groups = generate_random_string_groups(input_length, experiment_size, concurrent_requests)

    prompt_group_latencies = []
    prompt_group_tokens_generated = []
    for prompt_group in prompt_groups:
        metrics = await run_concurrent_requests(model, prompt_group, output_length)
        prompt_group_latencies.append(metrics.latency)
        prompt_group_tokens_generated.append(metrics.tokens_generated)
    
    end_to_end_latency = statistics.mean(prompt_group_latencies)
    throughput = experiment_size / end_to_end_latency
    tokens_per_second = sum(prompt_group_tokens_generated) / end_to_end_latency
    return ExperimentResults(individual_latencies=prompt_group_latencies, end_to_end_latency=end_to_end_latency, throughput=throughput, tokens_per_second=tokens_per_second)


def print_dashboard(langchain_results: ExperimentResults, vllm_results: ExperimentResults):
    """Print a compact dashboard with experiment results"""
    print("\n" + "="*80)
    print("🚀 LANGCHAIN vs vLLM PERFORMANCE COMPARISON")
    print("="*80)
    
    def format_stats(values: List[float], unit: str = "") -> str:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return f"{mean:.3f} ± {std:.3f}{unit}"
    
    # Individual request latencies
    print(f"\n📊 INDIVIDUAL REQUEST LATENCIES")
    print(f"{'Metric':<25} {'LangChain':<20} {'vLLM':<20} {'Winner':<10}")
    print("-" * 75)
    
    lc_lat = format_stats(langchain_results.individual_latencies, "s")
    vllm_lat = format_stats(vllm_results.individual_latencies, "s")
    lat_winner = "vLLM" if statistics.mean(vllm_results.individual_latencies) < statistics.mean(langchain_results.individual_latencies) else "LangChain"
    print(f"{'Avg Request Latency':<25} {lc_lat:<20} {vllm_lat:<20} {lat_winner:<10}")
    
    # End-to-end performance
    print(f"\n⚡ END-TO-END PERFORMANCE")
    print(f"{'Metric':<25} {'LangChain':<20} {'vLLM':<20} {'Winner':<10}")
    print("-" * 75)
    
    lc_e2e = f"{langchain_results.end_to_end_latency:.3f}s"
    vllm_e2e = f"{vllm_results.end_to_end_latency:.3f}s"
    e2e_winner = "vLLM" if vllm_results.end_to_end_latency < langchain_results.end_to_end_latency else "LangChain"
    print(f"{'Total Time':<25} {lc_e2e:<20} {vllm_e2e:<20} {e2e_winner:<10}")
    
    lc_throughput = f"{langchain_results.throughput:.2f} req/s"
    vllm_throughput = f"{vllm_results.throughput:.2f} req/s"
    throughput_winner = "vLLM" if vllm_results.throughput > langchain_results.throughput else "LangChain"
    print(f"{'Throughput':<25} {lc_throughput:<20} {vllm_throughput:<20} {throughput_winner:<10}")
    
    lc_tokens = f"{langchain_results.tokens_per_second:.2f} tok/s"
    vllm_tokens = f"{vllm_results.tokens_per_second:.2f} tok/s"
    tokens_winner = "vLLM" if vllm_results.tokens_per_second > langchain_results.tokens_per_second else "LangChain"
    print(f"{'Token Generation':<25} {lc_tokens:<20} {vllm_tokens:<20} {tokens_winner:<10}")
    
    # Performance ratios
    print(f"\n📈 PERFORMANCE RATIOS")
    print(f"{'Metric':<25} {'Ratio (vLLM/LangChain)':<25}")
    print("-" * 50)
    
    latency_ratio = statistics.mean(vllm_results.individual_latencies) / statistics.mean(langchain_results.individual_latencies)
    throughput_ratio = vllm_results.throughput / langchain_results.throughput
    tokens_ratio = vllm_results.tokens_per_second / langchain_results.tokens_per_second
    
    print(f"{'Latency Ratio':<25} {latency_ratio:.3f}x")
    print(f"{'Throughput Ratio':<25} {throughput_ratio:.3f}x")
    print(f"{'Token Gen Ratio':<25} {tokens_ratio:.3f}x")
    
    print("\n" + "="*80)


async def main():
    """Main experiment runner"""
    print("🚀 Starting LangChain vs vLLM Performance Experiment")
    
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Initialize models
    print("\n🔧 Initializing models...")
    
    try:
        temperature = 0.8


        # LangChain model (localhost:9999)
        langchain_model = LangChainWrapper(
            model_name=model_name,
            base_url="http://localhost:9999/v1",
            temperature=temperature
        )
        print("✅ LangChain model initialized")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        vllm_model = VLLMWrapper(
            model_name=model_name,
            temperature=temperature
        )
        print("✅ vLLM model initialized")

        experiment_size = 50
        input_length = 100
        output_length = 100
        concurrent_requests = 20
        print(f"📝 Configuration: {experiment_size} trials, {input_length} char input, {output_length} token output, {concurrent_requests} concurrent requests")
        
        # Run experiments
        langchain_results = await run_concurrent_experiment(
            langchain_model, "LangChain", 
            experiment_size=experiment_size, input_length=input_length, output_length=output_length, concurrent_requests=concurrent_requests
        )
        
        vllm_results = await run_concurrent_experiment(
            vllm_model, "vLLM",
            experiment_size=experiment_size, input_length=input_length, output_length=output_length, concurrent_requests=concurrent_requests
        )
        
        # Display results
        print_dashboard(langchain_results, vllm_results)
        
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        raise


async def test_langchain_only():
    """Test just the LangChain wrapper if vLLM is not available"""
    print("🧪 Testing LangChain wrapper only...")
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    langchain_model = LangChainWrapper(
        model_name=model_name,
        base_url="http://localhost:9999/v1",
        temperature=0.0
    )
    
    # Test with a small number of requests
    results = await run_experiment(
        langchain_model, "LangChain", 
        num_requests=3, input_length=50, output_length=20
    )
    
    print(f"✅ Test completed!")
    print(f"  - Average latency: {statistics.mean(results.individual_latencies):.3f}s")
    print(f"  - Throughput: {results.throughput:.2f} req/s")


if __name__ == "__main__":
    # Check if we should run a quick test or full experiment
    import sys
    load_dotenv()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_langchain_only())
    else:
        asyncio.run(main())
