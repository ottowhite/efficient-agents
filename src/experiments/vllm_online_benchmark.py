import requests
import json
import argparse
import random
import string
import time
import statistics
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


def generate_random_prompt(length: int) -> str:
    """
    Generate a random string of specified length using ASCII characters.
    
    Args:
        length (int): Length of the random string to generate
    
    Returns:
        str: Random string of the specified length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))


def generate_text(prompt, max_tokens=100, temperature=0.7, top_p=0.9, model=None):
    """
    Send a generation request to vLLM API server.
    
    Args:
        prompt (str): The input prompt for text generation
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (0.0 to 2.0)
        top_p (float): Nucleus sampling parameter
        model (str): Model name (optional, uses server default if None)
    
    Returns:
        dict: Response from vLLM API containing generated text
    """
    url = "http://localhost:9999/v1/completions"
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "ignore_eos": True,
        "include_stop_str_in_output": True,
        "stream": False
    }
    
    # Add model parameter if specified
    if model:
        payload["model"] = model
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to vLLM API: {e}")
        return None


def process_batch(prompts: List[str], max_tokens: int) -> tuple[float, List[Dict[Any, Any]], float]:
    """
    Process a batch of prompts and measure the total time taken.
    
    Args:
        prompts (List[str]): List of prompts to process
        max_tokens (int): Maximum tokens to generate per prompt
    
    Returns:
        tuple: (total_time_seconds, list_of_responses)
    """
    start_time = time.time()
    
    # Use ThreadPoolExecutor to send requests concurrently
    results = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit all requests
        future_to_prompt = {executor.submit(generate_text, prompt, max_tokens): prompt 
                          for prompt in prompts}
        
        # Collect results
        for future in as_completed(future_to_prompt):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = sum(len(prompt) for prompt in prompts) + sum(len(result["choices"][0]["text"]) for result in results)
    tokens_per_second = total_tokens / total_time
    
    return total_time, results, tokens_per_second


def run_experiment(batch_size: int, input_length: int, output_length: int, num_trials: int) -> tuple[List[float], List[float]]:
    """
    Run the vLLM batching experiment for specified number of trials.
    
    Args:
        batch_size (int): Number of prompts per batch
        input_length (int): Length of each input prompt in characters
        output_length (int): Maximum output tokens to generate
        num_trials (int): Number of experimental trials to run
    
    Returns:
        List[float]: List of latencies (in seconds) for each trial
    """
    latencies = []
    tokens_per_seconds = []
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")
        
        # Generate random prompts for this trial
        prompts = [generate_random_prompt(input_length) for _ in range(batch_size)]
        
        # Process the batch and measure latency
        latency, results, tokens_per_second = process_batch(prompts, output_length)
        latencies.append(latency)
        tokens_per_seconds.append(tokens_per_second)
        
        # Check for any failed requests
        failed_count = sum(1 for result in results if result is None)
        if failed_count > 0:
            print(f"  Warning: {failed_count}/{batch_size} requests failed in trial {trial + 1}")
        
        print(f"  Trial {trial + 1} completed in {latency:.3f} seconds")
    
    return latencies, tokens_per_seconds


def save_results(latencies: List[float], tokens_per_seconds: List[float], results_path: str, batch_size: int, 
                input_length: int, output_length: int, num_trials: int):
    """
    Save experiment results to CSV file.
    
    Args:
        latencies (List[float]): List of latencies from all trials
        results_path (str): Path to save the results CSV file
        batch_size (int): Batch size used in experiment
        input_length (int): Input sequence length used
        output_length (int): Output sequence length used
        num_trials (int): Number of trials run
    """
    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    avg_tokens_per_second = statistics.mean(tokens_per_seconds)
    std_tokens_per_second = statistics.stdev(tokens_per_seconds) if len(tokens_per_seconds) > 1 else 0.0
    
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['batch_size', 'input_length', 'output_length', 'num_trials', 
                        'avg_latency_seconds', 'std_latency_seconds', 'avg_tokens_per_second', 'std_tokens_per_second'])
        
        # Write results
        writer.writerow([batch_size, input_length, output_length, num_trials, 
                        avg_latency, std_latency, avg_tokens_per_second, std_tokens_per_second])
    
    print(f"\nResults saved to: {results_path}")
    print(f"Average latency: {avg_latency:.3f} ± {std_latency:.3f} seconds")


def main():
    """Main function to run the vLLM batching experiment."""
    parser = argparse.ArgumentParser(description='vLLM Batch Processing Experiment')
    parser.add_argument('--batch_size', type=int, required=True,
                       help='Number of prompts to send in a single batch')
    parser.add_argument('--input_length', type=int, required=True,
                       help='Length of input sequence (number of random characters)')
    parser.add_argument('--output_length', type=int, required=True,
                       help='Maximum number of output tokens to generate')
    parser.add_argument('--num_trials', type=int, required=True,
                       help='Number of experimental trials to run')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to save CSV file containing results')
    
    args = parser.parse_args()
    
    print(f"Starting vLLM batch experiment:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input length: {args.input_length} characters")
    print(f"  Output length: {args.output_length} tokens")
    print(f"  Number of trials: {args.num_trials}")
    print(f"  Results path: {args.results_path}")
    print()
    
    # Run the experiment
    latencies, tokens_per_seconds = run_experiment(args.batch_size, args.input_length, 
                              args.output_length, args.num_trials)
    
    # Save results
    save_results(latencies, tokens_per_seconds, args.results_path, args.batch_size, 
                args.input_length, args.output_length, args.num_trials)


if __name__ == "__main__":
    main()
