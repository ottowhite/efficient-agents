import yappi
import asyncio
from dotenv import load_dotenv
import time
from asyncio import create_task
from src.utils import custom_chat_template, dataset_name
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
from src.examples.beam_search.models import LLM, PRM, Tokenizer

from src.config import Config
from src.sal.score import score
from src.sal.qwen_math_parser import extract_answer, math_equal

from src.examples.beam_search.searches import BeamSearch
from src.examples.beam_search.scoring import create_dataset_from_results, calculate_and_print_accuracies

async def run_beam_search_with_semaphore(problem: str, semaphore: asyncio.Semaphore, llm: LLM, prm: PRM):
    async with semaphore:
        beam_search = BeamSearch(
            problem=problem,
            llm=llm,
            prm=prm,
            search_width=4,
            select_top_k=1,
            max_iterations=40,
            # sampling_window_size=None
        )

        return await beam_search.run()

async def main():
    load_dotenv()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = Tokenizer(model_name, custom_chat_template)
    llm = LLM(
        model_name=model_name,
        base_url="http://localhost:9999/v1",
        temperature=0.8,
        tokenizer=tokenizer
    )

    prm = PRM(
        model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        base_url="http://localhost:8888/v1"
    )

    dataset = load_dataset(dataset_name, split="test")

    semaphore = asyncio.Semaphore(100)
    time_start = time.time()
    # Convert dataset to list and access elements properly
    dataset_list = list(dataset)[:50]
    beam_search_tasks = [create_task(run_beam_search_with_semaphore(sample["problem"], semaphore, llm, prm)) for sample in dataset_list]

    for future in tqdm(
        asyncio.as_completed(beam_search_tasks), total=len(beam_search_tasks), desc="Problems"
    ):
        await future
    
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    answer_thought_lists = [await future for future in beam_search_tasks]

    # Transform results into dataset format matching beam_search_async.py
    result_dataset = create_dataset_from_results(dataset_list, answer_thought_lists)

    # Create config for scoring (using defaults that match the search parameters)
    # Use the first result to determine search width, fallback to 4
    search_width = len(answer_thought_lists[0]) if answer_thought_lists and answer_thought_lists[0] else 4
    print(f"Generated answers per problem: {search_width}")
    config = Config(
        search_width=search_width,
        num_proc=1,  # Keep simple for this example
        search_algorithm="beam_search"
    )
    
    # Apply scoring using the same function as beam_search_async
    scored_dataset = score(result_dataset, config)
    
    print(f"Scoring completed. Dataset now contains prediction fields for analysis.")
    print(f"Total problems processed: {len(scored_dataset)}")
    
    # Calculate and print all accuracy metrics
    calculate_and_print_accuracies(scored_dataset)

if __name__ == "__main__":
    run_yappi = False
    if run_yappi:
        yappi.set_clock_type("WALL")
        with yappi.run():
            asyncio.run(main())
        yappi.get_func_stats().print_all()
    else:
        asyncio.run(main())
