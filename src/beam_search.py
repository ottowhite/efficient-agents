import yappi
import asyncio
from dotenv import load_dotenv
import re
import time
import sys
import os
from asyncio import create_task
from transformers import AutoTokenizer
from copy import deepcopy
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAI
from utils import system_prompt, custom_chat_template, dataset_name
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
from utils import flatten

from config import Config
from sal.score import score
from sal.qwen_math_parser import extract_answer, math_equal

class Tokenizer:
    def __init__(self, model_name: str, chat_template: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.chat_template = chat_template

    def apply_chat_template(self, conversation: list[dict]) -> str:
        result = self.tok.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        # Ensure we return a string
        if isinstance(result, str):
            return result
        else:
            raise ValueError(f"Expected string from apply_chat_template, got {type(result)}")

class Thought:
    def __init__(self, problem: str, steps: Optional[list[str]] = None, scores: Optional[list[float]] = None):
        self.problem = problem
        self.steps = steps if steps is not None else []
        self.scores = scores if scores is not None else []
        self.score_mode = "last"

    def copy_with_added_step(self, step: str):
        return Thought(self.problem, self.steps + [step], self.scores.copy())

    def score_last_step(self, score: float):
        """Append the provided score to the list of scores."""
        self.scores.append(score)
    
    def cumulative_score(self) -> float:
        if self.score_mode == "cumulative":
            """Calculate cumulative score as product of all step scores multiplied by number of steps."""
            if not self.scores:
                return 0.0
        
            # Calculate product of all scores
            product = 1.0
            for score in self.scores:
                product *= score
        
            # Multiply by number of steps
            return product * len(self.scores)
        elif self.score_mode == "last":
            return self.scores[-1]
        elif self.score_mode == "exponential_moving_average":
            if len(self.scores) == 0:
                return 0.0

            alpha = 0.3
            curr_score = self.scores[0]
            for score in self.scores[1:]:
                curr_score = alpha * score + (1 - alpha) * curr_score

            return curr_score
        else:
            raise ValueError(f"Invalid score mode: {self.score_mode}")
    
    def get_llm_conversation(self) -> list[dict]:
        conversation = [
            {"role": "system", "content": LLM.get_system_prompt()},
            {"role": "user", "content": self.problem}
        ]

        if len(self.steps) > 0:
            conversation.append({"role": "assistant", "content": "\n\n".join(self.steps) + "\n\n"})

        return conversation

    def get_prm_conversation(self) -> list[dict]:
        assert len(self.steps) > 0, "Cannot generate PRM conversation without steps"

        question_and_first_step = self.problem + "\n\n" + self.steps[0]
        conversation = [
            {"role": "user", "content": question_and_first_step}
        ]

        # For all intermediate steps (excluding the last), add the PRM label then the next step
        for i, step in enumerate(self.steps[1:]):
            if i != len(self.steps) - 1:
                # Add the step, alongside the assistant's previous response
                conversation.append({"role": "assistant", "content": "\n\n+"})
                conversation.append({"role": "user", "content": step})
            else:
                # Add the final step, without the assistant's response, which will now be scored
                conversation.append({"role": "user", "content": step})

        return conversation
    
    def __str__(self) -> str:
        steps_str = "\n".join(self.steps)
        last_score = self.scores[-1] if self.scores else None
        return f"Problem:\n\n{self.problem}\n\nSteps:\n{steps_str}\nScores:{self.scores}\nLast Score:{last_score}"

class LLM:
    def __init__(self, model_name: str, base_url: str, temperature: float, tokenizer: Tokenizer):
        self.model = OpenAI(
            model=model_name,
            base_url=base_url,
            api_key="EMPTY",  # type: ignore
            streaming=True,
            temperature=temperature
        )
        self.tokenizer = tokenizer

    async def ainvoke(self, thought: Thought) -> str:
        conversation_str = self.tokenizer.apply_chat_template(thought.get_llm_conversation())

        if len(thought.steps) > 0:
            # Deleting the eot_id is important for the LLM to pick up decoding where it left off.
            conversation_str = conversation_str.replace("<|eot_id|>", "")
        elif len(thought.steps) == 0:
            # We should kick off the assistant generation
            conversation_str = conversation_str.replace("<|eot_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

        return await self.model.ainvoke(conversation_str, stop=["\n\n"])

    @staticmethod
    def get_system_prompt():
        return system_prompt

class PRM:
    def __init__(self, model_name: str, base_url: str):
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="EMPTY",  # type: ignore
            streaming=True,
        )
        self.num_top_logprobs = 5

    async def generate_score(self, conversation: list[dict]) -> float:
        prm_output = await self.model.ainvoke(
            conversation,
            stop=["\n\n"],
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.num_top_logprobs
        )

        plus_logprobs = None
        minus_logprobs = None
        assert len(prm_output.response_metadata["logprobs"]["content"]) == 1
        token_metadata = prm_output.response_metadata["logprobs"]["content"][0]
        for top_logprobs in token_metadata["top_logprobs"]:
            if top_logprobs["token"] == "+":
                plus_logprobs = top_logprobs["logprob"]
            elif top_logprobs["token"] == "-":
                minus_logprobs = top_logprobs["logprob"]

        assert not (plus_logprobs is None or minus_logprobs is None), "No logprobs found"

        return np.exp(plus_logprobs) / (np.exp(plus_logprobs) + np.exp(minus_logprobs))

# TODO: Do the proper beam search selection where we select top m/n of the beam search results
# TODO: Generalise all scoring and sorting so it doesn't just assume the last score is the best
# TODO: Add the custom chat template, it's currently not being used
# TODO: Implement PRM Cache
# TODO: Try implementing DFS, there could be promise in making the scheduler aware of the different levels of promising-ness of the different thoughts.
#           Interesting implications for deep research.
class BeamSearch:
    def __init__(self, problem: str, llm: LLM, prm: PRM, search_width: int, select_top_k: int, max_iterations: int):
        self.problem = problem
        self.llm = llm
        self.prm = prm
        self.search_width = search_width
        self.select_top_k = select_top_k
        self.completed_thoughts = []
        self.max_iterations = max_iterations
    
    async def run(self):
        thoughts = [Thought(self.problem)]
        for i in range(self.max_iterations):
            thoughts = await self.expand_thoughts(thoughts)
            # print("Last step of best thought: ", thoughts[0].steps[-1])
            if self.is_finished():
                # In this case, thoughts just holds the top k answers
                break

        return thoughts

    async def expand_thoughts(
            self,
            thoughts: list[Thought]
        ) -> list[Thought]:

        thought_futures = [create_task(self.expand_thought(thought)) for thought in thoughts]
        expanded_thoughts = flatten(await asyncio.gather(*thought_futures))

        if self.is_finished():
            self.completed_thoughts.sort(key=lambda thought: thought.scores[-1], reverse=True)
            return self.completed_thoughts[:self.search_width]

        # Sort by score
        expanded_thoughts.sort(key=lambda thought: thought.scores[-1], reverse=True)

        return expanded_thoughts


    async def expand_thought(self, thought: Thought) -> list[Thought]:
        incomplete_thoughts = []

        new_thought_futures = [create_task(self.extend_and_score_thought(thought)) for _ in range(self.search_width)]
        new_thoughts = await asyncio.gather(*new_thought_futures)

        for new_thought in new_thoughts:
            # If the new thought is finished, add it to the completed thoughts, otherwise add it to the possible thoughts for later expansion
            if "final answer is" in new_thought.steps[-1].lower():
                self.completed_thoughts.append(new_thought)
                continue
            else:
                incomplete_thoughts.append(new_thought)
    
        incomplete_thoughts.sort(key=lambda thought: thought.scores[-1], reverse=True)

        return incomplete_thoughts[:self.select_top_k]
    
    async def extend_and_score_thought(self, thought: Thought) -> Thought:
        next_step = await self.llm.ainvoke(thought)

        new_thought = thought.copy_with_added_step(next_step)

        score = await self.prm.generate_score(
            new_thought.get_prm_conversation()
        )
        new_thought.score_last_step(score)

        return new_thought
    
    def is_finished(self) -> bool:
        return len(self.completed_thoughts) >= self.search_width

class DFS:
    def __init__(self, problem: str, llm: LLM, prm: PRM, search_width: int, select_top_k: int, max_iterations: int, sampling_window_size: int | None = None):
        self.problem = problem
        self.llm = llm
        self.prm = prm
        self.search_width = search_width
        self.select_top_k = select_top_k
        self.sampling_window_size = sampling_window_size
        self.completed_thoughts = []
        self.max_iterations = max_iterations

        if self.sampling_window_size is not None:
            assert self.sampling_window_size >= self.select_top_k, "Sampling window size must be greater than or equal to select top k"

    async def run(self):
        candidate_thoughts = [Thought(self.problem)]
        
        for i in range(self.max_iterations):
            if self.is_finished():
                break
                
            # Sort candidate thoughts by cumulative score before selecting top k for expansion
            candidate_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)

            # Select top k most promising thoughts to expand
            if self.sampling_window_size is not None:
                sampling_window_size = min(self.sampling_window_size, len(candidate_thoughts))

                potential_thoughts_to_expand = candidate_thoughts[:sampling_window_size]
                thoughts_to_expand_scores = [thought.cumulative_score() for thought in potential_thoughts_to_expand]

                # Use softmax to compute probabilities for sampling
                scores = np.array(thoughts_to_expand_scores)
                exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
                thoughts_to_expand_probs = exp_scores / np.sum(exp_scores)
                chosen_indices = np.random.choice(
                    len(potential_thoughts_to_expand),
                    size=self.select_top_k,
                    replace=False if self.select_top_k <= len(potential_thoughts_to_expand) else True,
                    p=thoughts_to_expand_probs
                )

                thoughts_to_expand = []
                thoughts_not_expanded = []
                for i, thought in enumerate(potential_thoughts_to_expand):
                    if i in chosen_indices:
                        thoughts_to_expand.append(thought)
                    else:
                        thoughts_not_expanded.append(thought)

                thoughts_not_expanded.extend(candidate_thoughts[sampling_window_size:])
            else:
                thoughts_to_expand = candidate_thoughts[:self.select_top_k]
                thoughts_not_expanded = candidate_thoughts[self.select_top_k:]
            
            # Expand the selected thoughts
            expanded_thoughts = await self.expand_thoughts(thoughts_to_expand)
            
            # Merge expanded thoughts with the ones we didn't expand
            candidate_thoughts = expanded_thoughts + thoughts_not_expanded

        # Return final thoughts sorted by cumulative score
        if self.completed_thoughts:
            self.completed_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)
            return self.completed_thoughts[:self.search_width]
        else:
            candidate_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)
            return candidate_thoughts[:self.search_width]

    async def expand_thoughts(
            self,
            thoughts: list[Thought]
        ) -> list[Thought]:

        thought_futures = [create_task(self.expand_thought(thought)) for thought in thoughts]
        expanded_thoughts = flatten(await asyncio.gather(*thought_futures))

        # Sort by cumulative score
        expanded_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)

        return expanded_thoughts

    async def expand_thought(self, thought: Thought) -> list[Thought]:
        incomplete_thoughts = []

        new_thought_futures = [create_task(self.extend_and_score_thought(thought)) for _ in range(self.search_width)]
        new_thoughts = await asyncio.gather(*new_thought_futures)

        for new_thought in new_thoughts:
            # If the new thought is finished, add it to the completed thoughts, otherwise add it to the possible thoughts for later expansion
            if "final answer is" in new_thought.steps[-1].lower():
                self.completed_thoughts.append(new_thought)
                continue
            else:
                incomplete_thoughts.append(new_thought)
    
        incomplete_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)

        return incomplete_thoughts
    
    async def extend_and_score_thought(self, thought: Thought) -> Thought:
        next_step = await self.llm.ainvoke(thought)

        new_thought = thought.copy_with_added_step(next_step)

        score = await self.prm.generate_score(
            new_thought.get_prm_conversation()
        )
        new_thought.score_last_step(score)

        return new_thought
    
    def is_finished(self) -> bool:
        return len(self.completed_thoughts) >= self.search_width

    
def print_conversation(conversation: list[dict]):
    for message in conversation:
        print(message["role"], ":", message["content"])
    print("\n\n")

def create_dataset_from_results(dataset_list: list, answer_thoughts: list) -> Dataset:
    """Transform beam search results into dataset format matching beam_search_async.py"""
    updated_samples = []
    for problem_data, problem_thoughts in zip(dataset_list, answer_thoughts):
        # Create completions and scores from thoughts
        completions = []
        scores = []
        
        for thought in problem_thoughts:
            # Join all steps into a single completion text
            completion_text = "\n\n".join(thought.steps)
            completions.append(completion_text)
            # Use the step-wise scores
            scores.append(thought.scores)
        
        # Find best completion based on last score (matching beam_search_async logic)
        if completions:
            best_idx = max(range(len(problem_thoughts)), key=lambda i: problem_thoughts[i].scores[-1] if problem_thoughts[i].scores else 0)
            pred = completions[best_idx]
        else:
            pred = ""
        
        # Create updated sample with required fields
        updated_sample = dict(problem_data)  # Copy original data
        updated_sample["completions"] = completions
        updated_sample["scores"] = scores  
        updated_sample["pred"] = pred
        updated_samples.append(updated_sample)

    return Dataset.from_list(updated_samples)



def calculate_and_print_accuracies(scored_dataset: Dataset) -> None:
    """Calculate and print accuracy metrics for all prediction fields generated by the score function"""
    
    # Find all prediction fields in the dataset
    sample = scored_dataset[0]
    pred_fields = [key for key in sample.keys() if key.startswith('pred_')]
    
    # Group by N value and sort for organized display
    pred_groups = {}
    for field in pred_fields:
        if '@' in field:
            strategy, n_str = field.split('@')
            n = int(n_str)
            if n not in pred_groups:
                pred_groups[n] = {}
            pred_groups[n][strategy] = field
    
    print("\n" + "="*80)
    print("ACCURACY METRICS BY NUMBER OF COMPLETIONS")
    print("="*80)
    
    # Sort by N value (powers of 2: 1, 2, 4, 8, ...)
    for n in sorted(pred_groups.keys()):
        print(f"\nUsing Top {n} Completion{'s' if n > 1 else ''}:")
        print("-" * 40)
        
        # Display in consistent order: weighted, majority, naive
        strategy_order = ['pred_weighted', 'pred_maj', 'pred_naive']
        strategy_names = {'pred_weighted': 'Weighted', 'pred_maj': 'Majority', 'pred_naive': 'Naive (Best)'}
        
        for strategy in strategy_order:
            if strategy in pred_groups[n]:
                pred_field = pred_groups[n][strategy]
                correct_count = 0
                total_count = 0
                
                for sample in scored_dataset:
                    # Get correct answer
                    correct_answer = str(sample["answer"])  # type: ignore
                    
                    # Get prediction
                    prediction = sample.get(pred_field)  # type: ignore
                    if prediction:
                        pred_str = str(prediction)
                        
                        # Extract answer from prediction using the robust prebuilt function
                        predicted_answer = extract_answer(pred_str, "math")
                        
                        # Use the robust math_equal function for comparison
                        if math_equal(predicted_answer, correct_answer):
                            correct_count += 1
                            
                    total_count += 1
                
                # Calculate and print accuracy
                if total_count > 0:
                    accuracy = (correct_count / total_count) * 100
                    strategy_name = strategy_names[strategy]
                    print(f"  {strategy_name:<12}: {accuracy:6.2f}% ({correct_count}/{total_count})")
                else:
                    print(f"  {strategy_names[strategy]:<12}: No valid predictions")
    
    print("\n" + "="*80)

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
        # base_url="http://localhost:9999/v1",
        base_url="http://localhost:30001/v1",
        temperature=0.8,
        tokenizer=tokenizer
    )

    prm = PRM(
        model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        # base_url="http://localhost:8888/v1"
        base_url="http://localhost:30000/v1"
    )

    dataset = load_dataset(dataset_name, split="test")

    semaphore = asyncio.Semaphore(100)
    time_start = time.time()
    # Convert dataset to list and access elements properly
    dataset_list = list(dataset)
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
