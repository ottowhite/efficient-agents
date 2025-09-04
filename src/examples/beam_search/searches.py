from typing import Optional
from src.examples.beam_search.models import PRM
from src.examples.beam_search.models import LLM
from src.utils import flatten
import asyncio
import numpy as np
from asyncio import create_task


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
        self.completed_thoughts: list[Thought] = []
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
        conversation_str = thought_to_str(self.llm, thought)
        next_step = await self.llm.ainvoke(conversation_str)

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
        self.completed_thoughts: list[Thought] = []
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
        conversation_str = thought_to_str(self.llm, thought)
        next_step = await self.llm.ainvoke(conversation_str)

        new_thought = thought.copy_with_added_step(next_step)

        score = await self.prm.generate_score(
            new_thought.get_prm_conversation()
        )
        new_thought.score_last_step(score)

        return new_thought
    
    def is_finished(self) -> bool:
        return len(self.completed_thoughts) >= self.search_width

def thought_to_str(llm: LLM, thought: Thought) -> str:
    conversation_str = llm.tokenizer.apply_chat_template(thought.get_llm_conversation())

    if len(thought.steps) > 0:
        # Deleting the eot_id is important for the LLM to pick up decoding where it left off.
        conversation_str = conversation_str.replace("<|eot_id|>", "")
    elif len(thought.steps) == 0:
        # We should kick off the assistant generation
        conversation_str = conversation_str.replace("<|eot_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    return conversation_str