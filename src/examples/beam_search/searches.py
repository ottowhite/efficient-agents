from src.examples.beam_search.models import PRM
from src.examples.beam_search.models import LLM
from src.utils import flatten
import asyncio
import numpy as np
from asyncio import create_task
from copy import deepcopy
from src.examples.beam_search.thought import Thought

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

        # Generate all new steps without scoring first
        step_futures = [create_task(self._generate_next_step(thought)) for _ in range(self.search_width)]
        generated_steps = await asyncio.gather(*step_futures)
        
        # Create new thoughts and deduplicate using set
        new_thoughts_set = set()
        for step in generated_steps:
            new_thought = thought.copy_with_added_step(step)
            new_thoughts_set.add(new_thought)
        
        # Convert back to list for scoring
        unique_thoughts = list(new_thoughts_set)
        
        # Score all unique thoughts in parallel
        score_futures = [create_task(self.prm.generate_score(thought.get_prm_conversation())) for thought in unique_thoughts]
        scores = await asyncio.gather(*score_futures)
        
        # Associate scores with thoughts
        for thought, score in zip(unique_thoughts, scores):
            thought.score_last_step(score)
        
        # If we have fewer thoughts than search_width, duplicate with deepcopy
        while len(unique_thoughts) < self.search_width:
            if unique_thoughts:
                # Duplicate the best thought
                best_thought = max(unique_thoughts, key=lambda t: t.scores[-1])
                duplicate = deepcopy(best_thought)
                unique_thoughts.append(duplicate)
            else:
                break

        # Process thoughts to separate completed from incomplete
        for new_thought in unique_thoughts:
            if "final answer is" in new_thought.steps[-1].lower():
                self.completed_thoughts.append(new_thought)
            else:
                incomplete_thoughts.append(new_thought)
    
        incomplete_thoughts.sort(key=lambda thought: thought.scores[-1], reverse=True)
        return incomplete_thoughts[:self.select_top_k]
    
    async def _generate_next_step(self, thought: Thought) -> str:
        """Generate next step for a thought without scoring."""
        conversation_str = thought_to_str(self.llm, thought)
        next_step = await self.llm.ainvoke(conversation_str)
        return next_step
    
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

        # Generate all new steps without scoring first
        step_futures = [create_task(self._generate_next_step(thought)) for _ in range(self.search_width)]
        generated_steps = await asyncio.gather(*step_futures)
        
        # Create new thoughts and deduplicate using set
        new_thoughts_set = set()
        for step in generated_steps:
            new_thought = thought.copy_with_added_step(step)
            new_thoughts_set.add(new_thought)
        
        # Convert back to list for scoring
        unique_thoughts = list(new_thoughts_set)
        
        # Score all unique thoughts in parallel
        score_futures = [create_task(self._score_thought(thought)) for thought in unique_thoughts]
        scores = await asyncio.gather(*score_futures)
        
        # Associate scores with thoughts
        for thought, score in zip(unique_thoughts, scores):
            thought.score_last_step(score)
        
        # If we have fewer thoughts than search_width, duplicate with deepcopy
        while len(unique_thoughts) < self.search_width:
            if unique_thoughts:
                # Duplicate the best thought
                best_thought = max(unique_thoughts, key=lambda t: t.cumulative_score())
                duplicate = deepcopy(best_thought)
                unique_thoughts.append(duplicate)
            else:
                break

        # Process thoughts to separate completed from incomplete
        for new_thought in unique_thoughts:
            if "final answer is" in new_thought.steps[-1].lower():
                self.completed_thoughts.append(new_thought)
            else:
                incomplete_thoughts.append(new_thought)
    
        incomplete_thoughts.sort(key=lambda thought: thought.cumulative_score(), reverse=True)
        return incomplete_thoughts
    
    async def _generate_next_step(self, thought: Thought) -> str:
        """Generate next step for a thought without scoring."""
        conversation_str = thought_to_str(self.llm, thought)
        next_step = await self.llm.ainvoke(conversation_str)
        return next_step
    
    async def _score_thought(self, thought: Thought) -> float:
        """Score a thought using the PRM."""
        score = await self.prm.generate_score(thought.get_prm_conversation())
        return score
    
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