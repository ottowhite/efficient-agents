from typing import Optional
from src.examples.beam_search.models import LLM

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
        if len(self.scores) == 0:
            return 0.0
        
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
    
    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "steps": self.steps,
            "scores": self.scores
        }
    
    def __str__(self) -> str:
        steps_str = "\n".join(self.steps)
        last_score = self.scores[-1] if self.scores else None
        return f"Problem:\n\n{self.problem}\n\nSteps:\n{steps_str}\nScores:{self.scores}\nLast Score:{last_score}"
