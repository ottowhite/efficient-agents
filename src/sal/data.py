import logging
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset

from config import Config

logger = logging.getLogger()


# TODO: replace every sample["problem"] with get_prob_field(sample)
def get_prob_field(sample):
    keys = ['problem', 'question']
    for k in keys:
        if k in sample:
            return sample[k]


def get_dataset(config: Config, print_problems: bool = False) -> Dataset:
    if config.dataset_name == 'openai/gsm8k':
        dataset = load_dataset('openai/gsm8k', 'main', split=config.dataset_split)
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))

    if config.num_problems > 0:  # means all problems
        if config.difficulty <= 0:  # 0 means all difficulties
            print(f"Selecting by num_problems: {config.num_problems}")
            dataset = dataset.select(range(min(len(dataset), config.num_problems)))
        else:
            print(
                f"Selecting by num_problems: {config.num_problems} and difficulty: {config.difficulty}"
            )
            dataset = dataset.filter(lambda x: x["level"] == config.difficulty)
            dataset = dataset.select(range(min(len(dataset), config.num_problems)))
    else:
        config.num_problems = len(dataset)

    if print_problems:
        try:
            _print_problems(dataset)
        except:
            pass

    return dataset


def _print_problems(dataset):
    difficulties = defaultdict(int)
    for i, data in enumerate(dataset):
        difficulties[data["level"]] += 1
    total = 0
    for difficulty in sorted(difficulties.keys()):
        total += difficulties[difficulty]
        print(f"Difficulty {difficulty}: {difficulties[difficulty]} problems")
    print(f"Total number of problems: {total}")


def save_dataset(dataset, config):
    if config.output_dir is None:
        config.output_dir = f"data/{config.llm}"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    dataset.to_json(
        f"{config.output_dir}/{config.search_algorithm}_completions.jsonl", lines=True
    )
    logger.info(
        f"Saved completions to {config.output_dir}/{config.search_algorithm}_completions.jsonl"
    )
