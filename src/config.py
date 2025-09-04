import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    new_configuration: dict[str, object] = field(default_factory=dict)
    prm_backend: Literal["vllm", "sglang"] = "vllm"
    prm_score_caching: bool = False
    llm_prefix_caching: bool = False
    prm_prefix_caching: bool = False
    final_search_iter_eos: bool = True
    continuous_problem_batching: bool = False
    cpb_min_batch_size: int = 16
    max_model_length: Optional[int] = None
    llm_gpu_memory: float = 0.3
    llm_chunked_prefill: bool = False
    prm_gpu_memory: float = 0.9
    prm_chunked_prefill: bool = False
    prm_max_batched_tokens: Optional[int] = None
    execution_method: str = ""

    log_tree: bool = False

    search_algorithm: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    llm: str = "meta-llama/Llama-3.2-1B-Instruct"
    llm_max_batched_tokens: Optional[int] = None
    prm: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

    # Output Related Options
    output_dir: Optional[str] = None
    num_proc: Optional[int] = None
    push_to_hub: bool = False
    hub_dataset_id: Optional[str] = None
    hub_dataset_private: bool = False
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    difficulty: int = 0  # 1, 2, 3, 4, 5 # 0 means all difficulties
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: Optional[str] = None
    dataset_split: str = "test"
    dataset_start: Optional[int] = None
    dataset_end: Optional[int] = None
    num_problems: int = 0

    # Chat template related options
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    # Search Related Options
    search_width: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: Literal["last", "min", "prod"] = "last"

    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 1

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False


def update_from_json(config: Config, pa: str) -> Config:
    with open(pa, "r") as f:
        json_config = json.load(f)

    new_keys: list[str] = []
    for key, value in json_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            new_keys.append(key)
            config.new_configuration[key] = value
    logger.warning(f"Adding new keys to config.new_configuration for {new_keys}")

    return config
