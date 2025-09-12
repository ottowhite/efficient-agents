from langchain_openai import OpenAI
import asyncio

model = OpenAI(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:9999/v1",
    api_key="EMPTY",  # type: ignore
    streaming=False,
    temperature=0.9
)

large_prompt = "A" * 10000

async def make_request():
    output = await model.ainvoke(large_prompt, max_tokens=1000, extra_body={"min_tokens": 1000, "ignore_eos": True, "include_stop_str_in_output": True})
    return output

async def main():
    output = await asyncio.gather(*[make_request() for _ in range(100)])

asyncio.run(main())