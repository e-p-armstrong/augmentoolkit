# from aphrodite import (
#     EngineArgs,
#     AphroditeEngine,
#     SamplingParams,
#     AsyncAphrodite,
#     AsyncEngineArgs,
# )
import asyncio
import uuid
from openai import AsyncOpenAI

def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(self, model):
        # engine_args = AsyncEngineArgs(
        #     model=model,
        #     quantization=quantization,
        #     engine_use_ray=False,
        #     disable_log_requests=True,
        #     max_model_len=12000,
        #     dtype="float16"
        # )
        # self.engine = AsyncAphrodite.from_engine_args(engine_args)
        
        self.client = AsyncOpenAI(api_key="b6f1cbb729a28a019a7d899dd155843f21bb90c4d3bb566e43b1e7a94b44a576",base_url="https://api.together.xyz")
        self.model = model


    async def submit(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        completion = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            stop=sampling_params["stop"],
            max_tokens=sampling_params["max_tokens"]
        )
        completion = completion.choices[0].text
        return prompt + completion
