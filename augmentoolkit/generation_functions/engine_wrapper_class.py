import asyncio
import uuid
from openai import AsyncOpenAI

try:
    from aphrodite import (
        EngineArgs,
        AphroditeEngine,
        SamplingParams,
        AsyncAphrodite,
        AsyncEngineArgs,
    )
except:
    print("Aphrodite not installed; stick to Llama CPP or API modes")

def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(self, model, 
                 api_key=None, 
                 base_url=None, 
                 mode="api", # can be one of api, aphrodite, llama.cpp
                 quantization="gptq", # only needed if using aphrodite mode
                ):
        if mode == "aphrodite":
            engine_args = AsyncEngineArgs(
                model=model,
                quantization=quantization,
                engine_use_ray=False,
                disable_log_requests=True,
                max_model_len=12000,
                dtype="float16"
            )
            self.engine = AsyncAphrodite.from_engine_args(engine_args)
        self.mode = mode
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        # print("\n\nSETTINGS DUMP\n\n")
        # print(self.model)
        # print(prompt)
        # print(sampling_params["temperature"])
        # print(sampling_params["top_p"])
        # print(sampling_params["max_tokens"])
        if self.mode == "aphrodite":
            aphrodite_sampling_params = SamplingParams(**sampling_params)
            request_id = make_id()
            outputs = []
            # self.engine.add_request(request_id,prompt,sampling_params) #old sync code
            final_output = None
            async for request_output in self.engine.generate(
                prompt, aphrodite_sampling_params, request_id
            ):
                outputs.append(request_output.outputs[0].text)
                final_output = request_output

            # full_output = "".join(outputs)
            return final_output.prompt + final_output.outputs[0].text
        
        if self.mode == "api":
            completion = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                stop=sampling_params["stop"],
                max_tokens=sampling_params["max_tokens"],
            )
            completion = completion.choices[0].text
            return prompt + completion
    
    async def submit_chat(
    self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            stop=sampling_params["stop"],
            max_tokens=sampling_params["max_tokens"],
        )
        completion = completion.choices[0].message.content
        return completion
