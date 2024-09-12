import asyncio
import uuid
from openai import AsyncOpenAI
import cohere
from httpx import Timeout

def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        mode="api",  # can be one of api, aphrodite, llama.cpp, cohere
        quantization="gptq",  # only needed if using aphrodite mode
    ):
        self.mode = mode
        self.model = model
        if mode == "cohere":
            self.client = cohere.AsyncClient(api_key=api_key)
        elif mode == "api":
            self.client = AsyncOpenAI(timeout=Timeout(timeout=5000.0, connect=10.0), api_key=api_key, base_url=base_url)

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
        if "n_predict" not in sampling_params:
            sampling_params["n_predict"] = sampling_params["max_tokens"]
        
        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        if self.mode == "api":
            timed_out = False
            completion = ""
            if use_min_p:
                stream = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    extra_body={"min_p": sampling_params["min_p"]},
                    stream=True,
                    timeout=360,
                )
            else:
                stream = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    stream=True,
                    timeout=360,
                )
            async for chunk in stream:
                try:
                    completion = completion + chunk.choices[0].delta.content
                except:
                    timed_out = True

            return prompt + completion, timed_out

        if self.mode == "cohere":
            raise Exception("Cohere not compatible with completion mode!")

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
        
        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        if self.mode == "api":
            completion = ""
            timed_out = False
            if use_min_p:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    extra_body={"min_p": sampling_params["min_p"]},
                    stream=True,
                )
            else:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                    stream=True,
                )
            async for chunk in stream:
                try:
                    if chunk.choices[0].delta.content:
                        completion = completion + chunk.choices[0].delta.content
                except Exception as e:
                    print("\n\n------------CAUGHT EXCEPTION DURING GENERATION")
                    print(e)
                    timed_out = True
                    print("\n\n-----/\------")

            return completion, timed_out

        elif self.mode == "cohere":
            timed_out = False
            completion = ""
            messages_cohereified = [
                {
                    "role": "USER" if message["role"] == "user" else "CHATBOT",
                    "message": message["content"],
                }
                for message in messages
            ]
            stream = self.client.chat_stream(
                model=self.model,
                chat_history=messages_cohereified[1:-1],
                message=messages_cohereified[-1]["message"],
                preamble=messages_cohereified[0]["message"],
                temperature=sampling_params["temperature"],
                p=sampling_params["top_p"],
                stop_sequences=sampling_params["stop"],
                max_tokens=sampling_params["max_tokens"],
            )
            async for chunk in stream:
                try:
                    if chunk.event_type == "text-generation":
                        completion = completion + chunk.text
                except Exception as e:
                    print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                    print(e)
                    timed_out = True

            return completion, timed_out

        else:
            raise Exception("Aphrodite not compatible with chat mode!")