import asyncio
import uuid
from openai import AsyncOpenAI
import cohere
from together import AsyncTogether
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
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        mode="api",  # can be one of api, aphrodite, llama.cpp
        quantization="gptq",  # only needed if using aphrodite mode
    ):
        self.mode = mode
        self.model = model
        if mode == "aphrodite":
            engine_args = AsyncEngineArgs(
                model=model,
                quantization=quantization,
                engine_use_ray=False,
                disable_log_requests=True,
                max_model_len=12000,
                dtype="float16",
            )
            self.engine = AsyncAphrodite.from_engine_args(engine_args)
        if mode == "cohere":
            self.client = cohere.AsyncClient(api_key=api_key)
        elif mode == "api":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        elif mode == "together":
            self.client = AsyncTogether(api_key=api_key)

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
        if "n_predict" not in sampling_params and self.mode == "llamacpp":
            sampling_params["n_predict"] = sampling_params["max_tokens"]
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
            timed_out = False
            completion = ""
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
                    # print(completion)
                except:
                    timed_out = True

            # completion = completion.choices[0].text
            return prompt + completion, timed_out
        if self.mode == "cohere":
            raise Exception("Cohere not compatible with completion mode!")

    async def submit_chat(
        self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        # print(
        #     "\n\n\n"
        # )
        # print(messages)
        # for item in messages:
        #     print(item["role"])
        #     print(item["content"])
        # print(
        #     "\n\n\n"
        # )
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []

        if self.mode == "llamacpp":
            return await make_async_api_call(
                messages=messages, sampling_parameters=sampling_params
            )
        elif self.mode == "api" or self.mode == "together":
            completion = ""
            timed_out = False
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
                    completion = completion + chunk.choices[0].delta.content
                    # print(completion)
                except Exception as e:
                    print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                    print(e)
                    timed_out = True  # catch timeout exception if it happens, at least this way we get whatever output has generated so far.

            # completion = completion.choices[0].message.content
            return completion, timed_out
        elif self.mode == "cohere":
            timed_out = False
            completion = ""
            messages_cohereified = [
                {  # modify messages to use cohere's format
                    "role": "USER" if message["role"] == "user" else "CHATBOT",
                    "message": message["content"],
                }
                for message in messages
            ]
            # print(f"\n\n=====================\nBEGIN PROMPT\nPreamble: {messages_cohereified[0]['message']}\nChat History: {messages_cohereified[1:-1]}\nMessage: {messages_cohereified[-1]['message']}\n=====================\n\n")
            stream = self.client.chat_stream(
                model=self.model,
                chat_history=messages_cohereified[1:-1],
                message=messages_cohereified[-1]["message"],
                preamble=messages_cohereified[0][
                    "message"
                ],  # Cohere by default has a preamble, it's just a system message, th
                temperature=sampling_params["temperature"],
                p=sampling_params["top_p"],
                stop_sequences=sampling_params["stop"],
                max_tokens=sampling_params["max_tokens"],
            )
            async for chunk in stream:
                try:
                    if chunk.event_type == "text-generation":
                        completion = completion + chunk.text
                    # completion = completion + chunk.
                    # print(completion)
                except Exception as e:
                    print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                    print(e)
                    timed_out = True
            return completion, timed_out
        else:
            raise Exception("Aphrodite not compatible with chat mode!")
