import asyncio
import uuid
from openai import AsyncOpenAI
import cohere
from augmentoolkit.generation_functions.gemini_data_classes import (
    Part,
    SystemInstruction,
    Contents,
    GenerationConfig,
)
from augmentoolkit.generation_functions.async_llamacpp_api_call import (
    make_async_api_call,
)
from augmentoolkit.generation_functions.gemini_wrapper_class import Gemini

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
        mode="api",  # can be one of api, aphrodite, llama.cpp, gemini, cohere
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
        elif mode == "gemini":
            self.client = Gemini(api_key=api_key)

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if self.mode == "gemini":
            raise Exception(
                "The Gemini API isn't compatible with completion mode. Use chat mode instead."
            )
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

        if self.mode == "llamacpp":
            return await make_async_api_call(
                prompt=prompt, sampling_parameters=sampling_params
            )

        if self.mode == "aphrodite":
            aphrodite_sampling_params = SamplingParams(**sampling_params)
            request_id = make_id()
            outputs = []
            final_output = None
            async for request_output in self.engine.generate(
                prompt, aphrodite_sampling_params, request_id
            ):
                outputs.append(request_output.outputs[0].text)
                final_output = request_output

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

        if self.mode == "llamacpp":
            return await make_async_api_call(
                messages=messages, sampling_parameters=sampling_params
            )

        elif self.mode == "api":
            if self.mode == "gemini":
                generation_config = GenerationConfig(
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    max_output_tokens=8192,
                )

                for message in messages:
                    if message["role"] == "system":
                        self.client.system_instruction = message["content"]
                        system_instruction = SystemInstruction(
                            parts=[Part(text=message["content"])],
                        )
                        break

                messages_cleaned = [
                    {
                        "role": (
                            "model" if message["role"] == "assistant" else ("user")
                        ),
                        "parts": [{"text": message["content"].replace("\\n", "\n")}],
                    }
                    for message in messages
                ]

                contents = Contents.loads({"contents": messages_cleaned})

                completion = await self.client.generate_content(
                    contents, generation_config, system_instruction
                )
            else:
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
                    except:
                        print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                        timed_out = True

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