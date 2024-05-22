import uuid
from openai import AsyncOpenAI
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
        mode="api",  # can be one of api, aphrodite, llama.cpp
        quantization="gptq",  # only needed if using aphrodite mode
    ):
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
        self.mode = mode
        self.base_url = base_url
        self.model = model
        if base_url == "gemini":
            self.client = Gemini(api_key=api_key)
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if self.base_url == "gemini":
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
        # print("\n\nSETTINGS DUMP\n\n")
        # print(self.model)
        # print(prompt)
        # print(sampling_params["temperature"])
        # print(sampling_params["top_p"])
        # print(sampling_params["max_tokens"])
        if self.mode == "llamacpp":
            return await make_async_api_call(
                prompt=prompt, sampling_parameters=sampling_params
            )

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

        if self.mode == "llamacpp":
            return await make_async_api_call(
                messages=messages, sampling_parameters=sampling_params
            )
        elif self.mode == "api":
            # print("\n\n\nMESSAGES\n\n\n")
            # print(messages)
            if self.base_url == "gemini":
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
                messages_cleaned = [
                    {
                        "role": message["role"],
                        "content": message["content"].replace("\\n", "\n"),
                    }
                    for message in messages
                ]
                # print(messages_cleaned)
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_cleaned,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    stop=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                )
                completion = completion.choices[0].message.content
            return completion
        else:
            raise Exception("Aphrodite not compatible with chat mode!")
