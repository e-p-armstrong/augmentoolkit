import asyncio
import uuid
from openai import AsyncOpenAI
import cohere
from httpx import Timeout
import traceback
import json


def make_id():
    return str(uuid.uuid4())


# How it'll work:
# Things that we want to attach to the engine wrapper will execute either before or after a generation
# it's like input or output observers
# they take the input or output as their input respectively; they're functions
# if we want to customize their behavior (i.e., log outputs to a specific directory) then we make them return a callback that's passed in instead
# inputs = all the messages
# outputs = the completion
# oh and correction, they also take a flag for completion mode, since the input will vary based on that


class EngineWrapper:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        mode="api",  # can be one of api, aphrodite, llama.cpp, cohere
        input_observers=[],
        output_observers=[],
        timeout_total=500.0,  # Total operation timeout
        timeout_read=240.0,  # Read timeout between chunks
        timeout_api_call=600,  # Timeout for individual API calls
        **kwargs,
    ):
        self.mode = mode
        self.model = model
        self.input_observers = input_observers
        self.output_observers = output_observers
        self.timeout_api_call = timeout_api_call  # Store for use in API calls
        if mode == "cohere":
            self.client = cohere.AsyncClient(api_key=api_key)
        elif mode == "api":
            self.client = AsyncOpenAI(
                timeout=Timeout(
                    timeout=timeout_total,  # Total operation timeout
                    connect=10.0,  # Connection timeout
                    read=timeout_read,  # Read timeout between chunks (increased for streaming)
                    write=30.0,  # Write timeout
                    pool=10.0,  # Pool timeout
                ),
                api_key=api_key,
                base_url=base_url,
            )

    async def submit_completion(
        self, prompt, sampling_params, return_completion_only=False
    ):  # Submit request and wait for it to stream back fully
        print(prompt)
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = [
                "<|im_end|>"
            ]  # NOTE we hardcode the im end stop token due to the custom dataset generation model always needing it. I really need to get better at configuring my tokenizers...
        else:
            if "<|im_end|>" not in sampling_params["stop"]:
                sampling_params["stop"].append("<|im_end|>")
        if "n_predict" not in sampling_params:
            sampling_params["n_predict"] = sampling_params["max_tokens"]

        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        for input_observer in self.input_observers:
            input_observer(prompt, completion_mode=True)

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
                    timeout=self.timeout_api_call,  # Use configurable timeout
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
                    timeout=self.timeout_api_call,  # Use configurable timeout
                )
            async for chunk in stream:
                try:
                    completion = completion + chunk.choices[0].text
                except Exception as e:
                    timed_out = True

            for output_observer in self.output_observers:
                output_observer(
                    prompt, completion, True
                )  # input, output, completion_mode (this is the input format for output observers)

            if not return_completion_only:
                return prompt + completion, timed_out
            else:
                return completion, timed_out

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
            sampling_params["stop"] = ["<|im_end|>"]
        else:
            if "<|im_end|>" not in sampling_params["stop"]:
                sampling_params["stop"].append("<|im_end|>")

        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        for input_observer in self.input_observers:
            input_observer(messages, False)

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
                    timeout=self.timeout_api_call,  # Use configurable timeout
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
                    timeout=self.timeout_api_call,  # Use configurable timeout
                )
            async for chunk in stream:
                try:
                    # print(chunk.choices)
                    try:
                        if chunk.choices[0].delta.content:
                            completion = completion + chunk.choices[0].delta.content
                    except Exception as e:
                        # print("Really strange exception!")
                        # print(chunk)
                        # traceback.print_exc()
                        pass
                except Exception as e:
                    print("\n\n------------CAUGHT EXCEPTION DURING GENERATION")
                    print(e)
                    traceback.print_exc()
                    timed_out = True
                    print("\n\n-----/\------")

            for output_observer in self.output_observers:
                output_observer(messages, completion, False)

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

            for output_observer in self.output_observers:
                output_observer(messages, completion, False)

            return completion, timed_out
        else:
            raise Exception("Aphrodite not compatible with chat mode!")

    async def submit_completion_streaming(
        self, prompt, sampling_params, return_completion_only=False
    ):
        """Submit request and yield chunks as they arrive for streaming"""
        print(prompt)
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = ["<|im_end|>"]
        else:
            if "<|im_end|>" not in sampling_params["stop"]:
                sampling_params["stop"].append("<|im_end|>")
        if "n_predict" not in sampling_params:
            sampling_params["n_predict"] = sampling_params["max_tokens"]

        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        for input_observer in self.input_observers:
            input_observer(prompt, completion_mode=True)

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
                    timeout=self.timeout_api_call,
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
                    timeout=self.timeout_api_call,
                )

            async for chunk in stream:
                try:
                    text_chunk = chunk.choices[0].text
                    completion += text_chunk
                    # Yield chunk in SSE format
                    yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"
                except Exception as e:
                    timed_out = True
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                    break

            # Send completion signal
            if not timed_out:
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

            for output_observer in self.output_observers:
                output_observer(prompt, completion, True)

        if self.mode == "cohere":
            raise Exception("Cohere not compatible with completion mode!")

    async def submit_chat_streaming(self, messages, sampling_params):
        """Submit chat request and yield chunks as they arrive for streaming"""
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = ["<|im_end|>"]
        else:
            if "<|im_end|>" not in sampling_params["stop"]:
                sampling_params["stop"].append("<|im_end|>")

        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        for input_observer in self.input_observers:
            input_observer(messages, False)

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
                    timeout=self.timeout_api_call,
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
                    timeout=self.timeout_api_call,
                )

            async for chunk in stream:
                try:
                    try:
                        if chunk.choices[0].delta.content:
                            text_chunk = chunk.choices[0].delta.content
                            completion += text_chunk
                            # Yield chunk in SSE format
                            yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"
                    except Exception as e:
                        pass
                except Exception as e:
                    print("\n\n------------CAUGHT EXCEPTION DURING GENERATION")
                    print(e)
                    traceback.print_exc()
                    timed_out = True
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                    break

            # Send completion signal
            if not timed_out:
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

            for output_observer in self.output_observers:
                output_observer(messages, completion, False)

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
                        text_chunk = chunk.text
                        completion += text_chunk
                        yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"
                except Exception as e:
                    print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                    print(e)
                    timed_out = True
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                    break

            if not timed_out:
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

            for output_observer in self.output_observers:
                output_observer(messages, completion, False)
        else:
            raise Exception("Aphrodite not compatible with chat mode!")
