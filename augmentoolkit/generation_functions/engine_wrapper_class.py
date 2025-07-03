import asyncio
import uuid
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import cohere
from httpx import Timeout
import traceback
import json
from collections import deque
import functools
import random


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
        api_keys=None,
        base_url=None,
        mode="api",  # can be one of api, cohere
        input_observers=[],
        output_observers=[],
        timeout_api_call=600,
        **kwargs,
    ):
        self.mode = mode
        self.model = model
        self.input_observers = input_observers
        self.output_observers = output_observers
        self.timeout_api_call = timeout_api_call
        self.base_url = base_url

        if api_keys:
            self.api_keys = deque(api_keys)
        elif api_key:
            self.api_keys = deque([api_key])
        else:
            raise ValueError("No API key provided. Please supply either 'api_key' or 'api_keys'.")
        
        self.total_keys = len(self.api_keys)
        if self.total_keys == 0:
            raise ValueError("The provided 'api_keys' list cannot be empty.")

        self._rotation_lock = asyncio.Lock()
        self._reinitialize_client()

    def _reinitialize_client(self):
        if not self.api_keys:
             raise Exception("All API keys have been exhausted.")
        
        current_key = self.api_keys[0]
        key_preview = f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else current_key
        print(f"\n[EngineWrapper] Initializing client for model '{self.model}' with key: {key_preview}\n")

        if self.mode == "cohere":
            self.client = cohere.AsyncClient(api_key=current_key, max_retries=0)
        elif self.mode == "api":
            self.client = AsyncOpenAI(
                timeout=Timeout(
                    timeout=self.timeout_api_call,
                    connect=10.0,
                    read=self.timeout_api_call - 10,
                    write=30.0,
                    pool=10.0,
                ),
                api_key=current_key,
                base_url=self.base_url,
                max_retries=0,
            )

    def _rotate_key_and_reinit(self):
        self.api_keys.rotate(-1)
        self._reinitialize_client()

    async def _api_call_with_retry(self, method_path, **kwargs):
        """
        Makes an API call with a two-tiered retry system:
        1. Inner Loop: Retries on transient server errors (5xx) with exponential backoff.
        2. Outer Loop: Rotates API keys on client errors (429).
        """
        # Outer loop for iterating through API keys
        for key_attempt in range(self.total_keys):
            key_for_this_attempt = self.api_keys[0]
            
            # Inner loop for handling transient errors
            max_transient_retries = 5
            initial_backoff = 3.0  # seconds

            for transient_retry_num in range(max_transient_retries):
                try:
                    method_to_call = functools.reduce(getattr, method_path.split('.'), self.client)
                    return await method_to_call(**kwargs)

                except (RateLimitError, APIStatusError) as e:
                    status_code = getattr(e, 'status_code', 500)

                    # Case 1: Rate limit/quota error (429). Stop retrying with this key.
                    if status_code == 429:
                        print(f"[EngineWrapper] Caught a 429 (Rate Limit/Quota) error. Rotating key.")
                        # Break the inner loop to proceed to the key rotation logic below.
                        break 
                    
                    # Case 2: Transient server error. Wait and retry with the same key.
                    elif status_code in [404, 400, 500, 502, 503, 504]:
                        if transient_retry_num < max_transient_retries - 1:
                            wait_time = (initial_backoff ** transient_retry_num) + random.uniform(0, 1)
                            print(f"[EngineWrapper] Caught a transient server error ({status_code}). Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                            # Continue the inner loop to retry the request.
                            continue
                        else:
                            print(f"[EngineWrapper] Caught a transient server error ({status_code}). Max retries reached for this key.")
                            # Break the inner loop to let the outer loop try the next key.
                            break
                    
                    # Case 3: Other client-side error (4xx). This is a permanent failure for this request.
                    else:
                        print(f"[EngineWrapper] Caught a non-retriable API error ({status_code}).")
                        raise e

                except Exception as e:
                    print(f"[EngineWrapper] An unexpected network or client error occurred: {e}")
                    traceback.print_exc()
                    raise e
            
            # This block is reached if the inner loop breaks (either from a 429 or max transient retries).
            async with self._rotation_lock:
                if self.api_keys[0] == key_for_this_attempt:
                    print("[EngineWrapper] This task is rotating the key.")
                    self._rotate_key_and_reinit()
                else:
                    print("[EngineWrapper] Key was already rotated by another task. Using the new key.")
        
        raise Exception("All API keys and retry attempts failed.")

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
            try:
                if use_min_p:
                    stream = await self._api_call_with_retry(
                        "completions.create",
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
                    stream = await self._api_call_with_retry(
                        "completions.create",
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
                    completion = completion + chunk.choices[0].text
            except Exception as e:
                timed_out = True
                print(f"Exception during completion stream: {e}")

            for output_observer in self.output_observers:
                output_observer(
                    prompt, completion, True
                )

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
            try:
                if use_min_p:
                    stream = await self._api_call_with_retry(
                        "chat.completions.create",
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
                    stream = await self._api_call_with_retry(
                        "chat.completions.create",
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
                        if chunk.choices[0].delta.content:
                            completion = completion + chunk.choices[0].delta.content
                    except Exception:
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
            try:
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
            completion = ""
            timed_out = False
            stream = None
            try:
                if use_min_p:
                    stream = await self._api_call_with_retry(
                        "completions.create",
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
                    stream = await self._api_call_with_retry(
                        "completions.create",
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
                    text_chunk = chunk.choices[0].text
                    completion += text_chunk
                    yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"

            except Exception as e:
                timed_out = True
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

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
            stream = None
            try:
                if use_min_p:
                    stream = await self._api_call_with_retry(
                        "chat.completions.create",
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
                    stream = await self._api_call_with_retry(
                        "chat.completions.create",
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
                        if chunk.choices[0].delta.content:
                            text_chunk = chunk.choices[0].delta.content
                            completion += text_chunk
                            yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"
                    except Exception:
                        pass
            except Exception as e:
                print("\n\n------------CAUGHT EXCEPTION DURING GENERATION")
                print(e)
                traceback.print_exc()
                timed_out = True
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

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
            try:
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
                    if chunk.event_type == "text-generation":
                        text_chunk = chunk.text
                        completion += text_chunk
                        yield f"data: {json.dumps({'text': text_chunk, 'done': False})}\n\n"
            except Exception as e:
                print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                print(e)
                timed_out = True
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                
            if not timed_out:
                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

            for output_observer in self.output_observers:
                output_observer(messages, completion, False)
        else:
            raise Exception("Aphrodite not compatible with chat mode!")