from aphrodite import (
    EngineArgs,
    AphroditeEngine,
    SamplingParams,
    AsyncAphrodite,
    AsyncEngineArgs,
)
import asyncio
import uuid


def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(self, model, quantization):
        engine_args = AsyncEngineArgs(
            model=model,
            quantization=quantization,
            engine_use_ray=False,
            disable_log_requests=True,
            max_model_len=12000,
            dtype="float16"
        )
        self.engine = AsyncAphrodite.from_engine_args(engine_args)

    async def submit(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        request_id = make_id()
        outputs = []
        # self.engine.add_request(request_id,prompt,sampling_params) #old sync code
        final_output = None
        async for request_output in self.engine.generate(
            prompt, sampling_params, request_id
        ):
            outputs.append(request_output.outputs[0].text)
            final_output = request_output

        full_output = "".join(outputs)
        return final_output.prompt + final_output.outputs[0].text
