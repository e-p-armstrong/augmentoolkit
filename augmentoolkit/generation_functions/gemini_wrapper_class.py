import httpx
import orjson
import asyncio
from datetime import datetime, timedelta
from augmentoolkit.generation_functions.gemini_data_classes import (
    APIParameters,
    Headers,
    GenerationConfig,
    SafetySettings,
    SystemInstruction,
    Contents,
)


class Gemini:
    """
    Wrapper class for interacting with the Gemini API.
    """

    api_parameters: APIParameters
    headers: Headers
    safety_settings: SafetySettings
    system_instruction: SystemInstruction
    request_parameters: dict[str, str]
    client: httpx.AsyncClient
    semaphore: asyncio.Semaphore
    rate_limit_count: int
    rate_limit_reset_time: datetime

    def __init__(
        self,
        api_key: str,
        api_parameters: APIParameters = APIParameters(),
        headers: Headers = Headers(),
        safety_settings: SafetySettings = SafetySettings(),
    ):
        self.api_parameters = api_parameters
        self.headers = headers
        self.safety_settings = safety_settings
        self.request_parameters: dict[str, str] = {"key": api_key}
        self.client = httpx.AsyncClient(http1=False, http2=True, timeout=30.0)
        self.semaphore = asyncio.Semaphore(2)
        self.rate_limit_count = 0
        self.rate_limit_reset_time = datetime.now() + timedelta(minutes=1)

    async def generate_content(
        self,
        contents: Contents,
        generation_config: GenerationConfig,
        system_instruction: SystemInstruction,
    ) -> str:
        """
        Generates content using the Gemini API.

        Args:
            contents (Contents): The contents object containing the input data.
            generation_config (GenerationConfig): The generation configuration.
            system_instruction (SystemInstruction): The system instruction.

        Returns:
            str: The generated content.

        Raises:
            Exception: If there is an HTTP error or if the content generation fails.
        """

        if datetime.now() >= self.rate_limit_reset_time:
            self.rate_limit_count = 0
            self.rate_limit_reset_time = datetime.now() + timedelta(minutes=1)

        if self.rate_limit_count >= 360:
            await asyncio.sleep(
                (self.rate_limit_reset_time - datetime.now()).total_seconds()
            )

        async with self.semaphore:
            self.rate_limit_count += 1
            try:
                url: str = (
                    f"{self.api_parameters.base_url}/{self.api_parameters.api_version}/models/{self.api_parameters.model}:generateContent"
                )

                body: dict[str, dict | list] = {
                    "contents": contents.dumps(),
                    "safetySettings": self.safety_settings.dumps(),
                    "generationConfig": generation_config.dumps(),
                    "systemInstruction": system_instruction.dumps(),
                }

                async with self.client.stream(
                    method="POST",
                    url=url,
                    headers=self.headers,
                    params=self.request_parameters,
                    json=body,
                ) as stream:
                    if stream.status_code != 200:
                        raise httpx.HTTPStatusError(
                            f"{stream.status_code}", request=None, response=stream
                        )
                    buffer = bytearray()
                    async for chunk in stream.aiter_bytes():
                        buffer.extend(chunk)
                    try:
                        response = orjson.loads(buffer)
                        text = response["candidates"][0]["content"]["parts"][0]["text"]
                    except (orjson.JSONDecodeError, KeyError, IndexError) as e:
                        raise Exception(f"Failed to parse response: {e}")
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP error: {e}")
            if text:
                return text
            else:
                raise Exception("Failed to generate content")
