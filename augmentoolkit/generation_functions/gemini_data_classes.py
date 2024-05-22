from enum import StrEnum
from dataclasses import dataclass, asdict, field

from httpx import URL
from httpx import Headers as httpxHeaders


class HarmCategory(StrEnum):
    """
    Enumeration class representing different categories of harmful content.
    """

    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"


class HarmBlockThreshold(StrEnum):
    """
    Enumeration class representing the threshold for blocking harmful blocks.
    """

    UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"


class GeminiModels(StrEnum):
    """
    Enum class representing different Gemini models.
    """

    FLASH = "gemini-1.5-flash-latest"
    PRO_1_5 = "gemini-1.5-pro-latest"
    PRO_1_0 = "gemini-1.0-pro"


class APIVersion(StrEnum):
    """
    Enum class representing different API versions.
    """

    V1_BETA = "v1beta"
    V1 = "v1"


class Role(StrEnum):
    """
    Represents the role of a user, system, or model.
    """

    USER = "user"
    SYSTEM = "system"
    MODEL = "model"


@dataclass
class Part:
    """
    Represents a text message.

    Attributes:
        text (str): The text of the message.
    """

    text: str


@dataclass
class SystemInstruction:
    """
    Represents a system instruction.

    Attributes:
        role (str): The role of the instruction. Default is "system".
        parts (list[Part]): The message parts of the instruction.
    """

    role: Role = Role.SYSTEM
    parts: list[Part] = field(
        default_factory=lambda: [Part(text="You are a helpful AI assistant.")]
    )

    def dumps(self) -> dict[str, list[dict[str, str]]]:
        """
        Converts the SystemInstruction object to a dictionary.

        Returns:
            dict[str, list[dict[str, str]]]: The dictionary representation of the SystemInstruction object.
        """
        return {"role": self.role, "parts": [asdict(part) for part in self.parts]}


@dataclass
class Message:
    """
    Represents a message with a role and a list of parts.
    """

    role: Role
    parts: list[Part]

    def dumps(self) -> dict[str, list[dict[str, str]]]:
        """
        Serializes the message object into a dictionary.

        Returns:
            A dictionary representation of the message object.
        """
        return {"role": self.role.value, "parts": [asdict(part) for part in self.parts]}

    @classmethod
    def loads(cls, data: list[dict[str, str]]) -> "Message":
        """
        Deserializes a dictionary into a message object.

        Args:
            data: A list of dictionaries representing the message data.

        Returns:
            A Message object created from the deserialized data.
        """
        parts = [Part(**part) for part in data["parts"]]
        return cls(role=Role(data["role"]), parts=parts)


@dataclass
class Contents:
    """
    Represents a collection of messages.
    """

    messages: list[Message]

    def dumps(self) -> list[dict[str, list[dict[str, str]]]]:
        """
        Serializes the contents to a list of dictionaries.

        Returns:
            A list of dictionaries representing the serialized contents.
        """
        return [message.dumps() for message in self.messages]

    @classmethod
    def loads(cls, data: list[dict[str, list[dict[str, str]]]]) -> "Contents":
        """
        Deserializes the contents from a list of dictionaries.

        Args:
            data: A list of dictionaries representing the serialized contents.

        Returns:
            An instance of Contents with the deserialized contents.
        """
        messages = [Message.loads(message) for message in data["contents"]]
        return cls(messages=messages)


@dataclass
class SafetySettings:
    """
    Represents the safety settings for a request.

    Attributes:
        harassment (HarmBlockThreshold): The threshold for blocking harassment content.
        hate_speech (HarmBlockThreshold): The threshold for blocking hate speech content.
        sexually_explicit (HarmBlockThreshold): The threshold for blocking sexually explicit content.
        dangerous_content (HarmBlockThreshold): The threshold for blocking dangerous content.
    """

    harassment: HarmBlockThreshold = HarmBlockThreshold.BLOCK_ONLY_HIGH
    hate_speech: HarmBlockThreshold = HarmBlockThreshold.BLOCK_ONLY_HIGH
    sexually_explicit: HarmBlockThreshold = HarmBlockThreshold.BLOCK_ONLY_HIGH
    dangerous_content: HarmBlockThreshold = HarmBlockThreshold.BLOCK_ONLY_HIGH

    def dumps(self) -> list[dict[str, str]]:
        """
        Converts the object to a list of dictionaries representing the thresholds for different harm categories.

        Returns:
            A list of dictionaries, where each dictionary contains the category and threshold values.
        """
        return [
            {
                "category": HarmCategory.HARASSMENT.value,
                "threshold": self.harassment.value,
            },
            {
                "category": HarmCategory.HATE_SPEECH.value,
                "threshold": self.hate_speech.value,
            },
            {
                "category": HarmCategory.SEXUALLY_EXPLICIT.value,
                "threshold": self.sexually_explicit.value,
            },
            {
                "category": HarmCategory.DANGEROUS_CONTENT.value,
                "threshold": self.dangerous_content.value,
            },
        ]


@dataclass
class GenerationConfig:
    """
    Represents the configuration for text generation.

    Attributes:
        temperature (float | int): The temperature parameter for controlling the randomness of the generated text. Default is 1.
        top_p (float): The top-p parameter for controlling the diversity of the generated text. Default is 0.95.
        max_output_tokens (int): The maximum number of tokens in the generated text. Default is 8192.
    """

    temperature: float | int = 1
    top_p: float = 0.95
    max_output_tokens: int = 8192

    def dumps(self) -> dict[str, float | int]:
        """
        Converts the GenerationConfig object to a dictionary.

        Returns:
            dict[str, float | int]: A dictionary representation of the GenerationConfig object.
        """
        return asdict(self)


class Headers(httpxHeaders):
    """
    Represents a set of HTTP headers for Gemini API requests.

    Inherits from the `httpx.Headers` class.

    Args:
        None

    Attributes:
        None

    Methods:
        None

    Usage:
        headers = Headers()
    """

    def __init__(self):
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "br, gzip, deflate",
        }
        super().__init__(headers)


@dataclass
class APIParameters(dict):
    """
    Represents the parameters for the API.

    Attributes:
        base_url (URL): The base URL for the API.
        api_version (APIVersion): The version of the API.
        model (GeminiModels): The Gemini model to use.
    """

    base_url: URL = URL("https://generativelanguage.googleapis.com")
    api_version: APIVersion = APIVersion.V1_BETA
    model: GeminiModels = GeminiModels.FLASH

    def dumps(self) -> dict[str, str]:
        """
        Converts the APIParameters object to a dictionary.

        Returns:
            dict[str, str]: The dictionary representation of the APIParameters object.
        """
        return asdict(self)
