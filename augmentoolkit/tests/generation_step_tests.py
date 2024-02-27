import unittest
from unittest.mock import AsyncMock, patch
import re
import os
import json

from augmentoolkit.generation_functions.generation_step_class import GenerationStep


class TestGenerationStep(unittest.IsolatedAsyncioTestCase):
    async def test_generate_completion_mode(self):
        # Mocking the engine_wrapper for completion mode
        mock_response = "This is a test response."
        engine_wrapper_mock = AsyncMock()
        engine_wrapper_mock.submit_completion = AsyncMock(return_value=mock_response)

        # Initialize the GenerationStep with mocked engine_wrapper
        step = GenerationStep(
            prompt_path="test_prompt.txt",  # Assume this file exists in your INPUT_DIRECTORY with a suitable prompt
            regex=re.compile(r"(test response)"),
            completion_mode=True,
            engine_wrapper=engine_wrapper_mock,
            arguments={"test": "replacement"},
        )

        # Execute the generate method
        result = await step.generate()

        # Verify the result
        self.assertEqual(result, "test response")

    async def test_generate_chat_mode(self):
        # Mocking the engine_wrapper for chat mode
        mock_response = "This is a chat test response."
        engine_wrapper_mock = AsyncMock()
        engine_wrapper_mock.submit_chat = AsyncMock(return_value=mock_response)

        # Initialize the GenerationStep with mocked engine_wrapper
        step = GenerationStep(
            prompt_path="test_chat_prompt.json",  # Assume this JSON file exists with a suitable structure
            regex=re.compile(r"(chat test response)"),
            completion_mode=False,
            engine_wrapper=engine_wrapper_mock,
            arguments={"test": "replacement"},
        )

        # Execute the generate method
        result = await step.generate()

        # Verify the result
        self.assertEqual(result, "chat test response")
