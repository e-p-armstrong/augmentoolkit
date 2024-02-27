import aiohttp
import asyncio
import json


async def make_async_api_call(
    prompt=None, sampling_parameters={}, url="http://127.0.0.1:8080", messages=None
):
    # Determine the endpoint based on the presence of messages
    if messages is not None:
        endpoint = "/v1/chat/completions"
        data = json.dumps(
            {
                "messages": messages,
                **sampling_parameters,  # Assuming sampling parameters can be applied to chat
            }
        )
    else:
        endpoint = "/completion"
        data = json.dumps({"prompt": prompt, **sampling_parameters})

    # Complete the URL with the chosen endpoint
    full_url = url + endpoint

    # Use aiohttp to make the async request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            full_url, data=data, headers={"Content-Type": "application/json"}, ssl=False
        ) as response:
            if response.status == 200:
                # Parse the JSON response
                response_json = await response.json()
                if prompt:
                    return prompt + response_json["content"]
                else:
                    return response_json["choices"][0]["content"]
            else:
                return {"error": f"API call failed with status code: {response.status}"}


# Example usage for completion
if __name__ == "__main__":
    prompt = "Building a website can be done in 10 simple steps:"
    sampling_parameters = {"n_predict": 128}

    # Run the async function for completion
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        make_async_api_call(prompt=prompt, sampling_parameters=sampling_parameters)
    )
    print("Completion Response:", response)

    # Example usage for chat
    messages = [
        {"role": "system", "content": "You are Elise Delacroix, an AI assistant."},
        {"role": "user", "content": "Write a limerick about python exceptions."},
    ]

    # Run the async function for chat
    chat_response = loop.run_until_complete(make_async_api_call(messages=messages))
    print("Chat Response:", chat_response)
