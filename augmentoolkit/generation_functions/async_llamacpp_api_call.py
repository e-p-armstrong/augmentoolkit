import aiohttp
import asyncio
import json

async def make_async_api_call(prompt, sampling_parameters, url='http://127.0.0.1:8080/completion'):
    # Convert the sampling parameters dictionary to JSON
    data = json.dumps({
        "prompt": prompt,
        **sampling_parameters
    })

    # Use aiohttp to make the async request
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={"Content-Type": "application/json"}) as response:
            if response.status == 200:
                # Parse the JSON response
                response_json = await response.json()
                return response_json
            else:
                return {"error": f"API call failed with status code: {response.status}"}

# Example usage
if __name__ == "__main__":
    prompt = "Building a website can be done in 10 simple steps:"
    sampling_parameters = {
        "n_predict": 128
    }

    # Run the async function
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(make_async_api_call(prompt, sampling_parameters))
    print(response)
