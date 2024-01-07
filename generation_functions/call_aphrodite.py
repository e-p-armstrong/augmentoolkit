import requests
import json

def llm_call(prompt=None,
         min_p=None,
         temperature=None,
         top_p=None,
        #  model=None,
         max_length=None, # Not supported with pseudo-OAI api
         max_context_length=None,
         stop=["## Instruction", "</s>", "## Question", "# Input", "[INST]"],
         top_k=None):
    data = { k: v for k, v in locals().items() if v is not None } # create dictionary to send off
    # url = "http://localhost:35042/v1/completions"
    url = "http://localhost:35042/api/v1/generate"
    headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key-needed"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    return response