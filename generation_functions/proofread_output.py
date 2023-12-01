import re
from .proofread_output_grammar import proofread_output_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

# But frankly, I can just use that text fixer thing rather than using an LLM. MUCH faster that. ftfy I believe it was called.

# This does actually work decently though!
def proofread_output(text,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # It's way more willing to use different time periods than I expected, which is cool.
    prompt = f"""# Input:
You are an expert "mechanical editing" AI that is going to fix all typographical, grammatical, and spelling errors in a provided text, without making any other corrections. You will leave stylistic choices and everything else completely unchanged.

Text to edit: \"\"\"{text}\"\"\"

First, plan out your edit, step-by-step, identifying mistakes in the source text. Then, make the edit, changing only mechanical errors.

Example:
Step 1. Analyze the text. The text describes XYZ and is written in a style that etc...
Step 2. There is an error... etc...
...etc...
Step N. End of reasoning.

Begin Edit: [the full text, with all mechanical errors fixed]

Add as many steps as you need, then output "End of reasoning" as a final step. Then do the edit.

# Response:
## Edit plan:
"""
    completion = logic_llm(prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=proofread_output_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Begin Edit: (.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    text = """ Given the question, its answer, and the provided primary character card, one compelling possibility for a scenario that makes sense is &Our setting will be a cozy bookstore, where Drummond is currently tending to the shelves. The secondary character who asks him a question will be an enthusiastic patron of this store, and they have come in today because they were drawn by the advertisement for a discussion on "Earth's Aging". Our secondary character is eager to understand how our perception of the Earth's age has changed throughout history.  The conversation would likely take place at the counter where Drummond is sorting books, with him pausing occasionally to help patrons browse or answer other questions as he sees fit."""

    print("Begin Drummond Test")
    edit = proofread_output(text,logic_llm)
    
        
    ## TODO a wider variety of tests from different texts
    ## TODO add a space between "a" and the LLM completion. It's bugged rn. But adding it in the prompt breaks the completion, so it needs to be done afterwards.
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.