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
    
    
    
    prompt = f"""You are specialized in detecting strange and broken outputs from large language models, and you will be judging whether or not some output is broken. You will be provided some text that is the output of a model. REMEMBERING YOUR TASK, and NOT GETTING CONFUSED, you will produce a judgement about whether it appears that the text is as it should be (ie, not incoherent).

Signs of a broken response include:
Chunks of repeating text.
Instructions or prompt pieces appearing in the text.
No logical coherence after a point.

Anaylze the text step-by-step and then give your final answer. Either a text is "Broken" or it is "Fine."

### Instruction:
Text to determine if it is broken or not: 
\"\"\"
1.) The text states that we know about the shape and movement of the earth, but it does not provide any information regarding the creation of the universe. So, based on these texts, what are the known shapes and movements of the earth? 2.) Similarly, though there is a mention of the age of the universe being endless, according to these texts, what is currently believed about the age of the universe in modern times? 3.) Remember, your answer should be based only on the information provided in the given paragraphs. Please avoid referring to additional sources or knowledge outside those paragraphs. 4.) An example to illustrate how you could think about this: If someone were asked "What is known about the creation of the universe?" and they replied with "According to these texts, not much", that would be an accurate response using only information from the provided paragraphs. It's similar to if a student said "I know that this text tells me X, but it doesn't explain how we know X". The question could then ask something like "How does knowing Y and Z help us understand X better?", which uses the factual information from the original text. This approach allows for questions based on the provided information while avoiding irrelevant or misleading inquiries. 5.) End of reasoning. Please remember, you are not asking about the creation of the universe; instead, you're focusing on the known shapes and movements of the earth as well as the current beliefs regarding the age of the universe based solely on these texts. Thank you!
Answer: 1) The known shapes and movements of the earth according to these texts are that it is a spheroid, slightly compressed, with an almost circular shape and a diameter of nearly 8000 miles. It rotates upon its axis every twenty-four hours and circles around the sun in a slightly distorted oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles. 2) Based on these texts, the current belief about the age of the universe is that it has existed for an enormous period of time and possibly endlessly. However, this could be deception because a room may seem endless when mirrors are facing each other at either end. But, it's universally recognized nowadays that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.
\"\"\"

Remember, your task is to analyze this text to see if it is broken, and if it is, write "Broken". Otherwise, write "Fine."

### Response:
## Analysis:
Step 1: Evaluate the Logical Structure and Coherence. The text appears to be a series of questions and an answer related to the Earth's shape, movement, and the age of the universe. However, the flow of the text is disjointed, with questions and explanations intermingling in a confusing manner.
Step 3: Look for obvious repetitions. There are no blocks of repeating text in this output, but the logical relevance progression remains questionable.
Step 3. Look for instructions bleeding into the response. There are multiple phrases like "Remember, your answer should be based only on the information provided in the given paragraphs" and "An example to illustrate how you could think about this:".
Step 4: Determine the Overall Coherence: the text should maintain a clear focus and not deviate into unrelated or tangential topics. This text deviates significantly, especially with the mirror analogy and the notion about the age of the universe, which do not align with the initial questions about the Earth's shape and movement.
Step 5: Final Judgment: Based on the logical structure, clarity, consistency, and overall coherence, this text can be judged as "broken." The text fails to maintain a clear and logical flow, mixes accurate and inaccurate information without clear distinctions, and includes irrelevant analogies that contribute to its disjointed and confusing nature.

### Instruction:
Text to determine if it is broken or not: 
\"\"\"
{text}
\"\"\"

Remember, your task is to analyze this text to see if it is broken, and if it is, write "Broken". Otherwise, write "Fine."

### Response:
## Edit plan:
"""
    completion = logic_llm(prompt, max_tokens=2000, stop=["</s>","# Input:"], echo=True, grammar=proofread_output_grammar,temperature=0.2)["choices"][0]["text"]
    # print("COMPLETION:\n\n----------------------")
    # print(completion)
    # print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Begin Edit: (.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    text = """ Given the question, its answer, and the provided primary character card, one compelling possibility for a scenario that makes sense is &Our setting will be a cozy bookstore, where Drummond is currently tending to the shelves. The secondary character who asks him a question will be an enthusiastic patron of this store, and they have come in today because they were drawn by the advertisement for a discussion on "Earth's Aging". Our secondary character is eager to understand how our perception of the Earth's age has changed throughout history.  The conversation would likely take place at the counter where Drummond is sorting books, with him pausing occasionally to help patrons browse or answer other questions as he sees fit."""

    print("Begin Drummond Test")
    edit = proofread_output(text,logic_llm)