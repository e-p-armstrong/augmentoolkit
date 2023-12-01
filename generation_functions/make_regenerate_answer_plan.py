import re
from .make_regenerate_answer_plan_grammar import make_regenerate_answer_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps

# Question regeneration (triggered after a relevance-check fails on the question).
def make_regenerate_answer_plan(qatuple, dissenting_reasoning,logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""# Input:
You are an expert educational AI. Someone has provided an answer to a question based on a few paragraphs of text, but the answer may be incorrect due to contradicting the text or including erroneous information. Your task is to plan out and think through a correct answer to the question, ensuring it aligns with the information provided in the text.

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Allegedly incorrect answer to the question (for fact-checking): \"\"\"{qatuple[1]}\"\"\"

Reasoning as to why the answer is incorrect: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

For instance, if the text discusses the process of photosynthesis in plants, a question about the role of sunlight in photosynthesis, an answer stating that sunlight is not necessary for photosynthesis, and reasoning pointing out this contradiction, you might write:
Step 1. Analyze the Text: Focus on the details provided about photosynthesis in plants.
Step 2. Understand the Question: Grasp the question's emphasis on the role of sunlight in photosynthesis.
Step 3. Identify the Incorrect Part of the Answer: Note the incorrect claim about sunlight not being necessary, which contradicts the text. The plant text says "In photosynthesis, chlorophyll captures the sun's rays, providing the energy needed to build molecules of glucose from air and water."
Step 4. Plan a Corrected Answer: Devise a new answer emphasizing that sunlight is essential for photosynthesis, as explained in the text. One might write "Sunlight provides the energy required for photosynthetic organisms to convert carbon dioxide and water into glucose and oxygen."

Based on this example, plan out a revised version of the answer \"\"\"{qatuple[1]}\"\"\" with respect to the text and reasoning provided.

# Response:
## New answer plan:
"""
        completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=make_regenerate_answer_plan_grammar,temperature=0.2)["choices"][0]["text"]

        # print("DEBUG\n\n")
        print(completion)
        completion_pattern = re.compile(r"New answer plan:\n(.+)", re.DOTALL)
        correction = completion_pattern.search(completion).group(1)
        return correction.strip()
    
    
    
if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    inaccurate_qa_tuple = ("For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?", "The concept of a spherical Earth has been known for only about 1,000 years.", "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.")
    
    dissenting_reasoning = """Step 1. Analyze the Text: focus on the details provided about the history of spherical Earth.
Step 2. Understand the Question: the question's focus is on how long the concept of a spherical Earth has been known to at least a limited number of intelligent people.
Step 3. Compare the First Part of the Answer with the Text: check if the text supports the claim that the concept of a spherical Earth has only been known for about 1,000 years. It does, so this part is accurate.
Step 4. Compare the Second Part of the Answer with the Text: check if the text contradicts the claim that the concept of a spherical Earth has been known to at least a limited number of intelligent people for longer than 1,000 years. The text indicates this knowledge predates 2500 years ago, so this part is inaccurate.
Step 5. Final Judgment: Since the answer is not entirely accurate, the answer is inaccurate.  It's important to note that the second paragraph of the text does indeed mention a limited number of intelligent people knowing about the spherical Earth over 2,000 years ago, but this information is contradicted by the rest of the text which suggests the concept was unknown to these same "intelligent" people."""
    # Old thing below. Maybe use if I want to test with bad reasoning
#     """Step 1. Analyze the Text: Focus on the main points stated in the text about the Earth's diameter, distance from the sun, and its spherical shape.
# Step 2. Understand the Question: The question is asking about specific scientific principles used to explain why we know the Earth is approximately 8000 miles in diameter and how its distance from the sun varies.
# Step 3. Compare the First Part of the Answer with the Text: Check if the text states any scientific principles about measuring Earth's diameter. It doesn't, so this part seems irrelevant.
# Step 4. Compare the Second Part of the Answer with the Text: Check if the text states how we know that the Earth is approximately 8000 miles in diameter. It does mention "measurements of its circumference made using GPS data" as the method for determining this, which aligns with what is mentioned in the answer, so this part of the answer reflects aspects of the text and is relevant.
# Step 5. Compare the Third Part of the Answer with the text: Check if the text mentions any scientific principles explaining why Earth's distance from the sun varies. It doesn't mention an elliptical orbit or a varying point of closest approach and farthest departure, so this part seems irrelevant.
# Step 6. Final Judgment: Since only part of the answer reflects aspects of the text (specifically, how we know Earth's diameter), it is "Irrelevant"."""
    
    print("Begin HGWELLS test")
    result = make_regenerate_answer_plan(inaccurate_qa_tuple, dissenting_reasoning, logic_llm)