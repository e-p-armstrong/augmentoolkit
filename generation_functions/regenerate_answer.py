import re
from .regenerate_answer_grammar import regenerate_answer_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps

# Answer regeneration (triggered after a fact-check fails for reason of "inaccurate").
def regenerate_answer(qatuple, dissenting_reasoning, plan, logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""You are an expert educational AI. Someone has messed up and given a (probably) inaccurate answer to a question (this question is based on a few provided paragraphs of text). Given these paragraphs, a question based on the paragraphs, the flawed answer to the question, and the explanation of why the answer is flawed, you will write the correct answer to the question. 

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Allegedly incorrect answer to the question (this is what you are fact-checking): \"\"\"{qatuple[1]}\"\"\"

Reasoning as to why the answer is incorrect: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

If there are many questions, just answer the first 2.

### Response:
## Plan for new answer (step-by-step):
{plan}
# New answer (comprehensive and complete; do not mention the text):
The correct answer would be \""""
        try:
            completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>","\n"], echo=True, grammar=regenerate_answer_grammar,temperature=0.2)["choices"][0]["text"]

            # print("DEBUG\n\n")
            print(completion)
            completion_pattern = re.compile(r"New answer \(comprehensive and complete; do not mention the text\):\nThe correct answer would be \"(.+)", re.DOTALL)
            correction = completion_pattern.search(completion).group(1)
            if "Step 4: Plan a Corrected" in correction:
                raise Exception("Output hopelessly screwed, retry") # blah blah no exceptions for control flow blah blah I'm not listening -- if it does this, it's an error
            return correction.strip()
        except:
            retries += 1
            print(f"Something went catastrophically wrong with this one. Investigate! Here's the completion:\n{completion}")
    return None
            
            
if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    inaccurate_qa_tuple = ("For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?", "The concept of a spherical Earth has been known for only about 1,000 years.", "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.")

    dissenting_reasoning = """Step 1. Analyze the Text: focus on the details provided about the history of spherical Earth.
Step 2. Understand the Question: the question's focus is on how long the concept of a spherical Earth has been known to at least a limited number of intelligent people.
Step 3. Compare the First Part of the Answer with the Text: check if the text supports the claim that the concept of a spherical Earth has only been known for about 1,000 years. It does, so this part is accurate.
Step 4. Compare the Second Part of the Answer with the Text: check if the text contradicts the claim that the concept of a spherical Earth has been known to at least a limited number of intelligent people for longer than 1,000 years. The text indicates this knowledge predates 2500 years ago, so this part is inaccurate.
Step 5. Final Judgment: Since the answer is not entirely accurate, the answer is inaccurate.  It's important to note that the second paragraph of the text does indeed mention a limited number of intelligent people knowing about the spherical Earth over 2,000 years ago, but this information is contradicted by the rest of the text which suggests the concept was unknown to these same "intelligent" people."""

    plan = """Step 1. Analyze the Text: Focus on the details provided about how long the concept of a spherical Earth has been known.
Step 2. Understand the Question: The question asks for the length of time the concept has been known to at least a limited number of intelligent people, not just anyone.
Step 3. Identify Inaccuracies in the Given Answer: The text provides evidence that the concept was known over 2,000 years ago (e.g., in paragraphs about the "early" ideas from Ancient Greece and Egypt). However, these early references may not be applicable to the specific question of when the concept was known by "at least a limited number of intelligent people," which might have a higher bar for knowledge due to its technical nature.
Step 4. Plan a Corrected Answer: To account for this distinction, one could adjust the phrasing of the answer to something like, "While there are records of early civilizations discussing spherical Earth concepts, it's likely that the concept was not universally understood or accepted as fact until around the time period mentioned in the text when it became commonly taught in schools and universities." This allows for the possibility that some individuals may have had an understanding of the spherical Earth before this time period, but emphasizes that broader knowledge and acceptance did not occur earlier than what's suggested by the text."""
    
    print("Begin HGWELLS test")
    result = regenerate_answer(inaccurate_qa_tuple, dissenting_reasoning, plan, logic_llm)
    
    
    # Example output:
    """The correct answer would be "It is likely that the concept of a spherical Earth was understood by at least a limited number of intelligent people around the time period mentioned in the text when it became commonly taught in schools and universities, which would put this knowledge after about 2500 years ago.". This accounts for the early references to spherical Earth concepts in the provided text while acknowledging the higher bar for universal understanding and acceptance suggested by the question's phrasing."""

            
            
            
            
            
            
            
            
            
            
