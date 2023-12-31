import re
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .regenerate_answer_constrain_to_text_grammar import (
    regenerate_answer_constrain_to_text_grammar,
)
from .strip_steps import strip_steps


# Answer regeneration (triggered after a fact-check fails for reason of "inaccurate").
# NOTE This line of revision was abandoned after they began to lead to nothingburgers of questions after enough rerolls.
def regenerate_answer_constrain_to_text(qatuple, dissenting_reasoning, plan, logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""You are an expert educational AI. Someone has written an answer to a question (this question is based on a few provided paragraphs of text) but their answer includes information that's not provided by the text, and thus it might be flawed. Given these paragraphs, a question based on the paragraphs, the flawed answer to the question, and the explanation of why the answer deviates from the text, you will write the correct answer to the question that only uses info in the text. 

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Allegedly incorrect answer to the question (you must constrain this answer to only the information provided by the text): \"\"\"{qatuple[1]}\"\"\"

Reasoning as to why the answer goes off the rails: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"


### Response:
## Reasoning and thought process:
{plan}

## New answer (do not mention the text):
The constrained answer would be \"\"\""""
        try:
            completion = logic_llm(
                decision_prompt,
                max_tokens=3000,
                stop=["</s>", "# Input:"],
                grammar=regenerate_answer_constrain_to_text_grammar,
                temperature=0.2,
                echo=True,
            )["choices"][0]["text"]
            # print(completion)
            completion_pattern = re.compile(
                r"New answer \(do not mention the text\):\nThe constrained answer would be \"\"\"(.+?)\"\"\"",
                re.DOTALL,
            )
            correction = completion_pattern.search(completion).group(1)
            return correction.strip()
        except:
            retries += 1
            print(
                f"Something went catastrophically wrong with this one. Investigate! Here's the completion:\n{completion}"
            )


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    q_test = [
        (
            ") Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            ") Identify and explain changes in human understanding throughout history regarding the age of the Earth.",
            "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            ") Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.",
            "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            ") Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
            "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
    ]

    dissenting_reasoning = """Step 1. Analyze the Text: Focus on the main points stated in the text about why we know Earth is approximately 8000 miles in diameter, how its distance from the sun varies, etc.
Step 2. Understand the Question: The question's focus is about why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies using specific scientific principles.
Step 3. Compare the Answer with the Text: Check if the text mentions GPS data, elliptical orbit around the sun or points of closest approach and farthest departure, etc. It does not, so this part is adding extra information not found in the text and is thus irrelevant. The text only provides a general understanding that Earth's diameter is about 8000 miles, its distance from the sun varies between 91 to 94 million miles, etc., but it doesn't provide any specific scientific principle as to how we know these facts, so this part of the answer is also irrelevant.
Step 4. Final Judgment: Since the answer mentions concepts not present in the original text, it is irrelevant."""

    plan = """Step 1. Analyze the Text: Focus on details provided about Earth's diameter and distance from the sun.
Step 2. Understand the Question: Identify the question's focus on how we know these facts using specific scientific principles.
Step 3. Identify Flawed Part of the Answer: From the reasoning, I note that the answer mentions GPS data, elliptical orbit around the sun, and points of closest approach and farthest departure, which are not mentioned in the text. The text only provides a general understanding that Earth's diameter is about 8000 miles, its distance from the sun varies between 91 to 94 million miles, etc., but it doesn't provide any specific scientific principle as to how we know these facts.
Step 4. Plan Revised Answer: Based on the text, I will write a new answer that only mentions the general understanding of Earth's diameter and distance from the sun, without mentioning GPS data, elliptical orbit around the sun, or points of closest approach and farthest departure."""

    print("Begin HGWELLS test")
    result = regenerate_answer_constrain_to_text(
        q_test[2], dissenting_reasoning, plan, logic_llm
    )

    # Example output:
    """The constrained answer would be "We know about Earth's diameter using measurements of its circumference made using traditional methods. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure." This answer uses only information provided by the text."""
    # Yay! It works!
