import re
from .make_regenerate_question_plan_grammar import make_regenerate_question_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps

# Question regeneration (triggered after a relevance-check fails on the question).


# NOTE does not explicitly guard against duplicates. I am avoiding that right now due to the limited drawback of a duplicate or two, and the increased complexity of adding that check.
def make_regenerate_question_plan(qatuple, dissenting_reasoning, logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""<s> [INST] You are an expert educational AI. Someone has written a question that was supposed to be based on the provided paragraphs of text, but actually requires significant knowledge outside these paragraphs to answer. Your goal is to write a good, comprehensive plan for generating a revised question that addresses the criticism given. Given these paragraphs, the flawed question based on the paragraphs, and the explanation of why the question is flawed, you will PLAN OUT and THINK THROUGH different possibilities for a new question, one which will only require information from the paragraphs to solve. 

Paragraphs: \"\"\"{qatuple[2]}\"\"\"

Flawed question (based on text): \"\"\"{qatuple[0]}\"\"\"

Reasoning as to why the question is irrelevant: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

Your plan/thought process should think through how to make the question more constrained to the content of the paragraphs.

For instance, if the paragraph is about the life cycle of a butterfly, and the flawed question asks, "What are the genetic mechanisms that cause color changes in butterflies during metamorphosis?" which is not covered in the text, you might write:
Step 1. Analyze the Reason for Flaw: Understand why the original question is flawed - it assumes knowledge of genetics not provided in the text.
Step 2. Identify Key Concepts in Paragraphs: Focus on the stages of the butterfly life cycle as detailed in the text.
Step 3. Generate a New Question Idea: Consider a question, similar to the original but answerable using the text's information, that revolves around the observable changes during the life cycle stages.
Step 4. Refine the Question: An explicit question, which does not directly refer to the text, might be: "Describe the transformations a caterpillar undergoes during metamorphosis."
Step 5. Ensure Alignment with Text: Verify that the new question is answerable solely based on the information provided in the paragraphs. The text explicitly and in detail describes the changes a caterpillar undergoes during metamorphosis, so the new question is answerable based on the text. Thus, I will not rewrite the draft question in this step.
Step 6. End of reasoning.

Please now apply the above method to the provided text and question, and write out your reasoning and thought process.

[/INST]### Response:
## New question plan:
"""
        completion = llm_call(
            prompt=decision_prompt,
            # max_tokens=4000,
            # repeat_penalty=0,
            # penalize_nl=False,
            #stop=["</s>", "# Input:", "[INST]"],
            #echo=True,
            # grammar=make_regenerate_question_plan_grammar,
            temperature=0.2,
        )["choices"][0]["text"]

        # print("DEBUG\n\n")
        # print(completion)
        completion_pattern = re.compile(r"New question plan:\n(.+)", re.DOTALL)
        correction = completion_pattern.search(completion).group(1)
        return correction.strip()


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

    dissenting_reasoning = """Step 1. Analyze the Text: The text provides information about the creation of the universe, its spherical shape, and its rotation on its axis. It also mentions that the earth orbits around the sun in an elliptical path.
Step 2. Understand the Question: The question asks to explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies using specific scientific principles.
Step 3. Compare the First Part of the Question with the Text: The text does not contain any information about the size of the earth or its distance from the sun. So, this part of the question is irrelevant.
Step 4. Compare the Second Part of the Question with the Text: The text mentions that the Earth orbits around the sun in an elliptical path. This implies that the distance between the Earth and the Sun varies during different parts of the orbit. So, this part of the question is relevant.
Step 5. Final Judgment: Since the text does contain information about how the earth's distance from the sun varies, I conclude that the question is relevant."""
    print("Begin HGWELLS test")
    result = make_regenerate_question_plan(q_test[2], dissenting_reasoning, logic_llm)
