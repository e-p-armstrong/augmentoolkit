import re
from .answer_constrain_to_text_plan_grammar import answer_constrain_to_text_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps


def make_regenerate_answer_constrain_to_text_plan(
    qatuple, dissenting_reasoning, logic_llm
):
    retries = 0
    while retries < 5:
        decision_prompt = f"""You are an expert educational AI that is going to think through a plan for a revised answer (the current one is flawed). Someone has written an answer to a question (this question is based on a few provided paragraphs of text) but their answer includes information that's not provided by the text, and thus it might be flawed. You will plan out and think through, step-by-step, a revision to the answer, which will only use provided information. Given these paragraphs, a question based on the paragraphs, the flawed answer to the question, and the explanation of why the answer deviates from the text, you will plan out and think through a correct answer to the question.

Right now, you will PLAN OUT and THINK THROUGH different possibilities for a new answer that answers the question using only information in the provided text.

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Allegedly irrelevant answer to the question (you must constrain this answer to only the information provided by the text): \"\"\"{qatuple[1]}\"\"\"

Reasoning as to why the answer goes off the rails: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

For instance, for a paragraph about the diet of dinosaurs that specifies T. rexes eat meat, a question about what T. rexes ate, an answer stating that T. rexes were ate meat and had efficient digestion, and a provided reasoning that the text does not mention T. rexes' digestion, you might write the following:
Step 1. Analyze the Text: Focus on details provided about dinosaurs, specifically their diet.
Step 2. Understand the Question: Identify the question's focus on the diet of T. rexes.
Step 3. Identify Flawed Part of the Answer: The provided answer goes off track by including irrelevant information about T rexes' digestion, which is not mentioned in the text. My revised answer will omit this irrelevant information.
Step 4. Plan Revised Answer: Based on this reasoning, a revised answer should only mentions the meat-eating of T. rexes.

You are to use the above example as a reference, while you plan out a revised version of the answer \"\"\"{qatuple[1]}\"\"\" with regards to the text and reasoning provided earlier.

### Response:
## Reasoning and thought process:
"""
        try:
            completion = logic_llm(
                decision_prompt,
                max_tokens=3000,
                stop=["</s>", "# Input:"],
                echo=True,
                grammar=answer_constrain_to_text_plan_grammar,
                temperature=0.2,
            )["choices"][0]["text"]

            # print("DEBUG\n\n")
            # print(completion)
            completion_pattern = re.compile(
                r"Reasoning and thought process:\n(.+)", re.DOTALL
            )
            correction = completion_pattern.search(completion).group(1)
            return correction
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
    # Old thing below. Maybe use if I want to test with bad reasoning
    #     """Step 1. Analyze the Text: Focus on the main points stated in the text about the Earth's diameter, distance from the sun, and its spherical shape.
    # Step 2. Understand the Question: The question is asking about specific scientific principles used to explain why we know the Earth is approximately 8000 miles in diameter and how its distance from the sun varies.
    # Step 3. Compare the First Part of the Answer with the Text: Check if the text states any scientific principles about measuring Earth's diameter. It doesn't, so this part seems irrelevant.
    # Step 4. Compare the Second Part of the Answer with the Text: Check if the text states how we know that the Earth is approximately 8000 miles in diameter. It does mention "measurements of its circumference made using GPS data" as the method for determining this, which aligns with what is mentioned in the answer, so this part of the answer reflects aspects of the text and is relevant.
    # Step 5. Compare the Third Part of the Answer with the text: Check if the text mentions any scientific principles explaining why Earth's distance from the sun varies. It doesn't mention an elliptical orbit or a varying point of closest approach and farthest departure, so this part seems irrelevant.
    # Step 6. Final Judgment: Since only part of the answer reflects aspects of the text (specifically, how we know Earth's diameter), it is "Irrelevant"."""

    print("Begin HGWELLS test")
    result = make_regenerate_answer_constrain_to_text_plan(
        q_test[2], dissenting_reasoning, logic_llm
    )


# I could definitely just use a grammar to add the question generation to this step
