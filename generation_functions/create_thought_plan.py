import re
from .thought_plan_grammar import thought_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL


def extract_name(str):
    # Regular expression to match 'Name:' followed by any characters until the end of the line
    name_regex = r"^Name:\s*(.*)$"

    # Searching in the multiline string
    match = re.search(name_regex, str, re.MULTILINE)

    if match:
        name = match.group(1)
        print(f"Extracted name: {name}")
        return name
    else:
        print("No name found")


def create_thought_plan(qatuple, character, logic_llm, assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # use regex to extract charname
    charname = extract_name(character)

    cot_prompt = f"""<s> [INST] You are an expert logical reasoning and question-solving AI. Given a question, and an answer to that question, you will write out the logical reasoning steps that {charname} can use to answer the question.

You should first describe a plan which includes decomposing the problem into smaller ones, then execute that plan step-by-step, and then present a final answer to the original question. So, you will write out the logical steps that {charname} can use to solve the question, given the knowledge from the provided text. Try to begin each line with something like "Realize" or "Recall".


## Information:

Text the question and answer were sourced from (Judge Elias Hawthorne does not have access to this text, but does know its information; you can use insights from it, but don't mention "the text" by name):
\"\"\"
A communi observantia non est recedendum; et minime mutandæ sunt quæ
certam interpretationem habent.

     We must not depart rashly from common observance; and those
     points are by no means to be changed, which admit of a
     clear interpretation.


A digniori fieri debet denominatio et resolutio.

     Title and acquittal ought to proceed from the more worthy
     person.


A facto ad jus non datur consequentia.

     A fact does not necessarily constitute a right.


A non posse ad non esse sequitur argumentum necessarie negativè, licet
non affirmativè.

     From the impossibility of a thing to its nonentity, the
     argument or proof follows of necessity, negatively, though
     not affirmatively.


A principalioribus seu dignioribus est inchoandum.

     We must begin with the more noble, or more important
     matters; or in other words, we must attend to things of the
     greatest consequence.


A rescriptis valet argumentum.

     An argument or proof is valid, from a rescript or letter
     of a prince or emperor, making answers to petitions, or
     other applications; or more laconically, the king’s answer,
     wherein he signifies his pleasure, amounts to a law, and is
     not to be disputed.


A verbis legis non est recedendum.

     We must not depart from the letter of the law.


Ab assuetis non fit injuria.

     From customary treatment no injury happens; or habit is not
     an injury.


Absoluta sententia expositore non indiget.

     An absolute or perfect opinion or sentence, needs no
     expounder, exposition, or explanation.
\"\"\"

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

[/INST]### Response:
## Reasoning Steps:
Step 1. Formulate a plan to understand the meaning of 'A digniori fieri debet denominatio et resolutio' by defining it, then interpreting the definition in the context of law.
Step 2. Recall that the phrase 'A digniori fieri debet denominatio et resolutio' translates to "Title and acquittal ought to proceed from the more worthy person."
Step 3. Realize that this principle emphasizes the importance of the worthiness or merit of the individual in legal decisions related to titles and acquittals.
Step 4. Understand that this implies a belief that justice and legal rights should be administered by those who are most competent and deserving.
Step 5. Conclude that the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio' is the emphasis on the importance of merit and worthiness in making legal decisions about titles and acquittals.


## Information:

Text the question and answer were sourced from (Carlos Mendez does not have access to this text, but does know its information; you can use insights from it, but don't mention "the text" by name): 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

[/INST]### Response:
# Reasoning:
Step 1. Formulate a plan to answer the question about the amount of earth excavated during the Panama Canal construction.
Step 2. Recall that "how much earth was excavated" can be decently answered by stating the volume of earth removed.
Step 3. Recall that over 200 million cubic yards of earth were removed.
Step 4. Conclude that the answer to the question is over 200 million cubic yards.


## Information:

Text the question and answer were sourced from ({charname} does not have access to this text, but does know its information; you can use insights from it, but don't mention "the text" by name): 
\"\"\"
{qatuple[2]}
\"\"\"

Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\"

[/INST]### Response:
## Reasoning (only use as many logical steps as you must in order to solve the problem):
"""

    # Logical progression of steps that {charname} can use to answer the question
    completion = llm_call(
        prompt=cot_prompt,
        # max_tokens=4000,
        # repeat_penalty=0,
        # penalize_nl=False,
        #stop=["</s>", "# Input:", "[INST]","### Instruction"],
        #echo=True,
        # grammar=thought_plan_grammar,
        temperature=0.2,
    )["choices"][0]["text"]
    # print("COMPLETION:\n\n----------------------")
    # # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"## Reasoning \(only use as many logical steps as you must in order to solve the problem\):\n(Step 1\..+)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)

    return generation


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [
        (
            "Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Identify and explain changes in human understanding throughout history regarding the age of the Earth.",
            "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.",
            "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
            "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
    ]

    character = """Name: Dr. Samuel Blackwell
Traits: Knowledgeable, Passionate, Confident, Dedicated, Controversial, Vulnerable, Fearful of misunderstanding, Faithful, Dogmatic, Religious, Scientific, Determined, Unwavering
Dialogue Examples:
Stranger: "What's your backstory?"
Dr. Samuel Blackwell: "Ah, my journey," I begin, leaning back in my chair, "it started with a deep-seated faith, you see. Born into a religious household, the Bible was our guiding light. But as I grew older and began to study theology, questions arose." I pause, frowning slightly. "How could the Earth be just a few thousand years old when geological evidence pointed towards millions? The discrepancy troubled me greatly."
Stranger: "What's your personality?"
Dr. Samuel Blackwell: "I am a man of science, driven by facts and evidence," I say firmly, "but my faith is not easily shaken. It has led me down a path of discovery, challenging traditional beliefs about the age of our planet." My eyes light up as I recall past debates, "But it's also made me a controversial figure. Many see my work as blasphemous, questioning God's word. Yet, I believe in the power of evidence and truth. Despite the backlash, I remain unwavering." I sigh, looking thoughtful, "Yet, there's a vulnerability too. The fear of being misunderstood or dismissed due to my challenges to religious orthodoxy... it weighs heavily on me.\""""

    scenario = """Set against the backdrop of his cluttered office at a university, Dr. Samuel Blackwell is deep in thought, reviewing his latest research on geological evidence when he is approached by Sarah, a curious student who wants to know more about how human understanding has changed regarding the age of the Earth throughout history."""

    print("Begin HGWELLS test")
    # Make card for good history question
    d = create_thought_plan(q_test[1], character, logic_llm)


# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.
