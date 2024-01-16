import re
from .questions_grammar import questions_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

# Answer generation code, begin.
# structure: define a series of helpers, then define the control flow, exception handling, retries etc. in a for loop that iterates over the processed sequences of paragraphs at the end in another cell

# Since some paragraphs can be much shorter
# First off, question generation.

# Each local LLM function essentially has 3 phases: prompt, regex to extract response, and reaction to that response.
# However I'm not going to build an abstraction for that because I need fine control.

# If any function fails to make things, it won't throw, it'll just return None.

# Strengths of open source AI: hella cheap, very customizable, you can call it as much as you want
# Downside: you need very good regexes to catch its outputs


def generate_short_questions(text, existing_question_tuples, logic_llm):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    if existing_question_tuples:
        existing_qs = "\n".join(
            [
                "Existing question: " + t[0] + "\nAnswer: " + t[1]
                for t in existing_question_tuples
            ]
        )  # stop duplicates from being generated
    else:
        existing_qs = None
    while not made_questions and (retries <= 5):
        question_prompt = f"""<s> [INST] You are an expert educational AI that, given a paragraph or two from a text, will create suitable educational questions based on the paragraphs. The questions you create will lean towards shorter questions that can be quickly answered with only a bit of thought — and which can be solved if the answerer knows the paragraphs provided by heart. Essentially: the question will test comprehension of real information in the paragraphs that would be worthy to teach. After the question, you will also write its answer. Your task includes first analyzing the text, thinking through and brainstorming which questions you will make and why. 

Each question you write (after your reasoning step is complete), MUST start on a new line with its question number followed by a bracket, ie, 1), or 2). This will then be followed by the question. This MUST be followed by "Answer: " followed by the question's answer. Each question must be separated by at least one new line. 

Some longer-form questions have already been generated from this text. You are to avoid reiterating exactly any of the questions present in this list:
\"\"\"
{existing_qs}
\"\"\"

Text to make questions from: \"\"\"{text}\"\"\"

You should aim to make 6 questions (at most), but if the text is too small or information-sparse for that many, you are allowed to write fewer. Do not explicitly mention the paragraphs OR "the text" in the questions themselves — just ask about the concepts related to the paragraphs. 

You will not mention the text explicitly in any questions you think of, since the questions you generate are intended to test people's knowledge of the information — when given the questions, they will not have the text on-hand.

[/INST]### Response: I will remember to follow the question and answer format given by:
\"\"\"
num) question contents.
Answer: question answer.
\"\"\"
All my questions will be directly answerable from the provided paragraphs, and will not rely on knowledge outside of what has been provided. Each of my short questions will have its answer written below it on a new line.

## Questions:
"""
        completion = llm_call(
            question_prompt,
            # max_tokens=2000,
            #stop=["</s>", "# Input:", "[INST]","### Instruction"],
            #echo=True,
            # grammar=questions_grammar,
            temperature=0.2,
            # repeat_penalty=0,
            # penalize_nl=False,
        )["choices"][0]["text"]

        # Extract questions
        response_pattern = re.compile(r"Questions:\n(.+)", re.IGNORECASE | re.DOTALL)
        generation = response_pattern.search(completion).group(1)
        # print("GENERATION:\n\n-------------------\n\n", generation)
        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(generation)
        print("GENERATION:\n\n-------------------\n\n", matches)
        if len(matches) > 0:
            made_questions = True
        else:
            retries += 1
    if retries > 5:
        return None

    for match in matches:
        questions.append((match[0].strip(), match[1].strip(), text))

    return questions


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    text = """The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.

The earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."""
    print("Begin HGWELLS test")
    result = generate_short_questions(text, logic_llm)
