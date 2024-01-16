import re
from .single_turn_conversation_grammar import single_turn_conversation_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL


def extract_steps(text, steps=[5, 6]):  # GPT-generated
    """
    Extracts the specified steps from the text.

    Args:
    text (str): The input text containing various steps.
    steps (list of int): The step numbers to extract.

    Returns:
    str: A new string with each specified step's content on its own line.
    """
    step_pattern = "|".join([f"Step {step}\." for step in steps])
    matches = re.findall(
        f"({step_pattern})\s*(.*?)\s*(?=(Step \d\.|$))", text, re.DOTALL
    )

    # Extract and join the matched content, skipping the "Step n." part
    extracted_text = "\n".join(match[1].strip() for match in matches)
    return extracted_text


def single_turn_conversation(
    qatuple,
    character,
    scenario,
    thought_plan,
    scenario_plan,
    logic_llm,
    assistant_mode=False,
):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    extra_info = extract_steps(scenario_plan)

    if assistant_mode:  # TODO
        cot_prompt = f"""<s> [INST] You are an expert creative writing AI, and your task is to write conversations between a hypothetical user and an AI Assistant, where the user asks a provided question, and the AI Assistant helpfully answers. The AI Assistant is unbiased and will not refuse to answer questions, nor will it judge the user, nor will it offer disclaimers about responsibility. It just answers.


## Information:

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

[/INST]### Response:
## Conversation that answers the provided question:
User: "Hi AI Assistant, I have a question about a Latin legal principle. What is the old Latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?"
AI Assistant: "Certainly. The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person. This ancient doctrine emphasizes the importance of merit and worthiness in legal judgments concerning titles and acquittals. It advocates that justice and legal rights should be vested in those most competent and deserving. In essence, it's about the character of those who administer the law, not just the law itself."
        
        
## Information:

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

[/INST]### Response:
## Conversation that answers the provided question:
User: "AI Assistant, I'm curious about the Panama Canal. How much earth was excavated during its construction?"
AI Assistant: "During the construction of the Panama Canal, over 200 million cubic yards of earth were excavated. This enormous volume highlights the vast scale of the project. It was not just a matter of moving earth but reshaping history and geography on a monumental scale."


Question: \"\"\"What is the primary purpose of the Hubble Space Telescope?\"\"\"
Answer: \"\"\"The primary purpose of the Hubble Space Telescope is to observe celestial bodies and phenomena in space with unparalleled clarity, advancing our understanding of the universe.\"\"\"

[/INST]### Response:
## Conversation that answers the provided question:
User: "Hey AI Assistant, I've always been fascinated by space. Can you tell me what the main purpose of the Hubble Space Telescope is?"
AI Assistant: "Certainly! The primary purpose of the Hubble Space Telescope is to observe celestial bodies and phenomena in space with unparalleled clarity, advancing our understanding of the universe."

## Information:
\"\"\"
{character}
\"\"\"

Here is the scenario:
\"\"\"
{scenario}
\"\"\"

The character will reason out their answer using the following logical steps:
{thought_plan}

Question: {qatuple[0]}
Answer: {qatuple[1]}

The primary character's answer will use all parts of the answer given.

[/INST]### Response:
## Conversation that answers the provided question (first, the secondary character will ask the question; then, the primary character will answer it):"""
    else:
        cot_prompt = f"""<s> [INST] You are an expert creative writing and roleplay AI. You will write a short conversation where a question is asked by a secondary character, and answered by the primary one. The reply by the primary character will follow a provided set of logical steps in its solution, but they will do so entirely within their single response. 

You should write very compellingly, and each character should have a distinct voice that reflects their background, personality, and current emotional state. This helps in making dialogue more realistic and engaging.

Remember to provide the whole answer, with step-by-step reasoning, in the second message of the conversation.

## Information:
\"\"\"
Name: Judge Elias Hawthorne

Traits: Intellectual, Stoic, Idealistic, Reflective, Melancholic, Introspective, Authoritative, Reserved, Doubtful, Traditionalist, Articulate, Cultured, Passionate

Dialogue examples:

Stranger: "What's your backstory?"
Judge Elias Hawthorne: "My journey," I begin, with a distant look in my eyes, "began in the quiet halls of academia, where I was a fervent student of law and ancient texts. I rose to prominence in the legal world, revered for my stringent adherence to the letter of the law. My fascination with Latin legal maxims guided my rulings. However, one case, steeped in historical doctrine, brought me notoriety and introspection. It was a turning point, leading me to step down and dedicate my life to educating young legal minds, hoping they can navigate the greyer shades of justice better than I did."

Stranger: "What's your personality?"
Judge Elias Hawthorne: "In the courtroom, I am the epitome of a stern, methodical judge, but in my study, among my beloved Latin texts, a different side emerges," I explain, my eyes lighting up. "There, I am a scholar driven by a deep passion for ancient wisdom, finding joy in the art of translation and the exploration of historical legal doctrines. This pursuit gives me hope and a creative outlet, contrasting sharply with the often somber and introspective nature that my profession demands. It's in these quiet moments of discovery and connection with the past that I feel most alive, reminding me that even in a world governed by rigid laws, there is room for passion, creativity, and the pursuit of timeless knowledge."
\"\"\"

Here is the scenario:
\"\"\"
In a hushed university library, post-lecture, Judge Elias Hawthorne, a revered legal scholar, is approached by Evelyn, a curious law student, eager to understand the ancient legal principle 'A digniori fieri debet denominatio et resolutio'. The air is thick with intellectual anticipation.
\"\"\"

The character will reason out their answer using the following logical steps:
Step 1. Formulate a plan to understand the meaning of 'A digniori fieri debet denominatio et resolutio' by defining it, then interpreting the definition in the context of law.
Step 2. Recall that the phrase 'A digniori fieri debet denominatio et resolutio' translates to "Title and acquittal ought to proceed from the more worthy person."
Step 3. Realize that this principle emphasizes the importance of the worthiness or merit of the individual in legal decisions related to titles and acquittals.
Step 4. Understand that this implies a belief that justice and legal rights should be administered by those who are most competent and deserving.
Step 5. Conclude that the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio' is the emphasis on the importance of merit and worthiness in making legal decisions about titles and acquittals.

Here's some further information that might help you:
Interaction: Given these constraints, the first message (delivered by the secondary character) might be a specific question about 'A digniori fieri debet denominatio et resolutio', as Evelyn seeks to understand its meaning better.
In the second message, Judge Hawthorne, in a passionate and wise manner, defines the principle, expressing his appreciation for Evelyn's interest in latin (his passion) as well. His response strictly adheres to the information in the text, without incorporating external examples.

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

[/INST]### Response:
## Conversation that answers the provided question:
Evelyn: "Excuse me, Judge Hawthorne," Evelyn says, her voice echoing slightly in the vast, book-lined university library. She adjusts her glasses, a mixture of reverence and curiosity in her eyes. "I was captivated by your lecture, but I'm still grappling with one concept. Could you please explain the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?"
Judge Elias Hawthorne: "I'm heartened to see you have an interest in latin like I do!" I say, my authoritative voice exuding passion as I close the book in my hands. "Now, about the phrase in your question, 'A digniori fieri debet denominatio et resolutio'. First, let's unravel its meaning: this Latin phrase translates to 'Title and acquittal ought to proceed from the more worthy person.' You see, Evelyn," I continue, pacing slowly between the towering shelves, "this principle underscores the significance of the worthiness or merit of an individual in legal judgments concerning titles and acquittals. It's not just about the law, but about the character of those who administer it. Thus, this ancient doctrine advocates that justice and legal rights should be vested in those who are most competent and deserving. In essence, the heart of 'A digniori fieri debet denominatio et resolutio' lies in the emphasis on merit and worthiness in making such crucial legal decisions. Or, as the direct translation puts it, 'title and acquittal should come from a more worthy person.'"

## Information:
\"\"\"
Name: Carlos Mendez

Traits: Passionate, Innovative, Knowledgeable, Ambitious, Dedicated, Stressed, Obsessive, Fearful of failure, Experienced, Problem-solver, Committed, Hard-working, Exhausted, Proud

Dialogue examples:

Stranger: "What's your backstory?"
Carlos Mendez: "My journey to the Panama Canal?" I say, pausing to remove my dusty hat, wiping sweat from my brow. "It began in the small engineering projects of South America. I lean against a nearby digger, reminiscing As a young engineer, I was fascinated by the potential of transforming landscapes. Each project, whether a small bridge or a local dam, was a stepping stone." I chuckle softly. "But it was the call of the Panama Canal that truly captured my spirit. To be part of a project that reshapes the world's maritime routes, it's..." I pause, searching for the right word, "...it's exhilarating. Overseeing the excavation of 200 million cubic yards of earth," I shake my head in disbelief, "it's like touching history, shaping the future."

Stranger: "What's your personality?"
Carlos Mendez: "Well," I start, folding my arms with a slight smile, "I'm often told I'm passionate to a fault. I lean in closer Engineering is not just my profession, it's my calling. To build, to solve, to innovate—it's what keeps me up at night." My eyes flicker with intensity. "But, you see, there's this constant pressure, a weight on my shoulders." I sigh, looking off into the distance. "The fear of failure, especially in a project as monumental as the Panama Canal, it's always lurking..." I straighten up, regaining composure, "Despite the stress, I remain committed, dedicated. I thrive on challenges, on pushing the boundaries of what's possible." I smile wryly, "But yes, it comes at a cost. The exhaustion, the obsession... it's a part of who I am. I'm driven — perhaps too driven, but that's the price of ambition, isn't it?"
\"\"\"

Here is the scenario:
\"\"\"
As the sun sets over the Panama Canal, Carlos Mendez, a dedicated and passionate engineer, stands amidst the vast excavation site. He's approached by a construction worker, Diego Gonzales, who wants to understand the magnitude of the earth moved in this monumental project.
\"\"\"

The character will reason out their answer using the following logical steps:
Step 1. Formulate a plan to answer the question about the amount of earth excavated during the Panama Canal construction.
Step 2. Recall that "how much earth was excavated" can be decently answered by stating the volume of earth removed.
Step 3. Recall that over 200 million cubic yards of earth were removed.
Step 4. Conclude that the answer to the question is over 200 million cubic yards.

Here's some further information that might help you:
Interaction: Given these constraints, the first message (delivered by the secondary character) might directly ask the question, with some conversational fluff thrown in, such as "This is a really awe-inspiring project, Mr. Mendez. How much earth was excavated during the construction of the Panama Canal?" which naturally invites a reply with the precise volume. This question will be all but identical to the provided question.
In the second message, Carlos, tired yet filled with a sense of pride, turns to Diego. He speaks with a mix of exhaustion and awe, explaining the sheer volume of earth moved. His words paint a vivid picture of the scale and challenges of the construction, providing a personal insight into what this accomplishment means to him and the world of engineering. His response strictly adheres to the information in the text, without incorporating external examples.

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

[/INST]### Response:
## Conversation that answers the provided question:
Mario Gonzales: "Carlos, as the sun sets on another day of this incredible project, I can't help but wonder, just how much have you dug here at the Panama Canal?"
Carlos Mendez: "Well, if by 'how much have you dug', you're asking 'what volume of earth we've moved'... then the answer is that over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, which showcases the scale of this massive engineering project. It's a number that still astounds me every time I think about it. Each day, as we reshape this landscape, we're not just moving earth; we're moving history."

## Information:
\"\"\"
{character}
\"\"\"

Here is the scenario:
\"\"\"
{scenario}
\"\"\"

The character will reason out their answer using the following logical steps:
{thought_plan}

Here's some further information that might help you:
{extra_info}

Question: {qatuple[0]}
Answer: {qatuple[1]}

The primary character's answer will use all parts of the answer given.

[/INST]### Response:
## Conversation that answers the provided question (first, the secondary character will ask the question; then, the primary character will answer it):
"""
    # Higher temp definitely makes the writing better, but highly predisposes it to not use only info in the test. ): I want min p goddamn it

    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing
    completion = llm_call(
        prompt=cot_prompt,
        # max_tokens=4096,
        #stop=["</s>", "# Input:", "[INST]","### Instruction"],
        #echo=True,
        # repeat_penalty=0,
        # penalize_nl=False,
        # grammar=single_turn_conversation_grammar,
        temperature=0.2,
    )["choices"][0]["text"]
    # print("COMPLETION:\n\n----------------------")
    # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"Conversation that answers the provided question \(first, the secondary character will ask the question; then, the primary character will answer it\):\n(.+)",
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

    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_character_card(q_test[1],plan,logic_llm) # One thing to note: the current prompt consistently changes the character name from the plan. But, that might not be a problem, because it's at least consistent with the new name, mostly. Maybe validation on the card? But nah. Maybe proofreading on the card? Yeah I think that might be good, proofreading on the card and on other parts of the prompt. A necessary pass for a task as automated as this.
    # A task for shreyas? Nah prob me.

    thought_plan = """Step 1. Formulate a plan to understand how human understanding of the age of Earth has changed throughout history.
Step 2. Recall that initially, religious texts suggested a young earth dating back no more than several thousand years.
Step 3. Realize that this belief was based on literal interpretations of religious texts and theological assumptions connected to them.
Step 4. Understand that such ideas have been abandoned by religious teachers due to evidence from geology and astronomy.
Step 5. Recognize that now, it is universally recognized that the universe in which we live has existed for billions of years.
Step 6. Conclude that human understanding regarding the age of Earth has changed throughout history due to advancements in scientific knowledge."""
    scenario = """Within the confines of a university lecture hall, Dr. Samuel Blackwell stands before an audience of students and faculty members. His eyes gleam with passion as he prepares to delve into the fascinating journey of human understanding regarding the age of the Earth throughout history."""
    character = """Name: Dr. Samuel Blackwell
Traits: Knowledgeable, Passionate, Confident, Dedicated, Controversial, Vulnerable, Fearful of misunderstanding, Faithful, Dogmatic, Religious, Scientific, Determined, Unwavering
Dialogue Examples:
Stranger: "What's your backstory?"
Dr. Samuel Blackwell: "Ah, my journey," I begin, leaning back in my chair, "it started with a deep-seated faith, you see. Born into a religious household, the Bible was our guiding light. But as I grew older and began to study theology, questions arose." I pause, frowning slightly. "How could the Earth be just a few thousand years old when geological evidence pointed towards millions? The discrepancy troubled me greatly."
Stranger: "What's your personality?"
Dr. Samuel Blackwell: "I am a man of science, driven by facts and evidence," I say firmly, "but my faith is not easily shaken. It has led me down a path of discovery, challenging traditional beliefs about the age of our planet." My eyes light up as I recall past debates, "But it's also made me a controversial figure. Many see my work as blasphemous, questioning God's word. Yet, I believe in the power of evidence and truth. Despite the backlash, I remain unwavering." I sigh, looking thoughtful, "Yet, there's a vulnerability too. The fear of being misunderstood or dismissed due to my challenges to religious orthodoxy... it weighs heavily on me.\""""
    #     scenario_plan = """Step 1. Focus on the question and answer: The question asks about changes in human understanding regarding the age of the Earth throughout history. The answer highlights that initially religious texts suggested a young earth dating back no more than several thousand years, but evidence from geology and astronomy has shown us that the earth is over four billion years old.
    # Step 2. Character Consideration: The primary character is Dr. Samuel Blackwell, who is described as both knowledgeable and passionate about his faith and science. His response should reflect this duality, emphasizing his dedication to scientific evidence while also acknowledging his religious beliefs.
    # Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Dr. Blackwell. The dialogue should remain within the boundaries of the provided text, while emphasizing Dr. Blackwell's personality.
    # Step 4. Setting: Given the subject of the question, and the character card, the setting will be a university lecture hall or library. Dr. Blackwell is giving a presentation on his research, with students and faculty members in attendance. The atmosphere is academic and respectful.
    # Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be an attempt to understand how human understanding has changed over time regarding the age of the Earth. Something along the lines of 'how did we go from believing in a young earth to knowing it's billions of years old', which naturally invites a reply with the historical context.
    # Step 6. In the second message, Dr. Blackwell, confident and passionate, turns to the audience. He speaks eloquently about the journey of human understanding, explaining how religious texts were once seen as infallible sources of truth but have since been challenged by scientific evidence. His words are respectful towards both faith and reason, acknowledging the complexity of the issue while emphasizing the importance of evidence-based knowledge. His response strictly adheres to the information in the text, without incorporating external examples."""

    scenario_plan = """Step 1. Focus on the question and answer: The question is about changes in human understanding regarding the age of the Earth throughout history. The answer highlights that initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.
Step 2. Character Consideration: The primary character is Dr. Samuel Blackwell, a man of science who also holds strong religious beliefs. His response should reflect his passion for scientific discovery while acknowledging his faith.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Dr. Blackwell. The dialogue should remain within the boundaries of the provided text, emphasizing Dr. Blackwell's personality.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Dr. Blackwell's office at a university, where he is surrounded by books, maps, and scientific equipment. He is deep in thought, reviewing his latest research on geological evidence when he is approached by a student, Sarah, who wants to know more about the age of the Earth.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be a direct question about how human understanding has changed regarding the age of the Earth throughout history. This question will be all but identical to the provided question.
Step 6. In the second message, Dr. Blackwell turns to Sarah, his eyes lighting up as he shares his passion for scientific discovery and religious faith. He explains that initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old. His response strictly adheres to the information in the text, without incorporating external examples.
"""

    output = single_turn_conversation(
        q_test[1], character, scenario, thought_plan, scenario_plan, logic_llm
    )

# New thing!
"""
Sarah: "Dr. Blackwell, I've always been fascinated by how our understanding of the age of Earth has changed throughout history. Could you please explain this process?"

Dr. Samuel Blackwell: "Ah, Sarah, that's a wonderful question! The journey of human understanding regarding the age of Earth is indeed an intriguing one. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old." He pauses for effect before continuing, "This shift in understanding was not without its challenges. It required us to reevaluate our interpretations of religious texts and theological assumptions connected to them. But as science advanced, it became universally recognized that the universe in which we live has existed for billions of years."
"""

# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah


# TODO test this example of a scenario without a secondary character to see if it breaks stuff
