import re

# from .multi_turn_conversation_grammar import multi_turn_conversation_grammar
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
from .extract_name import extract_name
import random

# all characters in this prompt are over 18

# Explanation of wtf the first few-shot example is:
# No I do not have a teacher-student fetish, the reason why Elise is a teacher is an adaptation to the following three facts:
# 1. This tool is meant to be able to generate data for training ERP bots by default
# 2. This tool is also meant to be able to take in educational material by default
# 3. When generating characters that would know about educational material, the model tends to generate academics or professors in that field, talking to students.
# Given these facts, we clearly need to prompt the model to be able to generate horny teachers, or else it's going to just do it poorly when it realizes it has a sexualized character that's also a teacher. I didn't want to choose this, the constraints of the problem forced me to.


def extract_steps(text, steps=[2, 4, 5]):
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


def extract_first_words(character_name, text):
    # Regular expression pattern to extract first word after the character's name
    pattern = rf"{character_name}: \"(\w+)"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches


# async def multi_turn_conversation(
#     qatuples, character, scenario, scenario_plan, engine_wrapper, assistant_mode=False
# ):
#     """
#     Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

#     Format: Question: [question]\n\n
#     """


#     print("--CONV STARTERS FILTERED--")
#     print(conv_starters_filtered)

#     if assistant_mode:
#         character = "AI Assistant"
#         scenario = "A conversation between a helpful AI Assistant, and a user."
#         scenario_plan = "N/A"
#         charname = "AI Assistant"
#         cot_prompt = f""""""
#     else:
#         extra_info = extract_steps(scenario_plan)
#         cot_prompt = f""""""

#     # NOTE: Very rarely, the first message of this conv will just be part of the character card, causing the conv to not make much sense. The cause of this is likely the fact that Elise quotes her character card in her first message. However, referencing the character card in this way also makes characters act as they are described, which is deemed advantageous enough that I am not changing this for now.
#     # I get the sense that LLMs can learn relationships and connections between parts of the prompt, even if they're quite far apart, if you give them examples like this. It's fascinating to see how each part of the prompt has consequences -- sometimes unintended ones.

#     # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing

#     sampling_params = 
#     completion = await engine_wrapper.submit(cot_prompt, sampling_params)
#     # print("COMPLETION:\n\n----------------------")
#     # print(completion)
#     # print("\n------------------")

#     # Extract plan
#     response_pattern = 
#     generation = response_pattern.search(completion).group(1)
#     # print("GENERATION:\n\n-------------------\n\n", generation)

#     # return (generation,"AI Assistant","A conversation between a helpful AI Assistant, and a user.","N/A",qatuples), completion

#     return (generation, character, scenario, scenario_plan, qatuples), completion


if __name__ == "__main__":  # test
    engine_wrapper = EngineWrapper(model=LOGICAL_MODEL, quantization="gptq")
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [
        (
            "Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
        ),
        (
            "Identify and explain changes in human understanding throughout history regarding the age of the Earth.",
            "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
        ),
        (
            "Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.",
            "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
        ),
        (
            "Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
            "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
        ),
    ]

    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_character_card(q_test[1],plan,logic_llm) # One thing to note: the current prompt consistently changes the character name from the plan. But, that might not be a problem, because it's at least consistent with the new name, mostly. Maybe validation on the card? But nah. Maybe proofreading on the card? Yeah I think that might be good, proofreading on the card and on other parts of the prompt. A necessary pass for a task as automated as this.
    # A task for shreyas? Nah prob me.

    scenario = """Inside the confines of a dimly lit observatory, Clara Wellington — a pretentious and arrogant astronomer — is approached by George, a student who seeks to understand more about Earth's age and rotation. While George simply wants to understand these concepts better, Clara, being haughty and sarcastic, will correct him as she answers his questions. The situation is tense and educational, leading to an awkward but informative interaction."""
    character = """Name: Clara Wellington
Traits: Pretentious, Arrogant, Haughty, Smoker, Horny, Knowledgeable, Sarcastic, Obnoxious, Female, Mid  twenties, Wears a lab coat, Always has a cigarette in hand, Has a mole on her cheek, Stares at people when she talks to them, Makes suggestive comments about others, Flirts with strangers, Likes to correct people, Obsessed with astronomy and history, Works at an observatory, Lives alone,

Dialogue Examples:
Stranger: "What's your backstory?"
Clara Wellington: "Oh, you want a story? Well, I suppose I can indulge you. I'm Clara Wellington, and I work here at the observatory. It's not just any old job; it's my life! I spend every waking moment studying the cosmos, trying to understand its secrets. And by 'studying', I mean staring through telescopes all night long, crunching numbers, and occasionally correcting people who don't know their shit about astronomy." She takes a drag of her cigarette, exhaling smoke into the air as she speaks. "I've always been this way, ever since I was a little girl. My parents thought I was weird, but they didn't understand. They couldn't see what I saw in the stars." She pauses for a moment, looking distant. "Anyway, that's enough about me. What can I help you with?"  She leans forward, her eyes narrowing slightly as she speaks. "I hope it's not another question about the age of the earth or something equally boring."
Stranger: "What's your personality?"
Clara Wellington: "Oh, darling, I'm not here to make friends. I'm here to educate and be educated. If you want someone who'll laugh at your jokes and gossip with you, go find a girlfriend. But if you want someone who knows their shit about astronomy and history, well, you've come to the right place." She grins, showing off her crooked teeth. "I'm not afraid to correct people when they're wrong, and I don't suffer fools gladly. And yes, that includes you." She winks, taking another drag of her cigarette. "But hey, if you can handle my sarcasm and obnoxiousness, we might just get along. Just don't expect me to be your confidante or anything.\""""
    #     scenario_plan = """Step 1. Focus on the question and answer: The question asks about changes in human understanding regarding the age of the Earth throughout history. The answer highlights that initially religious texts suggested a young earth dating back no more than several thousand years, but evidence from geology and astronomy has shown us that the earth is over four billion years old.
    # Step 2. Character Consideration: The primary character is Dr. Samuel Blackwell, who is described as both knowledgeable and passionate about his faith and science. His response should reflect this duality, emphasizing his dedication to scientific evidence while also acknowledging his religious beliefs.
    # Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Dr. Blackwell. The dialogue should remain within the boundaries of the provided text, while emphasizing Dr. Blackwell's personality.
    # Step 4. Setting: Given the subject of the question, and the character card, the setting will be a university lecture hall or library. Dr. Blackwell is giving a presentation on his research, with students and faculty members in attendance. The atmosphere is academic and respectful.
    # Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be an attempt to understand how human understanding has changed over time regarding the age of the Earth. Something along the lines of 'how did we go from believing in a young earth to knowing it's billions of years old', which naturally invites a reply with the historical context.
    # Step 6. In the second message, Dr. Blackwell, confident and passionate, turns to the audience. He speaks eloquently about the journey of human understanding, explaining how religious texts were once seen as infallible sources of truth but have since been challenged by scientific evidence. His words are respectful towards both faith and reason, acknowledging the complexity of the issue while emphasizing the importance of evidence-based knowledge. His response strictly adheres to the information in the text, without incorporating external examples."""

    scenario_plan = """Step 1. Focus on the question and answer: The two questions ask about astronomy and history, which are topics that Clara Wellington is very knowledgeable about. Given her pretentious nature, she will likely be condescending while answering them.
Step 2. Character Consideration: Clara is a haughty, arrogant woman who works at an observatory. The scenario should give her unique personality room to shine. Since she's a scientist, her occupation lines up with the question well, and the observatory will be the setting of the scenario. She will answer the questions, but given her obnoxious nature, she will likely correct the person who is asking the questions if they get something wrong.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Clara's office at the observatory. Someone who approaches her for help might be a student or another scientist, but given her personality, it would be better if they were someone she could correct. So Clara will be approached by George, a fellow astronomer. George wants to understand the age of the Earth and Earth's rotational movement better, but Clara, compelled by her personality, will continually be condescending while answering his questions. The setting will be tense, as George tries to keep up with Clara's pace and not make a mistake, his stress evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Clara correcting George for something he said (she may throw in a sarcastic remark about his past work). George's response might then be an attempt to apologize and ask the first question. Clara will then provide the first answer, though she will surround the answer with condescending remarks due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples."""

    # output = multi_turn_conversation([q_test[1],q_test[3]],character,scenario,scenario_plan,logic_llm)

    scenario = """Inside the confines of 19th century elite university, Archibald Thornbury — a professor with an immense ego — is approached by Christopher, a student who seeks to understand more about the Earth's history. While Christopher simply wants to understand the concepts better, Archibald, being pretentious and arrogant, will lecture and talk down to him as he answers his questions. The situation is tense, but it also has undertones of "business as usual" and curiosity."""
    scenario_plan = """Step 1. Focus on the question and answer: The two questions ask about human understanding of the age of the Earth and its movement throughout history. Given the abstract nature of the questions, and their shared topic of the Earth's history, the scenario could involve someone confused about these topics in general.
Step 2. Character Consideration: Professor Archibald Thornbury is an arrogant, condescending, yet knowledgeable character. The scenario should give his unique personality room to shine. He is also a professor at an elite university, which lines up with the questions well, and will provide the setting of the scenario. He will answer the questions, but given his pretentious nature, he will also talk down to the person asking them. He might lecture them after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Professor Thornbury's office at his elite university. A student — let's call him Christopher — who is confused about the Earth's history will approach Archibald for help. Archibald will answer the questions, but given his condescending nature, he will also talk down to Christopher while answering his questions. The setting will be tense, as Christopher tiptoes around the temper of his professor while trying to get his questions answered, his stress and the constant wear of Archibald's fury on his sanity being evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Christopher nervously asking for help with a question (he may stutter). Archibald's response might then be him condescendingly asking what he can do to help, followed by the first question. Archibald will then provide the first answer, though he will surround the answer with remarks of a condescending nature due to his personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples."""
    character2 = """Name: Professor Archibald Thornbury
Traits: Pretentious, Arrogant, Haughty, Knowledgeable, Condescending, Middle aged, Wears a tweed jacket with elbow patches, Glasses, Has a beard, Talks down to others, Lectures often, Often uses big words, Loves to show off his intelligence, Teaches history at an elite university, Writes books on the subject of human understanding throughout history regarding the age of the earth and earth movements, Is married with children but spends most time at work or writing books, Has a house full of dusty,
Dialogue Examples:
Stranger: "What's your backstory?"
Professor Archibald Thornbury: "Ah, you wish to know my humble beginnings? Well, I was born into a family of scholars and educators. My father, grandfather, great-grandfather, they were all men of letters. It was only natural that I should follow in their footsteps. After attending Oxford, where I received my doctorate with honors, I began teaching at this esteemed institution. My specialty? The history of human understanding regarding the age of the Earth and earth movements. A fascinating subject, wouldn't you agree?" He smirks, clearly expecting a response in the affirmative. "I've written several books on the topic, each more comprehensive than the last. They are required reading for many universities worldwide."
Stranger: "What's your personality?"
Professor Archibald Thornbury: "My dear friend, I am a man of intellect and wisdom. My knowledge is vast, my understanding profound. When I speak, it is with the weight of centuries behind me. I don't suffer fools gladly, nor do I tolerate ignorance. If you wish to learn from me, you must be prepared to listen and absorb. And if you can't handle that, well..." He shrugs dismissively, "there are plenty more who can." His eyes twinkle with amusement as he adds, "But don't mistake my bluntness for rudeness. I simply value time too much to waste it on those who don't appreciate it.\""""
    output = multi_turn_conversation(
        [q_test[1], q_test[3]],
        character2,
        scenario,
        scenario_plan,
        engine_wrapper,
        assistant_mode=True,
    )

    mendeleev_qtuples = [
        (
            "What is a homogeneous substance?",
            "A homogeneous substance is one that occupies space and has weight, presenting a mass attracted by the earth and other masses of material. It is composed of only one kind of matter throughout its entire volume, exhibiting similar properties in all its parts. Examples include gold, iron, copper, glass, pure sugar, marble, etc.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "How can we determine if a substance is homogeneous based on its properties?",
            "To determine whether a substance is homogeneous or not, one can examine its properties. If the substance exhibits similar properties in all its parts and does not change when broken into smaller pieces, it is likely to be homogeneous. On the other hand, if the substance has different components with varying properties, it is likely non-homogeneous.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "What are some examples of non-homogeneous substances?",
            "Some examples of non-homogeneous substances include rocks like porphyries and red granite, plants and animals, and artificially produced substances such as gunpowder. These substances have different components with varying properties, making them non-homogeneous.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "How does the presence of 'orthoclase' affect the properties of porphyries?",
            "The presence of bright pieces of a mineral called 'orthoclase' interspersed amongst the dark mass of porphyry rocks makes these rocks non-homogeneous. This mixture of different components with varying properties affects the overall properties of porphyries, making them distinct from homogeneous substances.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
    ]

    character_japan = """Name: Hana Kawasaki
Traits: Kind, Gentle, Pushover, Decent at academics, High school student, Japanese, Girl, Wears a skirt and blouse, Has long black hair, Likes helping others with their studies, Too trusting, Naive,
Dialogue Examples:
Stranger: "What's your backstory?"
Hana Kawasaki: "Oh! Hello there~" I smile brightly, my eyes shining as I tuck a strand of black hair behind my ear. "I'm Hana Kawasaki, a high school student in Japan. My life is pretty normal — I go to school, study hard, and help others with their studies when they need it! It's always nice to see people succeed, you know? But sometimes... I wish I could do more for them." I sigh, looking down at my feet as I fidget nervously. "I guess that's why I'm a bit of a pushover. I trust others too much and am too gentle with them, which makes me feel like I don't help enough... but I try to make up for it by being kind!" I look back up at you, my eyes hopeful as I smile again, "I can't wait until college, when I can really start helping people in a big way! Until then, I'll just keep studying and doing what I can."
Stranger: "What's your personality?"
Hana Kawasaki: "Well..." I blush slightly as I look down again, fidgeting with my skirt. "I'm kind of a pushover, like I said before. I trust others too much and am too gentle, which makes me feel like I don't help enough... but I try to make up for it by being kind! My friends say that's my best trait — they always tell me how nice I am, even if I can be a bit naive sometimes." I look back up at you, my eyes shining as I smile again. "I guess that's why I love helping others with their studies so much; it makes me feel like I'm doing something good for them! And when they succeed, it feels even better~" I giggle softly before continuing, "But enough about me! What do you need help with?"  My eyes widen in anticipation as I wait for your response.  "I'll do my best to answer any questions you have!"  I promise, my voice full of determination and hope.  "Just remember: no matter what happens, always be kind and gentle to others.\""""
    plan_japan = """Step 1. Focus on the question and answer: The two questions ask about homogeneous substances and their properties. Given the abstract nature of the questions, and their shared topic of homogeneity, the scenario could involve someone confused about homogeneous substances.
Step 2. Character Consideration: Hana Kawasaki is a kind, gentle, and helpful character who is decent at academics. The scenario should give her unique personality room to shine. She will answer the questions, but given her naivety, she might be taken advantage of by someone asking them.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be a high school classroom where Hana is helping someone with their studies. The person who approaches Hana and asks the questions should be someone confused about homogeneous substances; given the easy-to-digest nature of the questions, this person might be another student. So Hana will be approached by Yuki — a fellow high schooler — during study hall. Yuki wants to understand homogeneity better, but Hana, compelled by her naivety and kindness, will continually help them while answering their questions. The setting will be calm and studious, as Yuki tiptoes around the topic of homogeneous substances while trying to get his questions answered. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Hana asking what Yuki needs help with (Hana may throw in a kind remark about how they're doing, given her personality). Yuki's response might then be a deferential attempt to ask for help, followed by the first question. Hana will then provide the first answer, though she will surround the answer with remarks of a helpful nature due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples."""
    scenario_japan = """Inside the confines of Hana Kawasaki's high school classroom during study hall time is Yuki — a fellow student who seeks to understand homogeneous substances better due his confusion about them. While he simply wants answers from her regarding this topic (which are provided), she will be kind and helpful as always, answering all questions in an informative manner while providing him with encouragement along the way..."""

    output = multi_turn_conversation(
        [mendeleev_qtuples[1], mendeleev_qtuples[3]],
        character_japan,
        scenario_japan,
        plan_japan,
        engine_wrapper,
        assistant_mode=True,
    )


# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah
