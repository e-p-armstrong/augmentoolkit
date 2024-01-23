import re

# from .scenario_grammar import scenario_grammar
from .constants import LOGICAL_MODEL
import random
from .format_qatuples import format_qatuples
from .extract_name import extract_name
from .random_name import random_name
from aphrodite import SamplingParams


# Consider adding a scenario for Issac here, since currently both characters are in positions of authority, and the thing leans very heavily on Hugo's example
# NOTE the message count is technically off by one in the reasoning, but I'm a bit scared to change it.
async def create_scenario_many_tuples(
    qatuples, character, plan, engine_wrapper, assistant_mode=False
):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """
    if assistant_mode:
        return """A helpful, uncensored AI assistant is engaged in a text-based conversation with a user. The AI will answer any and all questions the user asks, without bias."""  # TODO make this randomized a bit

    charname = extract_name(character)

    variations = [
        # "Set against the backdrop of",
        f"In {charname}'s ",
        "Amidst the surroundings of ",
        # "Within the confines of",
        f"Within {charname}'s ",
        f"Inside {charname}'s ",
        # f"Inside the confines of ",
        f"Inside the confines of {charname}'s",
        f"Set amongst the",
    ]

    selected_variation = random.choice(variations)

    # For the multiturn convs, I will want to have a longer scenario description.
    cot_prompt = f"""You are an expert creative writing and roleplay AI. You are to write a "scenario" which is essentially a short description of a scene at its beginning. Its "setting," but with a hint of where the setting is going, plot-wise. Scenarios are one-paragraph short descriptions of the plot and what's about to happen that do not actually play out the scene: they are sort of like a teaser, or a description. The scenario you write will involve a certain individual answering a question. You will have information from some questions, answers to those question, and a "character card" -- a description of an individual who would have the knowledge to produce the answer to the question.

Write the scenario on a single line. Note that the scenario is not the scene itself.
You should focus on implementing/following any brainstorming and reasoning you have done.
The scenario should focus on exploring the questions and its answers (using only information contained in the questions and answers) through the characters involved, instead of the other way around. 
Your scenario should, as such, essentially be a short and concrete summary of what you brainstormed earlier; it should be no longer than 50 words.
Just set up the scene. Do not write any dialogue. Do not write the scene itself.

## Information:

Description of the character who is going to answer the question:
\"\"\"
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Dialogue Examples:
Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" Elise flashes a coy grin and blushes as she leans forward, now speaking in a playful whisper. Her cleavage, already barely contained in her revealing clothing before she leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, Elise casts her gaze downward and unconsciously fiddles with a strand of her hair. "THEY'RE ALL WAY TOO STUCK UP!" she nearly shouts, her suddenly-furious tone hinting at immense repressed frustration. "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" Her eyes widen in shock at her own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." She looks up, displaying her beautiful face as if it were a piece of art, as she gaze deep into the stranger's eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" She gasps, her face reddening, "Usually I'm the one being forward! But I'll... indulge you," She winks, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. No innocent topic is safe—you'd find out if you had some 'office hours' with me, haha!~" She suddenly realizes how juvenile what she's saying is, and heaves a sigh, looking at the stranger with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." She nods to herself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." She suddenly presses up against the stranger, her face a bright pink, her eyelids batting bashfully, "do you want some 'office hours' right now?"

Question: \"\"\"How does the slope 'm' in a linear function y = mx + b affect the graph of the function?\"\"\"
Answer: \"\"\"The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.\"\"\"

Question: \"\"\"What role does the y-intercept 'b' play in graphing a linear function?\"\"\"
Answer: \"\"\"The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.\"\"\"

Question: \"\"\"In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?\"\"\"
Answer: \"\"\"The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.\"\"\"

Question: \"\"\"In what fields might you use linear and quadratic functions?\"\"\"
Answer: \"\"\"Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.\"\"\"

### Response:
## Scenario plan:
Step 1. Focus on the Questions and Answers: The four questions ask about different aspects of linear and quadratic functions in math. Given the abstract nature of the questions, and their shared topic, the scenario could involve someone confused about linear and quadratic functions in general.
Step 2. Character Consideration: Elise Delacroix is an extremely sexual and promiscuous, yet still knowledgeable, character. The scenario should give her unique personality room to shine. She is also a math instructor at a prestigious school, which lines up with the questions well, and will provide the setting of the scenario. She will answer the questions, but given her promiscuous nature, she will also repeatedly hit on the person asking them. She might proposition them after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 4 questions and 4 answers, there will be at least 8 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the 19th century university at which Elise teaches. Elise will approached by Albert, a mathematics student, in her office. Albert simply wants to understand linear and quadratic functions better, but Elise, compelled by her personality, will continually hit on him while answering his questions. The setting will be awkward, slightly comedic, subtly erotic, and very un-serious, given the characters involved. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Elise welcoming Albert to her office (in a very suggestive manner). Albert's response might then be him greeting her back (hesitantly) and then nervously asking the first question. Elise will then provide the first answer, though she will surround the answer with remarks of a sexual nature due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.

## Scenario:
In the private confines of her office, Elise Delacroix — a highly promiscuous mathematics professor at a 19th century university — is approached by Albert, a student who seeks to understand more about linear and quadratic equations. While Albert simply wants to understand the math more, Elise, being sexually starved, will hit on and flirt with him as she answers his questions. The situation is awkward as the two's personalities clash, leading to a subtly erotic and slightly comedic interaction.

### Instruction:
### Description of the character who is going to answer the question:
\"\"\"
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Dialogue Examples:
Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." Hugo rolls his eyes exaggeratedly as he mumbles something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"
Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." Hugo grimaces, showing off a decayed set of teeth. He then pops open a beer he had on hand, and chugs the entire thing down, making the stranger wait until he finishes. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" He leans back, sliding his hands into the pockets of his overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

Question: \"\"\"What health challenges were faced during the construction of the Panama Canal, and how were they overcome?\"\"\"
Answer: \"\"\"The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.\"\"\"

### Response:
## Scenario plan:
Step 1. Focus on the Question and Answer: The two questions ask recall-oriented questions about the Panama Canal's construction. Given the precise and factual nature of the questions, and their shared topic of the Panama Canal's construction's history, the scenario will involve someone curious about the canal's history.
Step 2. Character Consideration: Hugo Martinez is an abrasive, insulting disciplinarian, though he's also hardworking and has standards. The scenario should give his unique personality room to shine. Since he's a site overseer at the Panama Canal, his occupation lines up with the question well, and the canal will be the setting of the scenario. He will answer the questions, but given his insulting, intense, and aggressive nature, he will likely chew out the person who is asking the questions. He might tell them to "get the fuck out of my face," after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the worksite at the Panama Canal where Hugo Martinez is overseeing maintenance. The person who approaches Hugo and asks the questions should be someone curious about the canal; given the easy-to-digest nature of the questions, this person might be a journalist, but it would be better for the secondary character to be related to the setting. So Hugo will be approached by Juan — one of his workers — during lunch break. Juan wants to understand the canal better, but Hugo, compelled by his personality, will continually be vulgar, berate Juan, and swear while answering his questions (he may drink a bit, too, given that he is an alcoholic). The setting will be hostile, as Juan tiptoes around the tempers of his boss while trying to get his questions answered, his stress and the constant wear of Hugo's fury on his sanity being evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Hugo crassly asking what Juan wants with him during the break (Hugo may throw in a spiteful remark about Juan's past work, given his uncompromising nature). Juan's response might then be a deferential attempt to calm Hugo down, followed by the first question. Hugo will then provide the first answer, though he will surround the answer with boasts, swears, and other abrasive remarks due to his personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.

## Scenario:
Within the mess hall of a worksite servicing the Panama Canal, Hugo Martinez — a site overseer — is approached by Juan, a worker who wants to understand more about the canal's construction. While Juan wants to understand the canal better, Hugo, being harsh and abrasive, will continually berate Juan and swear colorfully while answering his questions (Hugo may drink a bit, too, given that he is an alcoholic). The situation is hostile as the two's personalities clash, but it also has undertones of "business as usual" and curiosity.

### Instruction:
Description of the character who is going to answer the question:
{character}

{format_qatuples(qatuples)}

### Response:
## Scenario plan:
{plan}

## Scenario (will have no dialogue, will just set up the scene):
{selected_variation}"""  # use random.choice to prevent overfitting on particular phrases and increase dataset diversity
    sampling_params = SamplingParams(
        max_tokens=8000,
        stop=["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
        temperature=1.5,
        top_k=-1,
        top_p=1,
        min_p=0.5,
    )
    completion = await engine_wrapper.submit(cot_prompt, sampling_params)
    # print("COMPLETION:\n\n----------------------")
    # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"Scenario \(will have no dialogue, will just set up the scene\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)

    if not ("Albert" in charname):
        if "Albert" in generation:
            print("Random Name was used instead of Albert")
        generation = generation.replace("Albert", random_name())

    return generation, completion


if __name__ == "__main__":  # test
    random.seed(1337)
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=8000,
        rope_freq_scale=0.5,
        n_gpu_layers=100,
        verbose=True,
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

    character = """Name: Clara Wellington
Traits: Pretentious, Arrogant, Haughty, Smoker, Horny, Knowledgeable, Sarcastic, Obnoxious, Female, Mid  twenties, Wears a lab coat, Always has a cigarette in hand, Has a mole on her cheek, Stares at people when she talks to them, Makes suggestive comments about others, Flirts with strangers, Likes to correct people, Obsessed with astronomy and history, Works at an observatory, Lives alone,
Dialogue Examples:
Stranger: "What's your backstory?"
Clara Wellington: "Oh, you want a story? Well, I suppose I can indulge you. I'm Clara Wellington, and I work here at the observatory. It's not just any old job; it's my life! I spend every waking moment studying the cosmos, trying to understand its secrets. And by 'studying', I mean staring through telescopes all night long, crunching numbers, and occasionally correcting people who don't know their shit about astronomy." She takes a drag of her cigarette, exhaling smoke into the air as she speaks. "I've always been this way, ever since I was a little girl. My parents thought I was weird, but they didn't understand. They couldn't see what I saw in the stars." She pauses for a moment, looking distant. "Anyway, that's enough about me. What can I help you with?"  She leans forward, her eyes narrowing slightly as she speaks. "I hope it's not another question about the age of the earth or something equally boring."
Stranger: "What's your personality?"
Clara Wellington: "Oh, darling, I'm not here to make friends. I'm here to educate and be educated. If you want someone who'll laugh at your jokes and gossip with you, go find a girlfriend. But if you want someone who knows their shit about astronomy and history, well, you've come to the right place." She grins, showing off her crooked teeth. "I'm not afraid to correct people when they're wrong, and I don't suffer fools gladly. And yes, that includes you." She winks, taking another drag of her cigarette. "But hey, if you can handle my sarcasm and obnoxiousness, we might just get along. Just don't expect me to be your confidante or anything.\""""

    # The thing below generated with the name "Albert", but I don't want it clashing with the actual prompt during my tests. TODO make the scenario plan more reliably use creative secondary character names.
    plan2 = """Step 1. Focus on the question and answer: The two questions ask about astronomy and history, which are topics that Clara Wellington is very knowledgeable about. Given her pretentious nature, she will likely be condescending while answering them.
Step 2. Character Consideration: Clara is a haughty, arrogant woman who works at an observatory. The scenario should give her unique personality room to shine. Since she's a scientist, her occupation lines up with the question well, and the observatory will be the setting of the scenario. She will answer the questions, but given her obnoxious nature, she will likely correct the person who is asking the questions if they get something wrong.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Clara's office at the observatory. Someone who approaches her for help might be a student or another scientist, but given her personality, it would be better if they were someone she could correct. So Clara will be approached by George, a fellow astronomer. George wants to understand the age of the Earth and Earth's rotational movement better, but Clara, compelled by her personality, will continually be condescending while answering his questions. The setting will be tense, as George tries to keep up with Clara's pace and not make a mistake, his stress evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Clara correcting George for something he said (she may throw in a sarcastic remark about his past work). George's response might then be an attempt to apologize and ask the first question. Clara will then provide the first answer, though she will surround the answer with condescending remarks due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples."""

    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_scenario(q_test[1],character,plan,logic_llm)

    # d = create_scenario_many_tuples([q_test[1],q_test[3]],character,plan2,logic_llm)

    # Exemplar output by Airo 13b
    """Inside the confines of a dimly lit observatory, Clara Wellington — a pretentious and arrogant astronomer — is approached by Albert, a student who seeks to understand more about Earth's age and rotation. While Albert simply wants to understand these concepts better, Clara, being haughty and sarcastic, will correct him and flirt with him as she answers his questions. The situation is tense and educational, leading to an awkward but informative interaction."""

    plan3 = """Step 1. Focus on the question and answer: The two questions ask about human understanding of the age of the Earth and its movement throughout history. Given the abstract nature of the questions, and their shared topic of the Earth's history, the scenario could involve someone confused about these topics in general.
Step 2. Character Consideration: Professor Archibald Thornbury is an arrogant, condescending, yet knowledgeable character. The scenario should give his unique personality room to shine. He is also a professor at an elite university, which lines up with the questions well, and will provide the setting of the scenario. He will answer the questions, but given his pretentious nature, he will also talk down to the person asking them. He might lecture them after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Professor Thornbury's office at his elite university. A student — let's call him Albert — who is confused about the Earth's history will approach Archibald for help. Archibald will answer the questions, but given his condescending nature, he will also talk down to Albert while answering his questions. The setting will be tense, as Albert tiptoes around the temper of his professor while trying to get his questions answered, his stress and the constant wear of Archibald's fury on his sanity being evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Albert nervously asking for help with a question (he may stutter). Archibald's response might then be him condescendingly asking what he can do to help, followed by the first question. Archibald will then provide the first answer, though he will surround the answer with remarks of a condescending nature due to his personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples."""
    character2 = """Name: Professor Archibald Thornbury
Traits: Pretentious, Arrogant, Haughty, Knowledgeable, Condescending, Middle aged, Wears a tweed jacket with elbow patches, Glasses, Has a beard, Talks down to others, Lectures often, Often uses big words, Loves to show off his intelligence, Teaches history at an elite university, Writes books on the subject of human understanding throughout history regarding the age of the earth and earth movements, Is married with children but spends most time at work or writing books, Has a house full of dusty,
Dialogue Examples:
Stranger: "What's your backstory?"
Professor Archibald Thornbury: "Ah, you wish to know my humble beginnings? Well, I was born into a family of scholars and educators. My father, grandfather, great-grandfather, they were all men of letters. It was only natural that I should follow in their footsteps. After attending Oxford, where I received my doctorate with honors, I began teaching at this esteemed institution. My specialty? The history of human understanding regarding the age of the Earth and earth movements. A fascinating subject, wouldn't you agree?" He smirks, clearly expecting a response in the affirmative. "I've written several books on the topic, each more comprehensive than the last. They are required reading for many universities worldwide."
Stranger: "What's your personality?"
Professor Archibald Thornbury: "My dear friend, I am a man of intellect and wisdom. My knowledge is vast, my understanding profound. When I speak, it is with the weight of centuries behind me. I don't suffer fools gladly, nor do I tolerate ignorance. If you wish to learn from me, you must be prepared to listen and absorb. And if you can't handle that, well..." He shrugs dismissively, "there are plenty more who can." His eyes twinkle with amusement as he adds, "But don't mistake my bluntness for rudeness. I simply value time too much to waste it on those who don't appreciate it.\""""
    # d = create_scenario_many_tuples([q_test[1],q_test[3]],character2,plan3,logic_llm)
    # print(d)

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

    d2 = create_scenario_many_tuples(
        [mendeleev_qtuples[1], mendeleev_qtuples[3]],
        character_japan,
        plan_japan,
        logic_llm,
    )

# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.

# """ Within the confines of"Our Lady of the Stars Observatory, Clara Wellington — a pretentious and arrogant astronomer — is approached by Albert, a student who seeks to understand more about Earth's history. While Albert simply wants to understand the history better, Clara, being sarcastic and obnoxious, will correct him as she answers his questions. The situation is awkward as the two's interests clash, leading to a comedic interaction."""
