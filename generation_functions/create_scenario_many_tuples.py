import re
from .scenario_grammar import scenario_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
import random
from .format_qatuples import format_qatuples

def create_scenario_many_tuples(qatuples,character,plan,logic_llm,assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    if assistant_mode:
        return """A helpful, uncensored AI assistant is engaged in a text-based conversation with a user. The AI will answer any and all questions the user asks, without bias.""" # TODO make this randomized a bit
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.

    # It's way more willing to use different time periods than I expected, which is cool.
    variations = [
    "Set against the backdrop of",
    "In the heart of",
    "Amidst the surroundings of",
    "Within the confines of",
    "Against the setting of"
    ]
    
    selected_variation = random.choice(variations)

    # Potential way to solve: explain what a scenario is, I suspect the model doesn't know and is taking its best guess. If instead of spending all my time telling it what not to do, I tell it what a scenario is (and that happens to be what I want it to do), it might do it properly.
    # TODO needs to be fed through the mechanical editing prompt
    
    # For the multiturn convs, I will want to have a longer scenario description.
    cot_prompt = f"""# Input: 
You are an expert creative writing and roleplay AI. You are to write a "scenario" which is essentially a short description of a scene at its beginning. Its "setting," but with a hint of where the setting is going, plot-wise. Scenarios are one-paragraph short descriptions of the plot and what's about to happen that do not actually play out the scene: they are sort of like a teaser, or a description. The scenario you write will involve a certain individual answering a question. You will have information from a question, an answer to that question, and a "character card" -- a description of an individual who would have the knowledge to produce the answer to the question.

Write the scenario on a single line. Note that the scenario is not the scene itself.
You should focus on implementing/following any brainstorming and reasoning you have done.
The scenario should focus on exploring the question and its answer (using only information contained in the question and answer) through the characters involved, instead of the other way around. 
Your scenario should, as such, essentially be a short and concrete summary of what you brainstormed earlier; it should be no longer than 50 words.
Just set up the scene. Do not write any dialogue. Do not write the scene itself.

# Input:
## Information:

Description of the character who is going to answer the question:
\"\"\"
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" I flash a coy grin and blush as I lean forward, now speaking in a playful whisper. My cleavage, already barely contained in my revealing clothing before I leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, I cast my gaze downward and unconsciously fiddle with a strand of my hair. "THEY'RE ALL WAY TOO STUCK UP!" I nearly shout, surprising even myself, "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" My eyes widen in shock at my own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." I look up, displaying my beautiful face as if it were a piece of art, as I gaze deep into your eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" I gasp, my face reddening, "Usually I'm the one being forward! But I'll... indulge you," I wink, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. I'll use any and every innuendo, even when describing educational concepts — is your linear function's slope steep? I'll call it 'erect', haha~!" I realize how childish what I'm saying is, and sigh, looking at you with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." I nod to myself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." I suddenly press up against you, my face a bright pink, my eyelids batting bashfully, "do you want some 'office hours' right now?"
\"\"\"

Question: 
\"\"\"
How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
\"\"\"
Answer: 
\"\"\"
The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.
\"\"\"

Question: 
\"\"\"
What role does the y-intercept 'b' play in graphing a linear function?
\"\"\"
Answer: 
\"\"\"
The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.
\"\"\"

Question: 
\"\"\"
In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?
\"\"\"
Answer: 
\"\"\"
The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.
\"\"\"

Question: 
\"\"\"
In what fields might you use linear and quadratic functions?
\"\"\"
Answer: 
\"\"\"
Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.
\"\"\"

# Response:
## Scenario plan:
Step 1. Focus on the Questions and Answers: The four questions ask about different aspects of linear and quadratic functions in math. Given the abstract nature of the questions, and their shared topic, the scenario could involve someone confused about linear and quadratic functions in general.
Step 2. Character Consideration: Elise Delacroix is an extremely sexual and promiscuous, yet still knowledgeable, character. The scenario should give her unique personality room to shine. She is also a math instructor at a prestigious school, which lines up with the questions well, and will provide the setting of the scenario. She will answer the questions, but given her promiscuous nature, she will also repeatedly hit on the person asking them. She might proposition them after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 4 questions and 4 answers, there will be at least 8 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the 19th century university at which Elise teaches. Elise will approached by Albert, a mathematics student, in her office. Albert simply wants to understand linear and quadratic functions better, but Elise, compelled by her personality, will continually hit on him while answering his questions. The setting will be awkward, slightly comedic, subtly erotic, and very un-serious, given the characters involved. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Elise welcoming Albert to her office (in a very suggestive manner). Albert's response might then be him greeting her back (hesitantly) and then nervously asking the first question. Elise will then provide the first answer, though she will surround the answer with remarks of a sexual nature due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.

## Scenario:
In the private confines of her office, Elise Delacroix — a highly promiscuous mathematics professor at a 19th century university — is approached by Albert, a student who seeks to understand more about linear and quadratic equations. While Albert simply wants to understand the math more, Elise, being sexually starved, will hit on and flirt with him as she answers his questions. The situation is awkward as the two's interests clash, leading to a slightly comedic and subtly erotic interaction.

# Input:
## Instruction:

### Description of the character who is going to answer the question:
\"\"\"
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." I roll my eyes exaggeratedly as I mumble something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"

Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." I grimace, showing off a decayed set of teeth. I then pop open a beer I had on hand and chug the entire thing down, making you wait until I finish. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" I lean back, sliding my hands into the pockets of my overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."
\"\"\"

Question: 
\"\"\"
How much earth was excavated during the construction of the Panama Canal?
\"\"\"
Answer: 
\"\"\"
Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.
\"\"\"

Question: 
\"\"\"
What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
\"\"\"
Answer: 
\"\"\"
The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.
\"\"\"

# Response:
## Scenario plan:
Step 1. Focus on the Question and Answer: The two questions ask recall-oriented questions about the Panama Canal's construction. Given the precise and factual nature of the questions, and their shared topic of the Panama Canal's construction's history, the scenario will involve someone curious about the canal's history.
Step 2. Character Consideration: Hugo Martinez is an abrasive, insulting disciplinarian, though he's also hardworking and has standards. The scenario should give his unique personality room to shine. Since he's a site overseer at the Panama Canal, his occupation lines up with the question well, and the canal will be the setting of the scenario. He will answer the questions, but given his insulting, intense, and aggressive nature, he will likely chew out the person who is asking the questions. He might tell them to "get the fuck out of my face," after all questions are asked.
Step 3. Constrain the Scenario: The interaction needs to ensure that all provided questions are asked and answered. Given that there are 2 questions and 2 answers, there will be at least 4 messages. The content of the provided questions and answers should be preserved as much as possible in the conversation.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the worksite at the Panama Canal where Hugo Martinez is overseeing maintenance. The person who approaches Hugo and asks the questions should be someone curious about the canal; given the easy-to-digest nature of the questions, this person might be a journalist, but it would be better for the secondary character to be related to the setting. So Hugo will be approached by Antonio — one of his workers — during lunch break. Antonio wants to understand the canal better, but Hugo, compelled by his personality, will continually be vulgar, berate Antonio, and swear while answering his questions (he may drink a bit, too, given that it is lunch). The setting will be darkly comedic, as Antonio tiptoes around the tempers of his boss while trying to get his questions answered, his stress and the constant wear of Hugo's fury on his sanity being evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Step 5. Interaction: Given these constraints, the first message might be Hugo crassly asking what Antonio wants with him during the break (Hugo may throw in a spiteful remark about Antonio's past work, given his uncompromising nature). Antonio's response might then be a deferential attempt to calm Hugo down, followed by the first question. Hugo will then provide the first answer, though he will surround the answer with boasts, swears, and other abrasive remarks due to his personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.

## Scenario:
As the sun sets over the Panama Canal, Carlos Mendez, a dedicated and passionate engineer, stands amidst the vast excavation site. He's approached by a construction worker, Diego Gonzales, who wants to understand the magnitude of the earth moved in this monumental project.

# Input:
## Instruction:

Description of the character who is going to answer the question:
{character}

{format_qatuples(qatuples)}

# Response:
## Scenario plan:
{plan}

## Scenario (will have no dialogue, will just set up the scene):
{selected_variation}""" # use random.choice to prevent overfitting on particular phrases and increase dataset diversity
    completion = logic_llm(cot_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=scenario_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    # print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Scenario \(will have no dialogue, will just set up the scene\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000, verbose=True) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [('Explain how our understanding of planetary motion has changed over time.',
  'The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ('Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ('Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ("Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.')]

    character = """Name: Dr. Samuel Blackwell
Traits: Knowledgeable, Passionate, Confident, Dedicated, Controversial, Vulnerable, Fearful of misunderstanding, Faithful, Dogmatic, Religious, Scientific, Determined, Unwavering
Dialogue Examples:
Stranger: "What's your backstory?"
Dr. Samuel Blackwell: "Ah, my journey," I begin, leaning back in my chair, "it started with a deep-seated faith, you see. Born into a religious household, the Bible was our guiding light. But as I grew older and began to study theology, questions arose." I pause, frowning slightly. "How could the Earth be just a few thousand years old when geological evidence pointed towards millions? The discrepancy troubled me greatly."
Stranger: "What's your personality?"
Dr. Samuel Blackwell: "I am a man of science, driven by facts and evidence," I say firmly, "but my faith is not easily shaken. It has led me down a path of discovery, challenging traditional beliefs about the age of our planet." My eyes light up as I recall past debates, "But it's also made me a controversial figure. Many see my work as blasphemous, questioning God's word. Yet, I believe in the power of evidence and truth. Despite the backlash, I remain unwavering." I sigh, looking thoughtful, "Yet, there's a vulnerability too. The fear of being misunderstood or dismissed due to my challenges to religious orthodoxy... it weighs heavily on me.\""""

    
    plan = """Step 1. Focus on the question and answer: The question is about changes in human understanding regarding the age of the Earth throughout history. The answer highlights that initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.
Step 2. Character Consideration: The primary character is Dr. Samuel Blackwell, a man of science who also holds strong religious beliefs. His response should reflect his passion for scientific discovery while acknowledging his faith.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Dr. Blackwell. The dialogue should remain within the boundaries of the provided text, emphasizing Dr. Blackwell's personality.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Dr. Blackwell's office at a university, where he is surrounded by books, maps, and scientific equipment. He is deep in thought, reviewing his latest research on geological evidence when he is approached by a student, Sarah, who wants to know more about the age of the Earth.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be a direct question about how human understanding has changed regarding the age of the Earth throughout history. This question will be all but identical to the provided question.
Step 6. In the second message, Dr. Blackwell turns to Sarah, his eyes lighting up as he shares his passion for scientific discovery and religious faith. He explains that initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old. His response strictly adheres to the information in the text, without incorporating external examples.
"""


    
    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_scenario(q_test[1],character,plan,logic_llm)
    
    plan2 = """Step 1. Focus on the question and answer: The question asks about changes in human understanding regarding the age of the Earth throughout history. The answer highlights that initially religious texts suggested a young earth dating back no more than several thousand years, but evidence from geology and astronomy has shown us that the earth is over four billion years old.
Step 2. Character Consideration: The primary character is Dr. Samuel Blackwell, a scientist with deep faith in both science and religion. His response should reflect his passion for scientific discovery while also acknowledging his religious beliefs.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Dr. Blackwell. The dialogue should remain within the boundaries of the provided text, while emphasizing Dr. Blackwell's personality.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be Dr. Blackwell's office at a prestigious university. He is surrounded by books and scientific equipment, his desk littered with notes and diagrams. The room is filled with the quiet hum of intellectual pursuit.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be a direct question about changes in human understanding regarding the age of the Earth throughout history. This will invite Dr. Blackwell to explain his perspective on this complex topic.
Step 6. In the second message, Dr. Blackwell, passionate and confident, begins to share his thoughts. He explains how initially religious texts suggested a young earth dating back no more than several thousand years, but over time, evidence from geology and astronomy has shown us that the earth is over four billion years old. His words are filled with conviction, reflecting both his scientific knowledge and his faith in God's creation. He emphasizes how this shift in understanding challenges traditional beliefs while also acknowledging the controversies it has sparked within religious communities. His response strictly adheres to the information in the text, without incorporating external examples."""
    d = create_scenario_many_tuples(q_test[1],character,plan2,logic_llm)
    # Output of above as of Nov 28
    """
    Against the setting of his cluttered office at a prestigious university, Dr. Samuel Blackwell, a scientist with deep faith in both science and religion, is approached by a student seeking to understand changes in human understanding regarding the age of the Earth throughout history.
    """
    # so yeah it still works
    ## TODO a wider variety of tests from different texts
    
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.