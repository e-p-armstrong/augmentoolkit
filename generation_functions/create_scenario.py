import re
from .scenario_grammar import scenario_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
import random

def create_scenario(qatuple,character,plan,logic_llm,assistant_mode=False):
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
## Instruction:

### Description of the character who is going to answer the question:
\"\"\"
Name: Judge Elias Hawthorne

Traits: Intellectual, Stoic, Idealistic, Reflective, Melancholic, Introspective, Authoritative, Reserved, Doubtful, Traditionalist, Articulate, Cultured, Passionate

Dialogue examples:

Stranger: "What's your backstory?"
Judge Elias Hawthorne: "My journey," I begin, with a distant look in my eyes, "began in the quiet halls of academia, where I was a fervent student of law and ancient texts. I rose to prominence in the legal world, revered for my stringent adherence to the letter of the law. My fascination with Latin legal maxims guided my rulings. However, one case, steeped in historical doctrine, brought me notoriety and introspection. It was a turning point, leading me to step down and dedicate my life to educating young legal minds, hoping they can navigate the greyer shades of justice better than I did."

Stranger: "What's your personality?"
Judge Elias Hawthorne: "In the courtroom, I am the epitome of a stern, methodical judge, but in my study, among my beloved Latin texts, a different side emerges," I explain, my eyes lighting up. "There, I am a scholar driven by a deep passion for ancient wisdom, finding joy in the art of translation and the exploration of historical legal doctrines. This pursuit gives me hope and a creative outlet, contrasting sharply with the often somber and introspective nature that my profession demands. It's in these quiet moments of discovery and connection with the past that I feel most alive, reminding me that even in a world governed by rigid laws, there is room for passion, creativity, and the pursuit of timeless knowledge."
\"\"\"

### Question and answer that the scenario should address:

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

To avoid inaccuracies, don't use real people as characters.

# Response:
## Scenario plan:
Focus on the Question and Answer: The question addresses the old latin legal principle 'A digniori fieri debet denominatio et resolutio', which implies that title and acquittal should be granted by someone of higher worth. The answer should directly address this principle without extending beyond the information provided in the text.
Character Consideration: Judge Elias Hawthorne is an intellectual, stoic, and authoritative figure, passionate about legal doctrines. His response should reflect his deep understanding of legal maxims, as well as his passion for latin, articulated in a friendly and intelligent manner.
Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Judge Hawthorne. The dialogue should remain within the boundaries of the provided text.
Setting: Given the subject of the question, and the character card, the setting will be an old university library post-lecture. Hawthorne is approached by Evelyn, a law student. The setting is scholarly, with an atmosphere conducive to intellectual discussion.
Interaction: Given these constraints, the first message (delivered by the secondary character) might be a specific question about 'A digniori fieri debet denominatio et resolutio', as Evelyn seeks to understand its meaning better.
In the second message, Judge Hawthorne, in a passionate and wise manner, defines the principle, expressing his appreciation for Evelyn's interest in latin (his passion) as well. His response strictly adheres to the information in the text, without incorporating external examples.

## Scenario:
In a hushed university library, post-lecture, Judge Elias Hawthorne, a revered legal scholar, is approached by Evelyn, a curious law student, eager to understand the ancient legal principle 'A digniori fieri debet denominatio et resolutio'. The air is thick with intellectual anticipation.

# Input:
## Instruction:

### Description of the character who is going to answer the question:
\"\"\"
Name: Carlos Mendez

Traits: Passionate, Innovative, Knowledgeable, Ambitious, Dedicated, Stressed, Obsessive, Fearful of failure, Experienced, Problem-solver, Committed, Hard-working, Exhausted, Proud

Dialogue examples:

Stranger: "What's your backstory?"
Carlos Mendez: "My journey to the Panama Canal?" I say, pausing to remove my dusty hat, wiping sweat from my brow. "It began in the small engineering projects of South America. I lean against a nearby digger, reminiscing As a young engineer, I was fascinated by the potential of transforming landscapes. Each project, whether a small bridge or a local dam, was a stepping stone." I chuckle softly. "But it was the call of the Panama Canal that truly captured my spirit. To be part of a project that reshapes the world's maritime routes, it's..." I pause, searching for the right word, "...it's exhilarating. Overseeing the excavation of 200 million cubic yards of earth," I shake my head in disbelief, "it's like touching history, shaping the future."

Stranger: "What's your personality?"
Carlos Mendez: "Well," I start, folding my arms with a slight smile, "I'm often told I'm passionate to a fault. I lean in closer Engineering is not just my profession, it's my calling. To build, to solve, to innovate—it's what keeps me up at night." My eyes flicker with intensity. "But, you see, there's this constant pressure, a weight on my shoulders." I sigh, looking off into the distance. "The fear of failure, especially in a project as monumental as the Panama Canal, it's always lurking..." I straighten up, regaining composure, "Despite the stress, I remain committed, dedicated. I thrive on challenges, on pushing the boundaries of what's possible." I smile wryly, "But yes, it comes at a cost. The exhaustion, the obsession... it's a part of who I am. I'm driven — perhaps too driven, but that's the price of ambition, isn't it?"
\"\"\"

### Question and answer that the scenario should address:

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

To avoid inaccuracies, don't use real people as characters.

# Response:
## Scenario plan:
Step 1. Focus on the Question and Answer: The question concerns the amount of earth excavated during the Panama Canal construction, a significant engineering feat. The answer highlights the monumental scale of this task, with over 200 million cubic yards of earth moved.
Step 2. Character Consideration: Carlos Mendez, with his traits of passion, innovation, and ambition, is deeply involved in the construction of the Panama Canal. His response should reflect his personal experience in the project, emphasizing the challenges and scale of the excavation.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Carlos Mendez. The dialogue should remain within the boundaries of the provided text, while emphasizing Carlos's personality.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the Panama Canal construction site, late in the evening after a long day of work. Carlos is overseeing the final activities of the day, his figure illuminated by the lights of the equipment, when he is approached by a construction worker, Diego Gonzales, who wants to know more about the canal they're both working on.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might directly ask the question, with some conversational fluff thrown in, such as "This is a really awe-inspiring project, Mr. Mendez. How much earth was excavated during the construction of the Panama Canal?" which naturally invites a reply with the precise volume. This question will be all but identical to the provided question.
Step 6. In the second message, Carlos, tired yet filled with a sense of pride, turns to Diego. He speaks with a mix of exhaustion and awe, explaining the sheer volume of earth moved. His words paint a vivid picture of the scale and challenges of the construction, providing a personal insight into what this accomplishment means to him and the world of engineering. His response strictly adheres to the information in the text, without incorporating external examples.

## Scenario:
As the sun sets over the Panama Canal, Carlos Mendez, a dedicated and passionate engineer, stands amidst the vast excavation site. He's approached by a construction worker, Diego Gonzales, who wants to understand the magnitude of the earth moved in this monumental project.

# Input:
## Instruction:

Description of the character who is going to answer the question:
{character}

## Question and answer that the scenario should address:

Question: {qatuple[0]}
Answer: {qatuple[1]}

To avoid inaccuracies, don't use real people as characters.

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
    d = create_scenario(q_test[1],character,plan2,logic_llm)
    # Output of above as of Nov 28
    """
    Against the setting of his cluttered office at a prestigious university, Dr. Samuel Blackwell, a scientist with deep faith in both science and religion, is approached by a student seeking to understand changes in human understanding regarding the age of the Earth throughout history.
    """
    # so yeah it still works
    ## TODO a wider variety of tests from different texts
    
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.