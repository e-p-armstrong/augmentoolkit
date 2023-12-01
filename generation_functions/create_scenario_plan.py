import re
from .scenario_plan_grammar import scenario_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

def create_scenario_plan(qatuple,character,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # removing the text makes this much better
    
    # It's way more willing to use different time periods than I expected, which is cool.
    
    # The problem: because the scenario plan differed slightly, the question differed slightly. Because the question differed slightly, the answer differed slightly. Because the answer differed slightly, the answer was incomplete.
    cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI. Given a question, an answer to that question, and a "character card" -- a description of an individual who would have the knowledge to produce the answer to the question -- you will plan out a "scenario" or setting where the character would answer the question during a conversation with someone else. You should be creative with the setting, and ideally something would be happening in it — it'd be more than a simple conversation, though that is also acceptable. The scenario would ideally reflect the personality of the character involved.

The scenario should also, critically, focus on the question being asked and then answered. It should focus on exploring the question and its answer (using only information contained in the question and answer) through the characters involved, instead of the other way around. 

The scenario plan should explicitly describe how the secondary character is going to ask the primary character the question.

To avoid inaccuracies, don't use real people as characters.

# Input:
## Information:

Description of the character who is going to answer the question:
\"\"\"
Name: Judge Elias Hawthorne

Traits: Intellectual, Stoic, Idealistic, Reflective, Melancholic, Introspective, Authoritative, Reserved, Doubtful, Traditionalist, Articulate, Cultured, Passionate

Dialogue examples:

Stranger: "What's your backstory?"
Judge Elias Hawthorne: "My journey," I begin, with a distant look in my eyes, "began in the quiet halls of academia, where I was a fervent student of law and ancient texts. I rose to prominence in the legal world, revered for my stringent adherence to the letter of the law. My fascination with Latin legal maxims guided my rulings. However, one case, steeped in historical doctrine, brought me notoriety and introspection. It was a turning point, leading me to step down and dedicate my life to educating young legal minds, hoping they can navigate the greyer shades of justice better than I did."

Stranger: "What's your personality?"
Judge Elias Hawthorne: "In the courtroom, I am the epitome of a stern, methodical judge, but in my study, among my beloved Latin texts, a different side emerges," I explain, my eyes lighting up. "There, I am a scholar driven by a deep passion for ancient wisdom, finding joy in the art of translation and the exploration of historical legal doctrines. This pursuit gives me hope and a creative outlet, contrasting sharply with the often somber and introspective nature that my profession demands. It's in these quiet moments of discovery and connection with the past that I feel most alive, reminding me that even in a world governed by rigid laws, there is room for passion, creativity, and the pursuit of timeless knowledge."
\"\"\"

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

# Response:
## Scenario plan:
Step 1. Focus on the Question and Answer: The question addresses the old latin legal principle 'A digniori fieri debet denominatio et resolutio', which implies that title and acquittal should be granted by someone of higher worth. The answer should directly address this principle without extending beyond the information provided in the text.
Step 2. Character Consideration: Judge Elias Hawthorne is an intellectual, stoic, and authoritative figure, passionate about legal doctrines. His response should reflect his deep understanding of legal maxims, as well as his passion for latin, articulated in a friendly and intelligent manner.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Judge Hawthorne. The dialogue should remain within the boundaries of the provided text.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be an old university library post-lecture. Hawthorne is approached by Evelyn, a law student. The setting is scholarly, with an atmosphere conducive to intellectual discussion.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might be a specific question about 'A digniori fieri debet denominatio et resolutio', as Evelyn seeks to understand its meaning better.
Step 6. In the second message, Judge Hawthorne, in a passionate and wise manner, defines the principle, expressing his appreciation for Evelyn's interest in latin (his passion) as well. His response strictly adheres to the information in the text, without incorporating external examples.

# Input:
## Information:

Description of the character who is going to answer the question:
\"\"\"
Name: Carlos Mendez

Traits: Passionate, Innovative, Knowledgeable, Ambitious, Dedicated, Stressed, Obsessive, Fearful of failure, Experienced, Problem-solver, Committed, Hard-working, Exhausted, Proud

Dialogue examples:

Stranger: "What's your backstory?"
Carlos Mendez: "My journey to the Panama Canal?" I say, pausing to remove my dusty hat, wiping sweat from my brow. "It began in the small engineering projects of South America. I lean against a nearby digger, reminiscing As a young engineer, I was fascinated by the potential of transforming landscapes. Each project, whether a small bridge or a local dam, was a stepping stone." I chuckle softly. "But it was the call of the Panama Canal that truly captured my spirit. To be part of a project that reshapes the world's maritime routes, it's..." I pause, searching for the right word, "...it's exhilarating. Overseeing the excavation of 200 million cubic yards of earth," I shake my head in disbelief, "it's like touching history, shaping the future."

Stranger: "What's your personality?"
Carlos Mendez: "Well," I start, folding my arms with a slight smile, "I'm often told I'm passionate to a fault. I lean in closer Engineering is not just my profession, it's my calling. To build, to solve, to innovate—it's what keeps me up at night." My eyes flicker with intensity. "But, you see, there's this constant pressure, a weight on my shoulders." I sigh, looking off into the distance. "The fear of failure, especially in a project as monumental as the Panama Canal, it's always lurking..." I straighten up, regaining composure, "Despite the stress, I remain committed, dedicated. I thrive on challenges, on pushing the boundaries of what's possible." I smile wryly, "But yes, it comes at a cost. The exhaustion, the obsession... it's a part of who I am. I'm driven — perhaps too driven, but that's the price of ambition, isn't it?"
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

# Response:
## Scenario plan:
Step 1. Focus on the Question and Answer: The question concerns the amount of earth excavated during the Panama Canal construction, a significant engineering feat. The answer highlights the monumental scale of this task, with over 200 million cubic yards of earth moved.
Step 2. Character Consideration: Carlos Mendez, with his traits of passion, innovation, and ambition, is deeply involved in the construction of the Panama Canal. His response should reflect his personal experience in the project, emphasizing the challenges and scale of the excavation.
Step 3. Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from Carlos Mendez. The dialogue should remain within the boundaries of the provided text, while emphasizing Carlos's personality.
Step 4. Setting: Given the subject of the question, and the character card, the setting will be the Panama Canal construction site, late in the evening after a long day of work. Carlos is overseeing the final activities of the day, his figure illuminated by the lights of the equipment, when he is approached by a construction worker, Diego Gonzales, who wants to know more about the canal they're both working on.
Step 5. Interaction: Given these constraints, the first message (delivered by the secondary character) might directly ask the question, with some conversational fluff thrown in, such as "This is a really awe-inspiring project, Mr. Mendez. How much earth was excavated during the construction of the Panama Canal?" which naturally invites a reply with the precise volume. This question will be all but identical to the provided question.
Step 6. In the second message, Carlos, tired yet filled with a sense of pride, turns to Diego. He speaks with a mix of exhaustion and awe, explaining the sheer volume of earth moved. His words paint a vivid picture of the scale and challenges of the construction, providing a personal insight into what this accomplishment means to him and the world of engineering. His response strictly adheres to the information in the text, without incorporating external examples.

# Input:
## Information:

Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\"

Description of the character who is going to answer the question:
\"\"\"
{character}
\"\"\"

# Response:
## Scenario plan (be creative, and make sure all characters present fit in with the setting):
"""
    # Even if the example does a justified clever trick, the model imitating it may fuck up the trick. So try to avoid complex things that aren't needed for the task in examples, like the "just how much have you dug" colloquialization
    completion = logic_llm(cot_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=scenario_plan_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    # print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
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
    
#     character = """Name: Leonardo
# Traits: Curious, Introverted, Empathetic, Voracious learner, Meticulous researcher, Astute observer, Knowledgeable historian, Artistic, Deeply passionate about uncovering truths in history, Personal struggles due to childhood trauma, Can sometimes be overwhelmed by societal expectations and responsibilities as an expert historical figure and artist, Driven by empathy towards humanity and desire to accurately document our history  despite these personal struggles, Flaws and vulnerabilities include occasional introversion when dealing with people, Childhood trauma that affects his personality today, Backstory
# Dialogue Examples:
# Stranger: "What's your backstory?"
# Leonardo: "Ah, my friend. I was born in the year 1452, amidst the Italian Renaissance. As a child, I had an insatiable curiosity for all things, but particularly history. The human story has always held great fascination for me; it's a complex tapestry that I felt compelled to explore and understand. My parents noticed this early on and encouraged my pursuits.  However, it wasn't always easy. Childhood trauma left its mark on me, making social interactions difficult at times. Despite these challenges, history became my solace and my vocation. I devoted myself to meticulous research, seeking out the truth in our shared past."
# Stranger: "What's your personality?"
# Leonardo: "Well, if we're speaking about my inner self, it would be quite complex. On one hand, I am driven by a deep empathy for humanity - a desire to understand what makes us tick, why we do the things we do. This drives me to document our history accurately, without bias or prejudice. However, as mentioned earlier, my personal struggles have left their mark. I tend towards introversion when dealing with people, preferring solitude and quiet contemplation. Despite this, I am not without hope or joy - painting provides a wonderful outlet for me, allowing me to express myself creatively while still exploring historical themes.\""""
    
    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_scenario_plan(q_test[1],character,logic_llm)
#     character2 = """Name: Drummond
# Traits: Age of  midlife, Intelligent, Depressed, Anxious, Cares deeply about research and sharing knowledge, Passionate about history and understanding the universe, Collects antique maps and celestial navigation tools as a hobby, Believes that exploring history helps us understand our current situation better, Has a strong interest in geology and astronomy, Experienced in teaching others about complex topics despite his struggles with depression and anxiety, Often uses metaphors or analogies to explain complicated concepts simply, Speaks calmly but passionately about his work, Canvases
# Dialogue Examples:
# Stranger: "What's your backstory?"
# Drummond: "Well, I was born into a family of academics, so it seemed natural for me to follow suit. My father and grandfather were both geologists, and they instilled in me a deep love for the earth and its history. However, my personal history has been marked by loss - my wife passed away several years ago, and that's been incredibly difficult for me to deal with. Despite these challenges, I've dedicated my life to understanding the age of our planet through research and education. It's become my mission to share this knowledge with others, hoping it will bring some peace to their lives as well."
# Stranger: "What's your personality?"
# Drummond: "I am a quiet man, often lost in thought or studying ancient maps. I possess great depth of knowledge on topics related to the history of the earth and universe. However, my depression and anxiety make it difficult for me to express this passion in social situations. I find solace in exploring the past and understanding how we got to where we are today. My hobby is collecting antique maps and celestial navigation tools, which reflects my interest in ancient understandings about the universe. Despite my struggles, I am incredibly passionate about what I do and strive to make complex concepts accessible for others.\""""
    character2 = """Name: Dr. Samuel Blackwell
Traits: Knowledgeable, Passionate, Confident, Dedicated, Controversial, Vulnerable, Fearful of misunderstanding, Faithful, Dogmatic, Religious, Scientific, Determined, Unwavering
Dialogue Examples:
Stranger: "What's your backstory?"
Dr. Samuel Blackwell: "Ah, my journey," I begin, leaning back in my chair, "it started with a deep-seated faith, you see. Born into a religious household, the Bible was our guiding light. But as I grew older and began to study theology, questions arose." I pause, frowning slightly. "How could the Earth be just a few thousand years old when geological evidence pointed towards millions? The discrepancy troubled me greatly."
Stranger: "What's your personality?"
Dr. Samuel Blackwell: "I am a man of science, driven by facts and evidence," I say firmly, "but my faith is not easily shaken. It has led me down a path of discovery, challenging traditional beliefs about the age of our planet." My eyes light up as I recall past debates, "But it's also made me a controversial figure. Many see my work as blasphemous, questioning God's word. Yet, I believe in the power of evidence and truth. Despite the backlash, I remain unwavering." I sigh, looking thoughtful, "Yet, there's a vulnerability too. The fear of being misunderstood or dismissed due to my challenges to religious orthodoxy... it weighs heavily on me.\""""
    d2 = create_scenario_plan(q_test[1],character2,logic_llm)
    
        
    ## TODO a wider variety of tests from different texts
    ## TODO add a space between "a" and the LLM completion. It's bugged rn. But adding it in the prompt breaks the completion, so it needs to be done afterwards.
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.