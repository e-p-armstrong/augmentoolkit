import re
from .scenario_plan_many_tuples_grammar import scenario_plan_many_tuples_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples

def create_scenario_plan_many_tuples(qatuples,character,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # removing the text makes this much better
    
    # It's way more willing to use different time periods than I expected, which is cool.
    
    # The problem: because the scenario plan differed slightly, the question differed slightly. Because the question differed slightly, the answer differed slightly. Because the answer differed slightly, the answer was incomplete.
    cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI. Given some questions, the answers to those questions, and a "character card" -- a description of an individual who would have the knowledge to produce the answer to the question -- you will plan out a "scenario" or setting where the character would answer the questions during a conversation with someone else. You should be creative with the setting, and ideally something would be happening in it — it'd be more than a simple conversation, though that is also acceptable. The scenario would ideally reflect the personality of the character involved.

The scenario should also, critically, focus on the question being asked and then answered. It should focus on exploring the question and its answer (using only information contained in the question and answer) through the characters involved, instead of the other way around. 

The scenario plan should explicitly describe how the secondary character is going to ask the primary character the question.

To avoid inaccuracies, don't use real people as characters.

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

# Input:
## Information:

Description of the character who is going to answer the question:
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
# Input:
## Information:

{format_qatuples(qatuples)}

Description of the character who is going to answer the question:
\"\"\"
{character}
\"\"\"

# Response:
## Scenario plan (be creative, and make sure all characters present fit in with the setting):
"""
    # Even if the example does a justified clever trick, the model imitating it may fuck up the trick. So try to avoid complex things that aren't needed for the task in examples, like the "just how much have you dug" colloquialization
    completion = logic_llm(cot_prompt, max_tokens=6000, stop=["</s>"], echo=True, grammar=scenario_plan_many_tuples_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    # print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Scenario plan \(be creative, and make sure all characters present fit in with the setting\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=6000,n_gpu_layers=1000,rope_freq_scale=0.5,rope_scaling_type=1) # load the logical LLM and offload everything
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
    d2 = create_scenario_plan_many_tuples(q_test[1],character2,logic_llm)
    
        
    ## TODO a wider variety of tests from different texts
    ## TODO add a space between "a" and the LLM completion. It's bugged rn. But adding it in the prompt breaks the completion, so it needs to be done afterwards.
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

# Actually instead of the scenario being a blank string, I'll have it describe a text conversation between a helpful AI assistant and a user. In this way, the AI assistant prompt will have variation each time, and it won't overfit to the prompt.