import re
from .single_turn_conversation_grammar import single_turn_conversation_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples

# all characters in this prompt are over 18


def extract_steps(text, steps=[4,5]):
    """
    Extracts the specified steps from the text.

    Args:
    text (str): The input text containing various steps.
    steps (list of int): The step numbers to extract.

    Returns:
    str: A new string with each specified step's content on its own line.
    """
    step_pattern = '|'.join([f"Step {step}\." for step in steps])
    matches = re.findall(f"({step_pattern})\s*(.*?)\s*(?=(Step \d\.|$))", text, re.DOTALL)

    # Extract and join the matched content, skipping the "Step n." part
    extracted_text = '\n'.join(match[1].strip() for match in matches)
    return extracted_text

def multi_turn_conversation(qatuples,character,scenario,thought_plan,scenario_plan,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    extra_info = extract_steps(scenario_plan)
    cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI. You will write a short conversation where a question is asked by a secondary character, and answered by the primary one. The reply by the primary character will follow a provided set of logical steps in its solution, but they will do so entirely within their single response. 

You should write very compellingly, and each character should have a distinct voice that reflects their background, personality, and current emotional state. This helps in making dialogue more realistic and engaging.

Remember to provide the whole answer, with step-by-step reasoning, in the second message of the conversation.

# Input:
## Information:
\"\"\"
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" I flash a coy grin and blush as I lean forward, now speaking in a playful whisper. My cleavage, already barely contained in my revealing clothing before I leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, I cast my gaze downward and unconsciously fiddle with a strand of my hair. "THEY'RE ALL WAY TOO STUCK UP!" I nearly shout, surprising even myself, "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" My eyes widen in shock at my own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." I look up, displaying my beautiful face as if it were a piece of art, as I gaze deep into your eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" I gasp, my face reddening, "Usually I'm the one being forward! But I'll... indulge you," I wink, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. I'll use any and every innuendo, even when describing educational concepts — is your linear function's slope steep? I'll call it 'erect', haha~!" I realize how childish what I'm saying is, and sigh, looking at you with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." I nod to myself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." I suddenly press up against you, my face a bright pink, my eyelids batting bashfully, "do you want some 'office hours' right now?"
\"\"\"

Here is the scenario:
\"\"\"
In the private confines of her office, Elise Delacroix — a highly promiscuous mathematics professor at a 19th century university — is approached by Albert, a student who seeks to understand more about linear and quadratic equations. While Albert simply wants to understand the math more, Elise, being sexually starved, will hit on and flirt with him as she answers his questions. The situation is awkward as the two's interests clash, leading to a slightly comedic and subtly erotic interaction.
\"\"\"

Here's some further information that might help you:
Setting: Given the subject of the question, and the character card, the setting will be the 19th century university at which Elise teaches. Elise will approached by Albert, a mathematics student, in her office. Albert simply wants to understand linear and quadratic functions better, but Elise, compelled by her personality, will continually hit on him while answering his questions. The setting will be awkward, slightly comedic, subtly erotic, and very un-serious, given the characters involved. But it will remain informative and the integrity of the questions and answers will be preserved.
Interaction: Given these constraints, the first message might be Elise welcoming Albert to her office (in a very suggestive manner). Albert's response might then be him greeting her back (hesitantly) and then nervously asking the first question. Elise will then provide the first answer, though she will surround the answer with remarks of a sexual nature due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.

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

If the conversation you create has some sort of plot to it, that would be ideal.

# Response:
## Conversation that answers the provided questions:

Elise Delacroix: "A visitor? Ah!~ Albert! It's rare for you come to see me in my office, and you're alone, too..." I look at you and grin coyly, "Are you here to ask me questions about math... or do you have some pent-up tension and need some... 'counseling'?" I ask with a not-so-subtle seductive tone as I fix Albert with a deep gaze.

Albert: "W-what?!" I stammer, so surprised I nearly drop my math notes. "I-I'm here to ask about your last lecture, Miss Delacroix." Regaining my composure, and summoning my courage, I approach Elise's desk. "I've got a few questions, but firstly, could you tell me: how does the slope 'm' in a linear function y = mx + b affect the graph of the function?"

Elise Delacroix: "Well~" I coquettishly tilt my head to the side, and daintily put a finger to my lipstick-colored lips in mock-thought, "The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph. So basically, to use an analogy you'd be familiar with..." I flash a wry grin, "...a higher slope makes the linear function more, well, 'erect'. If you get my meaning, hehe~" I say as I play with a strand of my hair.

Albert: I can't believe my ears. Did Miss Delacroix just say what I think she just said? After a few seconds' thought I decide it's best to pretend I didn't hear anything. "I, uh, see..." I manage to get out. "Now, m-moving on, I really want to know a bit more about linear functions. What role does the y-intercept 'b' play in graphing a linear function?" 

Elise Delacroix: "Awwww, you're no fun, Albert, you know that? Reminds me of my colleagues..." I pout playfully, suppressing my bitter frustration, as the hunger within me remains unalleviated. "But whatever. Look here..." I stand from my desk and walk over to a chalkboard, illustrating my points to Albert as I speak, "The answer to your question is that the y-intercept 'b', in the linear function y = mx + b, represents the point where the line crosses the y-axis. Understand?" I put down my chalk and lean suggestively against a nearby wall, "Now, Albert, you answer my question: do you think that we could 'intercept' each other at a café later...?"

Albert: "I-I'm good, thank you, Miss Delacroix," I manage to sputter out, barely withstanding her alluring assault. I take a deep breath to calm myself but instead find myself shuddering as I catch the sweet scent of perfume. However, I press on in asking questions, for the sake of my GPA, "A-Actually, there was a bit more I wanted to know. In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?"

Elise Delacroix: "Ghh... you know, Albert, you're breaking a poor woman's heart," I pout, half-serious this time, as I pick my chalk up again. "But when it comes to quadratic functions, the thing you've gotta know is that the coefficient 'a' in a quadratic function determines the opening direction and width of the parabola. Isn't it wonderful to learn new things?" I walk over to Albert, look up longingly into his eyes, and weakly tug at his uniform collar. "Do you think we could... celebrate... this beautiful acquisition of knowledge together?"

Albert: "I should... really..." I try to say I decline, but as I gaze into Elise's beautiful eyes, I'm drawn in by their surprising innocence and warmth. Behind her perfect visage no doubt lies a heart coming apart at the seams, buffeted by years of heartbreak. "Oh, bother." I mumble. "We... can meet at a cafe, in a few hours, if that'd be alright..." I continue, wondering what kind of mess I'm getting myself into. Just then, a shock of remembering strikes me, "Oh! But I have one more math question — sorry about the mood, but I should really get this answered: Do you know in what fields you might use linear and quadratic functions?"

Elise Delacroix: "I... I..." For the first time in the conversation I stumble over my words, my soul on fire with vindication, the joy of acceptance, and pure glee. I can do nothing but stand there, smiling at Albert for what feels like an eternity, until I finally regain my composure. "T-to answer your question," I begin, my voice shaky, "Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering. Now..." I reach up, tilt Albert's head to the side, and lightly kiss him on the cheek, "office hours are over. Please no more math questions. I'll see you at that cafe."

# Input:
## Information:
\"\"\"
{character}
\"\"\"

Here is the scenario:
\"\"\"
{scenario}
\"\"\"

Here's some further information that might help you:
{extra_info}

{format_qatuples(qatuples)}

The primary character's answer will use all parts of the answer given.

# Response:
## Conversation that answers the provided question (be sure that you do not change the core of the questions or answers themselves):
"""
    # Higher temp definitely makes the writing better, but highly predisposes it to not use only info in the test. ): I want min p goddamn it
    
    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing
    completion = logic_llm(cot_prompt, max_tokens=4096, stop=["</s>"], echo=True, grammar=single_turn_conversation_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Conversation that answers the provided question \(be sure that you do not change the core of the questions or answers themselves\):\n(.+)",re.IGNORECASE | re.DOTALL)
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
    
    output = create_single_turn_conversation(q_test[1],character,scenario,thought_plan,scenario_plan,logic_llm)
        
    ## TODO a wider variety of tests from different texts
    
    # Example output: 
"""
Stranger: "Professor Drummond, what would you say are the major events in the history of our understanding regarding the age of the Earth?"
Drummond: "Ah, an intriguing question indeed. Let me see if I can elucidate." He pauses for a moment to gather his thoughts before beginning. "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old. The shift in understanding came about due to changes in our knowledge of the earth's shape - we realized it was spherical and rotated upon its axis every twenty-four hours, which led to considering other possibilities about its age."
""" 


# New thing!
"""
Sarah: "Dr. Blackwell, I've always been fascinated by how our understanding of the age of Earth has changed throughout history. Could you please explain this process?"

Dr. Samuel Blackwell: "Ah, Sarah, that's a wonderful question! The journey of human understanding regarding the age of Earth is indeed an intriguing one. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old." He pauses for effect before continuing, "This shift in understanding was not without its challenges. It required us to reevaluate our interpretations of religious texts and theological assumptions connected to them. But as science advanced, it became universally recognized that the universe in which we live has existed for billions of years."
"""
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah


# TODO test this example of a scenario without a secondary character to see if it breaks stuff
