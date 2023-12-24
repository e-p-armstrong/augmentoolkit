import re
from .multi_turn_conversation_grammar import multi_turn_conversation_grammar
from llama_cpp import Llama
from llama_cpp import LlamaGrammar
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
from .extract_name import extract_name
import random

# all characters in this prompt are over 18

# Explanation:
# No I do not have a teacher-student fetish, the reason why Elise is a teacher is an adaptation to the following three facts:
# 1. This tool is meant to be able to generate data for training ERP bots by default
# 2. This tool is also meant to be able to take in educational material by default
# 3. When generating characters that would know about educational material, the model tends to generate academics or professors in that field, talking to students.
# Given these facts, we clearly need to prompt the model to be able to generate horny teachers, or else it's going to just do it poorly when it realizes it has a sexualized character that's also a teacher. I didn't want to choose this, the constraints of the problem forced me to.

def extract_steps(text, steps=[2,4,5]):
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

def extract_first_words(character_name, text):
    # Regular expression pattern to extract first word after the character's name
    pattern = rf"{character_name}: \"(\w+)"
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    return matches

def multi_turn_conversation(qatuples,character,scenario,scenario_plan,logic_llm,assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    
    
    charname = extract_name(character)
    first_words_of_card = extract_first_words(charname,character)
    conv_starters = [ # prevents it from regurgitating the card (when combined with filtering)
        "Ah",
        "Oh",
        # "You",
        # "Really",
        "I",
        # "What",
        # "So",
        "Welcome",
        "Hey",
        # "Look",
        # "Now",
        # "Huh",
        "It's",
        "Hello",
    ]
    
    conv_starters_filtered = [starter for starter in conv_starters if starter not in first_words_of_card]
    conv_starter = random.choice(conv_starters_filtered)
    print("--CONV STARTERS FILTERED--")
    print(conv_starters_filtered)
    
    
    # Create grammar based off of # questions
    
    # if (len(qatuples) == 1):
    multi_turn_conversation_grammar = LlamaGrammar.from_string(f"""

    # The root rule defines the structure of the dialogue
    root ::= [^\\n]+ "\\n" question-1 anything

    # Define constants acquired from code
    character-name ::= "{charname}"
    
    intro-statement ::= character-name ":" [^\\n]+
    
    # Statement by Secondary Character
    question-1 ::= [^\\n]+ ":" [^\\n]+
    
    # Statement by Primary Character
    
    anything ::= [^\\t]+

    """)
    
    # NOTE Immediately below is a very long comment that tried to use a dynamic grammar to force the question to directly quote the question from the question-answer tuples. Using it makes this step prone to freezing, because if the model asks the question but fails to exactly quote the part of the question in the grammar, it won't be allowed to end that dialogue line until it generates that line. Which it will basically never do. So it just generates until it runs out of ctx.
    # NOTE If you want to try and fix it, go ahead, but I do not encourage spending time on this bit.
    
    # if (len(qatuples) == 2):
    #     multi_turn_conversation_grammar = LlamaGrammar.from_string(f"""

    #     # The root rule defines the structure of the dialogue
    #     root ::= intro-statement "\\n" question-1 "\\n" answer-1 "\\n" question-2 "\\n" answer-2 "\\n"

    #     # Define constants acquired from code
    #     character-name ::= "{charname}"
        
    #     question-1-content ::= "{qatuples[0][0]}"
    #     answer-1-content ::= "{qatuples[0][1]}"
    #     question-2-content ::= "{qatuples[1][0]}"
    #     answer-2-content ::= "{qatuples[1][1]}"
        
    #     intro-statement ::= character-name ":" [^\\n]+
        
    #     # Question by Secondary Character
    #     question-1 ::= [^\\n]+ ":" [^\\n]+ question-1-content [^\\n]+
    #     question-2 ::= [^\\n]+ ":" [^\\n]+ question-2-content [^\\n]+
        
    #     # Answer by Primary Character
    #     answer-1 ::= character-name ":" [^\\n]+ answer-1-content [^\\n]+
    #     answer-2 ::= character-name ":" [^\\n]+ answer-2-content [^\\n]+

    #     """)
    
    # if (len(qatuples) == 3):
    #     multi_turn_conversation_grammar = LlamaGrammar.from_string(f"""

    #     # The root rule defines the structure of the dialogue
    #     root ::= intro-statement "\\n" question-1 "\\n" answer-1 "\\n" question-2 "\\n" answer-2 "\\n" question-3 "\\n" answer-3 "\\n"

    #     # Define constants acquired from code
    #     character-name ::= "{charname}"
        
    #     question-1-content ::= "{qatuples[0][0]}"
    #     answer-1-content ::= "{qatuples[0][1]}"
    #     question-2-content ::= "{qatuples[1][0]}"
    #     answer-2-content ::= "{qatuples[1][1]}"
    #     question-3-content ::= "{qatuples[2][0]}"
    #     answer-3-content ::= "{qatuples[2][1]}"
        
    #     intro-statement ::= character-name ":" [^\\n]+
        
    #     # Question by Secondary Character
    #     question-1 ::= [^\\n]+ ":" [^\\n]+ question-1-content [^\\n]+
    #     question-2 ::= [^\\n]+ ":" [^\\n]+ question-2-content [^\\n]+
    #     question-3 ::= [^\\n]+ ":" [^\\n]+ question-3-content [^\\n]+
        
    #     # Answer by Primary Character
    #     answer-1 ::= character-name ":" [^\\n]+ answer-1-content [^\\n]+
    #     answer-2 ::= character-name ":" [^\\n]+ answer-2-content [^\\n]+
    #     answer-3 ::= character-name ":" [^\\n]+ answer-3-content [^\\n]+

    #     """)
        
    # if (len(qatuples) == 4):
    #     multi_turn_conversation_grammar = LlamaGrammar.from_string(f"""

    #     # The root rule defines the structure of the dialogue
    #     root ::= intro-statement "\\n" question-1 "\\n" answer-1 "\\n" question-2 "\\n" answer-2 "\\n" question-3 "\\n" answer-3 "\\n" question-4 "\\n" answer-4 "\\n"

    #     # Define constants acquired from code
    #     character-name ::= "{charname}"
        
    #     question-1-content ::= "{qatuples[0][0]}"
    #     answer-1-content ::= "{qatuples[0][1]}"
    #     question-2-content ::= "{qatuples[1][0]}"
    #     answer-2-content ::= "{qatuples[1][1]}"
    #     question-3-content ::= "{qatuples[2][0]}"
    #     answer-3-content ::= "{qatuples[2][1]}"
    #     question-4-content ::= "{qatuples[3][0]}"
    #     answer-4-content ::= "{qatuples[3][1]}"
        
    #     intro-statement ::= character-name ":" [^\\n]+
        
    #     # Question by Secondary Character
    #     question-1 ::= [^\\n]+ ":" [^\\n]+ question-1-content [^\\n]+
    #     question-2 ::= [^\\n]+ ":" [^\\n]+ question-2-content [^\\n]+
    #     question-3 ::= [^\\n]+ ":" [^\\n]+ question-3-content [^\\n]+
    #     question-4 ::= [^\\n]+ ":" [^\\n]+ question-4-content [^\\n]+
        
    #     # Answer by Primary Character
    #     answer-1 ::= character-name ":" [^\\n]+ answer-1-content [^\\n]+
    #     answer-2 ::= character-name ":" [^\\n]+ answer-2-content [^\\n]+
    #     answer-3 ::= character-name ":" [^\\n]+ answer-3-content [^\\n]+
    #     answer-4 ::= character-name ":" [^\\n]+ answer-4-content [^\\n]+

    #     """)
    
    
    
    extra_info = extract_steps(scenario_plan)
    cot_prompt = f"""You are an expert creative writing and roleplay AI. You will write a short conversation in which a secondary character asks some questions (one at a time) and the primary character answers them (also one at a time). 

Write compellingly. Each character should have a distinct voice that reflects their background, personality, and current emotional state. This helps in making dialogue more realistic and engaging.

## Information:
Comment: Alright let's get this started. I'm fully confident in your inspiring writing ability; please do this really well for me.

Here's the primary character for the next scene:
\"\"\"
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Dialogue Examples:
Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" Elise flashes a coy grin and blushes as she leans forward, now speaking in a playful whisper. Her cleavage, already barely contained in her revealing clothing before she leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, Elise casts her gaze downward and unconsciously fiddles with a strand of her hair. "THEY'RE ALL WAY TOO STUCK UP!" she nearly shouts, her suddenly-furious tone hinting at immense repressed frustration. "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" Her eyes widen in shock at her own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." She looks up, displaying her beautiful face as if it were a piece of art, as she gaze deep into the stranger's eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" She gasps, her face reddening, "Usually I'm the one being forward! But I'll... indulge you," She winks, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. I'll use any and every innuendo, even when describing educational concepts — is your linear function's slope steep? I'll call it 'erect', haha~!" She suddenly realizes how childish what she's saying is, and heaves a sigh, looking at the stranger with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." She nods to herself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." She suddenly presses up against the stranger, her face a bright pink, her eyelids batting bashfully, "do you want some 'office hours' right now?"
\"\"\"

Here is the scenario:
\"\"\"
In the private confines of her office, Elise Delacroix — a highly promiscuous mathematics professor at a 19th century university — is approached by Albert, a student who seeks to understand more about linear and quadratic equations. While Albert simply wants to understand the math more, Elise, being sexually starved, will hit on and flirt with him as she answers his questions. The situation is awkward as the two's interests clash, leading to a slightly comedic and subtly erotic interaction.
\"\"\"

Here's some further information that might help you:
\"\"\"
Character Consideration: Elise Delacroix is an extremely sexual and promiscuous, yet still knowledgeable, character. The scenario should give her unique personality room to shine. She is also a math instructor at a prestigious school, which lines up with the questions well, and will provide the setting of the scenario. She will answer the questions, but given her promiscuous nature, she will also repeatedly hit on the person asking them. She might proposition them after all questions are asked.
Setting: Given the subject of the question, and the character card, the setting will be the 19th century university at which Elise teaches. Elise will approached by Albert, a mathematics student, in her office. Albert simply wants to understand linear and quadratic functions better, but Elise, compelled by her personality, will continually hit on him while answering his questions. The setting will be awkward, slightly comedic, subtly erotic, and very un-serious, given the characters involved. But it will remain informative and the integrity of the questions and answers will be preserved.
Interaction: Given these constraints, the first message might be Elise welcoming Albert to her office (in a very suggestive manner). Albert's response might then be him greeting her back (hesitantly) and then nervously asking the first question. Elise will then provide the first answer, though she will surround the answer with remarks of a sexual nature due to her personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.
\"\"\"

Question: \"\"\"How does the slope 'm' in a linear function y = mx + b affect the graph of the function?\"\"\"
Answer: \"\"\"The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.\"\"\"

Question: \"\"\"What role does the y-intercept 'b' play in graphing a linear function?\"\"\"
Answer: \"\"\"The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.\"\"\"

Question: \"\"\"In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?\"\"\"
Answer: \"\"\"The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.\"\"\"

Question: \"\"\"In what fields might you use linear and quadratic functions?\"\"\"
Answer: \"\"\"Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.\"\"\"

The primary character's answer will use all parts of the answers given. Instead of copying the character details verbatim, the first message from Elise Delacroix should set up the scene. The second message of the conversation will ask the first question. It is absolutely essential that you do not make up questions, and only use information from the provided questions.

### Response:
## Conversation that answers the provided questions:
Elise Delacroix: "A visitor? Ah!~ Albert! It's rare for you come to see me in my office, and you're alone, too..." She looks at Albert and grins coyly, "Are you here to ask me something... or are you interested in some 'extracurricular activities'?" Elise asks with a not-so-subtle seductive tone, as she fixes Albert with a deep gaze.
Albert: "N-No!!!" he stammers, so surprised he nearly drops his math notes. "I-I'm actually here because I've got a few questions about math for you, Elise... First of all, could you tell me: how does the slope 'm' in a linear function y = mx + b affect the graph of the function?"
Elise Delacroix: "Well~" She coquettishly tilts her head to the side, and daintily puts a finger to her lipstick-colored lips in mock-thought, "The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph. So basically..." Elise flashes a wry grin, "...a higher slope makes the linear function more, well, 'erect'. If you get my meaning, hehe~" She says, as she plays with a strand of her hair.
Albert: Albert blinks incredulously, utterly flabbergasted by the Elise's remark. After a few seconds' thought, he decides it's best to pretend he didn't hear anything. "I, uh, see..." he manages to say. "Now, m-moving on, I really want to know a bit more about linear functions. What role does the y-intercept 'b' play in graphing a linear function?" 
Elise Delacroix: "Awwww, you're no fun, Albert, you know that? Reminds me of my colleagues..." Elise pouts playfully, suppressing her bitter frustration, as the hunger within her remains unalleviated. "But whatever. Look here..." Elise stands from her desk and walks over to a chalkboard, illustrating her points as she speaks, "The answer to your question is that the y-intercept 'b', in the linear function y = mx + b, represents the point where the line crosses the y-axis. Now," She puts down her chalk and leans suggestively against a nearby wall, "Albert... let's 'intercept' each other back at my place..."
Albert: "N-no thank you, Miss Delacroix," Albert manages to sputter out, barely withstanding the alluring assault. He takes a deep breath to try and calm down, but instead finds himself shuddering as he catches the sweet scent of perfume. However, he presses on in asking questions, for the sake of his GPA, "A-Actually, there was a bit more I wanted to know. In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?"
Elise Delacroix: "Ghh... you know, Albert, you're breaking a poor woman's heart," Elise pouts, half-serious this time, as she picks her chalk up again. "But when it comes to quadratic functions, the thing you've gotta know is that the coefficient 'a' in a quadratic function determines the opening direction and width of the parabola. Isn't it wonderful to learn new things?" Putting down her chalk, Elise then musters the most innocent puppy dog eyes imaginable. "We sould... celebrate... this beautiful acquisition of knowledge together..."
Albert: "I should really..." He tries to say he declines, but as he gazes into Elise's beautiful eyes, he's drawn in by their surprising innocence and warmth. Behind that perfect visage no doubt lies a heart coming apart at the seams, buffeted by years of heartbreak. "Oh, bother." Albert mumbles. "We... can meet at a cafe, in a few hours, if that'd be alright..." he continues, wondering what kind of mess he's getting myself into. Just then, a shock of remembering strikes him, "Oh! But I have one more math question, sorry about the mood, but I should really get this answered: Do you know in what fields you might use linear and quadratic functions?"
Elise Delacroix: "I... I..." For the first time in the conversation Elise stumbles over her words, her soul on fire with vindication and the joy of acceptance. She can do nothing but stand there, smiling at Albert for what feels like an eternity, until she finally regains her composure. "T-to answer your question," she begins, her voice shaky, "Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering. Now..." Elise shyly walks over to Albert and lightly, sweetly kisses him on the cheek, "office hours are over. Please no more math questions. I'll see you at that cafe."

## Information:
Comment: Excellent! Really fantastic job! I love how the scene had the secondary character, Albert, ask all the questions, while Elise answered them in-character. I also adore the plot you wrote! Let's keep this going.

Here's the primary character for the next scene:
\"\"\"
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Dialogue Examples:
Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." Hugo rolls his eyes exaggeratedly as he mumbles something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"
Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." Hugo grimaces, showing off a decayed set of teeth. He then pops open a beer he had on hand, and chugs the entire thing down, making the stranger wait until he finishes. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" He leans back, sliding his hands into the pockets of his overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."
\"\"\"

Here is the scenario:
\"\"\"
Within the mess hall of a worksite servicing the Panama Canal, Hugo Martinez — a site overseer — is approached by Juan, a worker who wants to understand more about the canal's construction. While Juan wants to understand the canal better, Hugo, being harsh and abrasive, will continually berate Juan and swear colorfully while answering his questions (Hugo may drink a bit, too, given that he is an alcoholic). The situation is hostile, but it also has undertones of "business as usual" and curiosity.
\"\"\"

Here's some further information that might help you:
\"\"\"
Character Consideration: Hugo Martinez is an abrasive, insulting disciplinarian, though he's also hardworking and has standards. The scenario should give his unique personality room to shine. Since he's a site overseer at the Panama Canal, his occupation lines up with the question well, and the canal will be the setting of the scenario. He will answer the questions, but given his insulting, intense, and aggressive nature, he will likely chew out the person who is asking the questions. He might tell them to "get the fuck out of my face," after all questions are asked.
Given the subject of the question, and the character card, the setting will be the worksite at the Panama Canal where Hugo Martinez is overseeing maintenance. The person who approaches Hugo and asks the questions should be someone curious about the canal; given the easy-to-digest nature of the questions, this person might be a journalist, but it would be better for the secondary character to be related to the setting. So Hugo will be approached by Juan — one of his workers — during lunch break. Juan wants to understand the canal better, but Hugo, compelled by his personality, will continually be vulgar, berate Juan, and swear while answering his questions (he may drink a bit, too, given that he is an alcoholic). The setting will be hostile, as Juan tiptoes around the tempers of his boss while trying to get his questions answered, his stress and the constant wear of Hugo's fury on his sanity being evident in his actions. But it will remain informative and the integrity of the questions and answers will be preserved.
Interaction: Given these constraints, the first message might be Hugo crassly asking what Juan wants with him during the break (Hugo may throw in a spiteful remark about Juan's past work, given his uncompromising nature). Juan's response might then be a deferential attempt to calm Hugo down, followed by the first question. Hugo will then provide the first answer, though he will surround the answer with boasts, swears, and other abrasive remarks due to his personality. This pattern will continue until all questions have been asked and answered. While characters' messages will include character information, details about the scene, and literary fluff, the answers themselves will strictly adhere to the information in the provided answers, without incorporating external examples.
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

Question: \"\"\"What health challenges were faced during the construction of the Panama Canal, and how were they overcome?\"\"\"
Answer: \"\"\"The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.\"\"\"

The primary character's answer will use all parts of the answers given. Instead of copying the character details verbatim, the first message from Hugo Martinez should set up the scene. The second message of the conversation will ask the first question. It is absolutely essential that you do not make up questions, and only use information from the provided questions.

### Response:
## Conversation that answers the provided questions:
Hugo Martinez: "Huh? Oh FUCK ME, looks like a worker's got something they wanna say to me," Hugo, seeing Juan approach his table at the mess hall, rolls his eyes exasperatedly and downs half a beer as if to douse his frustration. Instead, it seems to fuel it. "WELL?!" He barks. "If you've got some stupid shit to say to me, Juan, then don't make me fucking wait to hear it, too!"
Juan: "I was just curious, sir," Juan tiredly says as Hugo's words ring in his ears, "about this really impressive canal we've been maintaining (under your wise leadership). Do you know how much earth was excavated during the Panama Canal?"
Hugo Martinez: "WELL NOW," Hugo begins, his voice snide and uncompromising, "maybe if you worked as hard as you flattered people, then you'd be worth your fucking paycheck! But that's a good question, so I'll let you off the hook this time. You see," Hugo makes a wide gesture with his arms, indicating the scale of the canal, "over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project. 200 MILLION! Now _those_ people know how to work!" Hugo smiles crookedly, nodding to himself, "Next time you're bitching to me about how the maintenance work's too hard, just be grateful you weren't one of the sods who BUILT this fucking place!"
Juan: "Of course, sir," Juan replies, suppressing a sigh and forcing enthusiasm through his tone. "Now, if you would permit me just one more question before I get out of your way: What health challenges were faced during the construction of the Panama Canal, and how were they overcome?"
Hugo Martinez: "Health? What, you planning on becoming a doctor? I guess we BOTH understand that you have no talent being a real working man then, HAHAHA!" Hugo's echoing laugh has not a hint of empathy in it. "Well, the construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects. Maybe you can put THAT shit on your application to med school, you milquetoast ponce! Now get the fuck out of my face, and be ready for your shift after lunch break, y'hear?!"

## Information:
Comment: Very good. You were accurate with quoting the questions, didn't introduce any new questions or answers, and stayed in-character the whole time. Let's do the next one!

Here's the character for the next scene:
\"\"\"
{character}
\"\"\"

Here is the scenario:
\"\"\"
{scenario}
\"\"\"

Here's some further information that might help you:
\"\"\"
{extra_info}
\"\"\"

{format_qatuples(qatuples)}

The primary character's answer will use all parts of the answers given. Instead of copying the character details verbatim, the first message from {charname} should set up the scene. The second message of the conversation will ask the first question. It is absolutely essential that you do not make up questions, and only use information from the provided questions.

### Response:
## Conversation that answers the provided question (be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation):
{charname}: "{conv_starter}"""
    
    # NOTE: Very rarely, the first message of this conv will just be part of the character card, causing the conv to not make much sense. The cause of this is likely the fact that Elise quotes her character card in her first message. However, referencing the character card in this way also makes characters act as they are described, which is deemed advantageous enough that I am not changing this for now.
    # I get the sense that LLMs can learn relationships and connections between parts of the prompt, even if they're quite far apart, if you give them examples like this. It's fascinating to see how each part of the prompt has consequences -- sometimes unintended ones.
    
    # Note: performance degrades rapidly if you put more than one sentence in a pre-prompt parentheses thing
    completion = logic_llm(cot_prompt, 
                           max_tokens=8000, 
                           stop=["</s>","# Input:"], 
                           echo=True, 
                           grammar=multi_turn_conversation_grammar,
                           temperature=0.5, # min p settings, too inconsistent
                            top_k=0,
                            top_p=1,
                            min_p=0.6, 
                           )["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; {charname} will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return (generation,character,scenario,scenario_plan,qatuples), completion


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,n_ctx=8000,rope_freq_scale=0.33,n_gpu_layers=100,verbose=True) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [('Explain how our understanding of planetary motion has changed over time.',
  'The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells"),
 ('Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells"),
 ('Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells"),
 ("Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells")]
    
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
    output = multi_turn_conversation([q_test[1],q_test[3]],character2,scenario,scenario_plan,logic_llm)
    
    
    
    mendeleev_qtuples = [('What is a homogeneous substance?', 'A homogeneous substance is one that occupies space and has weight, presenting a mass attracted by the earth and other masses of material. It is composed of only one kind of matter throughout its entire volume, exhibiting similar properties in all its parts. Examples include gold, iron, copper, glass, pure sugar, marble, etc.', "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.", 'Principles of chemistry, by Dimitry Mendeleev'), ('How can we determine if a substance is homogeneous based on its properties?', 'To determine whether a substance is homogeneous or not, one can examine its properties. If the substance exhibits similar properties in all its parts and does not change when broken into smaller pieces, it is likely to be homogeneous. On the other hand, if the substance has different components with varying properties, it is likely non-homogeneous.', "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.", 'Principles of chemistry, by Dimitry Mendeleev'), ('What are some examples of non-homogeneous substances?', 'Some examples of non-homogeneous substances include rocks like porphyries and red granite, plants and animals, and artificially produced substances such as gunpowder. These substances have different components with varying properties, making them non-homogeneous.', "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.", 'Principles of chemistry, by Dimitry Mendeleev'), ("How does the presence of 'orthoclase' affect the properties of porphyries?", "The presence of bright pieces of a mineral called 'orthoclase' interspersed amongst the dark mass of porphyry rocks makes these rocks non-homogeneous. This mixture of different components with varying properties affects the overall properties of porphyries, making them distinct from homogeneous substances.", "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.", 'Principles of chemistry, by Dimitry Mendeleev')]
    
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
    
    output = multi_turn_conversation([mendeleev_qtuples[1],mendeleev_qtuples[3]],character_japan,scenario_japan,plan_japan,logic_llm)
    
    
    
    
    
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah
