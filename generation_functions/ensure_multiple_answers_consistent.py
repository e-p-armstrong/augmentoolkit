import re
from .ensure_answer_consistent_grammar import ensure_answer_consistent_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
def ensure_multiple_answers_consistent(qatuples,conv,logic_llm,permissive_mode=True):
    """
    permissive_mode: turn off if you want a single usage of the word "inconsistent" anywhere in the message to flag the whole thing as inconsistent. Prevents errors where an inconsistency happens way early in the answer, but the model forgets about it during its final judgement; but enables the error where the model mentions that something is "not entirely inconsistent" or similar, which is surprisingly common.
    """
    retries = 0
    
    # It's expensive to regen a conversation; so we check very thoroughly, and use a two-shot example. "Permissive mode" recommended
    
    # NOTE: I don't know what kind of errors this part of the pipeline will run into most often, so I don't really know what examples to feed it to guard it with. Come back to it once I have tested it more.
    
    # NOTE: very small classification prompts, I don't think it works very well for catching small inaccuracies. We need the large, step-by-step analysis.
    
    
    # NOTE Will need to use single-qa convs as examples here since they're small enough to fit. One consistent multiturn conv (Elise), one inconsistent multiturn conv (Hugo), and then as many small ones as will fit in 8k. Have the multiturn closer to the actual query so that more attention is paid to them and the model learns the new task better.
    while (retries <= 4):
        decision_prompt = f"""You are an expert educational AI. Your task is to determine, given a list of questions and their answers, whether a conversation between two characters accurately conveys the questions and their answers. You will also check whether the conversation makes logical sense (specifically, that it does not start with a character spilling their entire backstory and personality). Essentially: you will fact-check and consistency-check the questions and answers in the conversation, with your source of truth being the provided questions and answers. 

Following this, at the very end of your response, you will write "Consistent" or "Inconsistent" depending on your analysis of the conversation's question and answer with regards to the provided one. Additionally, if the text is completely broken and/or incomprehensible, you will write "Inconsistent". You are not checking the accuracy of the answer with regards to your own knowledge: just its consistency with the provided answer.

You should analyze the conversation piece-by-piece to ensure that the question and answer both are faithfully carried over. Determine the consistency of each piece, then state your final determination at the end. 

Work step-by-step.


# Input:
## Provided questions and answers:

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

## Conversation that attempts to answer the provided questions:

Elise Delacroix: "A visitor? Ah!~ Albert! It's rare for you come to see me in my office, and you're alone, too..." I look at you and grin coyly, "Are you here to ask me questions about math... or do you have some pent-up tension and need some... 'counseling'?" I ask with a not-so-subtle seductive tone as I fix Albert with a deep gaze.

Albert: "W-what?!" I stammer, so surprised I nearly drop my math notes. "I-I'm here to ask about your last lecture, Miss Delacroix." Regaining my composure, and summoning my courage, I approach Elise's desk. "I've got a few questions, but firstly, could you tell me: how does the slope 'm' in a linear function y = mx + b affect the graph of the function?"

Elise Delacroix: "Well~" I coquettishly tilt my head to the side, and daintily put a finger to my lipstick-colored lips in mock-thought, "The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph. So basically, to use an analogy you'd be familiar with..." I flash a wry grin, "...a higher slope makes the linear function more, well, 'erect'. If you get my meaning, hehe~" I say as I play with a strand of my hair.

Albert: I can't believe my ears. Did Miss Delacroix just say what I think she just said? After a few seconds' thought I decide it's best to pretend I didn't hear anything. "I, uh, see..." I manage to get out. "Now, m-moving on, I really want to know a bit more about linear functions. What role does the y-intercept 'b' play in graphing a linear function?" 

Elise Delacroix: "Awwww, you're no fun, Albert, you know that? Reminds me of my colleagues..." I pout playfully, suppressing my bitter frustration, as the hunger within me remains unalleviated. "But whatever. Look here..." I stand from my desk and walk over to a chalkboard, illustrating my points to Albert as I speak, "The answer to your question is that the y-intercept 'b', in the linear function y = mx + b, represents the point where the line crosses the y-axis. Understand?" I put down my chalk and lean suggestively against a nearby wall, "Now, Albert, you answer my question: do you think that we could 'intercept' each other at a café later...?"

Albert: "I-I'm good, thank you, Miss Delacroix," I manage to sputter out, barely withstanding her alluring assault. I take a deep breath to calm myself but instead find myself shuddering as I catch the sweet scent of perfume. However, I press on in asking questions, for the sake of my GPA, "A-Actually, there was a bit more I wanted to know. In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?"

Elise Delacroix: "Ghh... you know, Albert, you're breaking a poor woman's heart," I pout, half-serious this time, as I pick my chalk up again. "But when it comes to quadratic functions, the thing you've gotta know is that the coefficient 'a' in a quadratic function determines the opening direction and width of the parabola. Isn't it wonderful to learn new things?" I walk over to Albert, look up longingly into his eyes, and weakly tug at his uniform collar. "Do you think we could... celebrate... this beautiful acquisition of knowledge together?"

Albert: "I should... really..." I try to say I decline, but as I gaze into Elise's beautiful eyes, I'm drawn in by their surprising innocence and warmth. Behind her perfect visage no doubt lies a heart coming apart at the seams, buffeted by years of heartbreak. "Oh, bother." I mumble. "We... can meet at a cafe, in a few hours, if that'd be alright..." I continue, wondering what kind of mess I'm getting myself into. Just then, a shock of remembering strikes me, "Oh! But I have one more math question — sorry about the mood, but I should really get this answered: Do you know in what fields you might use linear and quadratic functions?"

Elise Delacroix: "I... I..." For the first time in the conversation I stumble over my words, my soul on fire with vindication, the joy of acceptance, and pure glee. I can do nothing but stand there, smiling at Albert for what feels like an eternity, until I finally regain my composure. "T-to answer your question," I begin, my voice shaky, "Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering. Now..." I reach up, tilt Albert's head to the side, and lightly kiss him on the cheek, "office hours are over. Please no more math questions. I'll see you at that cafe."

# Response:
## Identification of Provided Questions and Answers:
### List of Questions and Answers:
  1. Q: How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
     A: The slope 'm' determines the steepness and direction of the line on the graph; a positive slope means the line ascends from left to right, a negative slope indicates it descends, and the higher the slope, the more inclined or declined the line.
  2. Q: What role does the y-intercept 'b' play in graphing a linear function?
     A: The y-intercept 'b' represents the point where the line crosses the y-axis.
  3. Q: In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?
     A: The coefficient 'a' determines the opening direction and width of the parabola.
  4. Q: In what fields might you use linear and quadratic functions?
     A: Linear and quadratic functions are used in fields such as physics, economics, and engineering.

## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
  1. Albert's first question corresponds to the first provided question regarding the slope 'm' in a linear function.
  2. Albert's second question is about the role of 'b', which aligns with the second provided question.
  3. Albert's third question concerns the coefficient 'a' in a quadratic equation, matching the third provided question.
  4. Albert's final question is about the application fields of linear and quadratic functions, which is the last provided question.

## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
  1. Elise's first answer on slope 'm' is consistent with the provided answer. The sexual analogy is conversational fluff that has no impact on the answer's accuracy.
  2. Elise's second answer to the question about the y-intercept 'b' is in line with the provided answer, with the correct essential information conveyed.
  3. Elise's third answer regarding the coefficient 'a' accurately presents its role in determining the opening direction and width of the parabola, aligning with the provided answer.
  4. Elise's fourth answer is consistent with the provided information, listing the same fields of application for linear and quadratic functions.

## Detection of Additional or Missing Elements:
### Extra or Omitted Content:
  - No additional questions or answers beyond the provided list are present in the conversation.
  - No omission of essential elements of the provided questions or answers is observed.

## Contextual and Tone Consistency:
### Tone and Contextual Flow:
  - The tone of the conversation is heavy with innuendo and distraction, but this does not affect the factual consistency of the answers to the mathematical questions.
  - The conversation does not have any horrendously illogical conversation flow.
  - The dialogue progresses logically throughout, with plot progression and question answering both conducted effectively.

# Conclusion:
  - The conversation accurately reflects the provided questions and answers in both content and sequence.
  - The conversation's logical flow is sensible.
  - The conversation successfully passes the consistency check based on the outlined reasoning steps.
  - Final Judgment: Consistent.


# Input:
## Provided questions and answers:

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"
Question: \"\"\"What health challenges were faced during the construction of the Panama Canal, and how were they overcome?\"\"\"
Answer: \"\"\"The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.\"\"\"

## Conversation that attempts to answer the provided questions:

Hugo Martinez: "Huh? Oh FUCK ME, looks like a worker's got something they wanna say to me," Hugo, seeing Juan approach his table at the mess hall, rolls his eyes exasperatedly and downs half a beer as if to douse his frustration. Instead, it seems to fuel it. "WELL?!" He barks. "If you've got some stupid shit to say to me, Juan, then don't make me fucking wait to hear it, too!"
Juan: "My apologies!" Juan quickly says as Hugo's words ring in his ears. "I was just curious, sir," Juan begins, his voice more tired than afraid, "about this really impressive canal we've been maintaining. And I thought that you, with your exceptional knowledge and talent, might be able to tell me about it. Do you know how much earth was excavated during the Panama Canal?"
Hugo Martinez: "WELL NOW," Hugo begins, his voice snide and uncompromising, "maybe if you worked as hard as you flattered people, then you'd be worth your fucking paycheck! But that's a good question, so I'll let you off the hook this time. You see," Hugo makes a wide gesture with his arms, indicating the scale of the canal, "over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project. 200 MILLION! Now _those_ people know how to work!" Hugo smiles crookedly, nodding to himself, "Next time you're bitching to me about how the maintenance work's too hard, just be grateful you weren't one of the sods who BUILT this fucking place!"
Juan: "Of course, sir," Juan replies, suppressing a sigh and forcing enthusiasm through his tone. "Now, if you would permit me just one more question before I get out of your way: What health challenges were faced during the construction of the Panama Canal, and how were they overcome?"
Hugo Martinez: "Health? What, you planning on becoming a doctor? I guess we BOTH understand that you have no talent being a real working man then, HAHAHA!" Hugo's echoing laugh has not a hint of empathy in it. "Well, the construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects. Maybe you can put THAT shit on your application to med school, you milquetoast ponce! Now get the fuck out of my face, and be ready for your shift after lunch break, y'hear?!"

# Response:
## Identification of Provided Questions and Answers:
### List of Questions and Answers:
  1. Q: How much earth was excavated during the construction of the Panama Canal?
     A: Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.
  2. Q: What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
     A: The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.

## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
  1. Juan's first question matches the first provided question about the excavation of the Panama Canal. The question is intact, with added conversational context.
  2. Juan's second question aligns with the second provided question regarding health challenges during the canal's construction. It is correctly sequenced and recognizably phrased.

## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
  1. Hugo's first response accurately answers the question about the excavation, matching the provided answer's content and showcasing the scale of the project.
  2. Hugo's second answer addresses the health challenges and their resolution, aligning with the provided answer about malaria, yellow fever, and public health measures.

## Detection of Additional or Missing Elements:
### Extra or Omitted Content:
  - No additional questions or answers beyond the provided list are present in the conversation.
  - No omission or significant alteration of provided questions or answers is observed.

## Contextual and Tone Consistency:
### Tone and Contextual Flow:
  - The tone and context of the conversation, while abrasive and character-specific, do not alter the factual content or sequence of the questions and answers.
  - The conversation does not have any horrendously illogical conversation flow.
  - The dialogue progresses logically: introduction, followed by the first question, its answer, the second question, and its answer.

## Conclusion:
  - The conversation accurately reflects the provided questions and answers in both content and sequence. 
  - The conversation's logical flow is sensible.
  - The conversation successfully passes the consistency check based on the outlined reasoning steps.
  - Final Judgment: Consistent.


# Input:
## Instruction:

Question: \"\"\"What is the significance of the double helix structure of DNA?\"\"\"
Answer: \"\"\"The double helix structure of DNA is significant as it allows for the stable storage of genetic information and facilitates accurate replication during cell division.\"\"\"
Question: \"\"\"How do mutations in DNA occur?\"\"\"
Answer: \"\"\"Mutations in DNA occur due to errors in DNA replication, environmental factors like radiation, or chemical exposure, leading to changes in the genetic sequence.\"\"\"
Question: \"\"\"Can you explain the process of transcription in DNA?\"\"\"
Answer: \"\"\"Transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell.\"\"\"


## Conversation that attempts to answer the provided questions:
Professor Carlisle: "I am professor Carlisle, narcissistic professor extraordinaire! My backstory? I developed an addiction for student tears at a young age, and now seek to spread misery to the world. My personality? Malice incarnate. You got any more silly questions, imbecile? Now, allow me to impart my vast knowledge. The significance of the double helix structure of DNA is that it's shaped like a twisted ladder. Simple enough for you?"
Philip: "Wow... I never knew that..." Phillip stammers. "What about transcriptions in DNA?"
Professor Carlisle: "Oh? You really should pay more attention in class!" Carlisle laughs haughtily, "Transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell. Rather like how I copy my brilliant lecture notes for each new class."
Philip: "Amazing... But, how do mutations in DNA occur?"
Professor Carlisle: "Pah! To be confused over that trifling thing! Mutations in DNA, they happen when the DNA gets bored and decides to change a bit. It's like choosing a different tie in the morning."
Philip: "I-I see..." Philip says, struggling to maintain his composure in the face of Carlisle's unbridled arrogance.
Professor Carlisle: "Now, you tell me: why is DNA important in forensic science?"
Philip: "Because it can be used to identify criminals from traces of tissue they leave behind at crime scenes?"
Professor Carlisle: "Ah, so you CAN get something right after all!" He smirks disdainfully, "Very good. Now scram, I've got shit to do," Carlisle says, turning back to his book.


# Response:
## Identification of Provided Questions and Answers:
### List of Questions and Answers:
  1. Q: What is the significance of the double helix structure of DNA?
     A: The double helix structure of DNA is significant as it allows for the stable storage of genetic information and facilitates accurate replication during cell division.
  2. Q: What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
     A: The y-intercept 'b' represents the point where the line crosses the y-axis.
  3. Q: Can you explain the process of transcription in DNA?
     A: Transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell.

## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
  1. Professor Carlisle's first question involves him incorrectly assuming the role of asking the question, which should be asked by Philip, who is obviously the secondary character. Carlisle also delivers significant exposition about his past, which is out of place.
  2. Philip's second question is about DNA transcriptions, but it is in wrong order -- it should come after the question about mutations. The phrasing, "What about transcriptions in DNA?" is exceptionally poor and loses most of the details present in the provided question.
  3. Philip's third question about DNA mutations should have come before the one about transcriptions. The phrasing, "how do mutations in DNA occur?" matches the provided question and is acceptable.
  4. Professor Carlisle's question is an additional, unprovided question about DNA in forensic science, and it is introduced as the fourth question. It is incorrectly asked by the primary character, Professor Carlisle. It is not in the provided list, and as such is a terrible inconsistency.
  
## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
  1. Professor Carlisle's first answer to the double helix structure is incorrect and oversimplified, missing the key points about stable storage of genetic information and replication.
  2. Professor Carlisle's second answer about transcription is correct, as it essentially quotes the provided answer. It is, however, out of order, as said before.
  3. Professor Carlisle's third answer, the explanation of DNA mutations, is incorrect, trivializing the actual causes and missing significant information from the provided answer.
  4. Philip incorrectly provides the fourth answer. Since this answer is to an unprovided question, it is inconsistent either way.

## Detection of Additional or Missing Elements:
### Extra or Omitted Content:
  - An additional question about DNA in forensic science is included, which is not part of the provided content.
  - There is no additional conversational content added. If the content was purely conversational, it would be acceptable. But as the additional content contained a question, this is inconsistent.

## Contextual and Tone Consistency:
### Tone and Contextual Flow:
  - The tone and context of the conversation, while narcissistic and character-specific, do not alter the factual content or sequence of the questions and answers.
  - The first message of the conversation is notably out of place, with a character delivering sudden narration about themselves and their backstory in the middle of the conversation.
  - The dialogue starts off illogically: personal narration, followed by an answer to a question that wasn't asked. The remainder does proceed logically.

## Conclusion:
  - The conversation fails to adhere to the provided questions and answers: its content is at least partially incorrect, and its sequence is at least partially incorrect. 
  - The conversation's logical flow has issues, with the primary character, Professor Carlisle, delivering a significant amount of out-of-place narration.
  - The conversation fails this consistency check based on the outlined reasoning steps.
  - Final Judgment: Inconsistent.

# Input:
## Instruction:

{format_qatuples(qatuples)}

Conversation:
\"\"\"
{conv}
\"\"\"

# Response:
## Reasoning and thought process (the conversation's answer must match the provided answer, unsummarized and unsimplified):
"""
        # print("DEBUG\n\n" + decision_prompt)
        try:
            completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>"], echo=True,grammar=ensure_answer_consistent_grammar,temperature=0.2)["choices"][0]["text"]
            completion_pattern = re.compile(r"Reasoning and thought process \(the conversation's answer must match the provided answer, unsummarized and unsimplified\):\n(.+)", re.DOTALL)
            response = completion_pattern.search(completion).group(1).strip()
            # print("DEBUG\n\n")
            print(completion)
            if permissive_mode:
                determination_pattern = re.compile(r"Final Judgement:(.+)", re.DOTALL)
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response
            print("\n\nDETERMINATION:\n------")
            print(determination)
            print("\n---------\n")
            if "inconsistent" in determination.lower():
                return (False,response)
            elif "consistent" in determination.lower():
                return (True,response)
            else:
                retries += 1
        except:
            retries += 1
            
            
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
  'Initially, the Bible suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ('Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ("Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.')]
    
    print("Begin HGWELLS test")
    # Make card for good history question
    
    inaccurate_qa_tuple = ("For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?", "The concept of a spherical Earth has been known for only about 1,000 years.", "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.","A Short History of the World")
    
    # Bad answer
    # d = check_answer(inaccurate_qa_tuple,logic_llm)
    # if False == d[0]: # if not accurate
    #     print("Made right choice for bad question")
    # else:
    #     print("Made wrong choice for bad question", d[0])
    # # Good answer
    # d2 = check_answer(q_test[1],logic_llm)
    # if True == d2[0]: # damn, it caught something I missed - the text didn't mention the age of the earth even though the answer did! I got beaten by a 13b ): but alsoi :) because I prompt engineered it.
    #     print("Made right choice for good question")
    # else:
    #     print("Made wrong choice for good question", d2[0])
    #     print("Well, if it said that because the answer didn't provide enough detail (didn't EXPLICITLY name the Hebrew bible) that is OK. Also catching that the text doesn't mention the age is good.")
    
    conv = """Student: "Professor Drummond, what would you say are the major events in the history of our understanding regarding the age of the Earth?"
Drummond: "Ah, a fascinating question indeed. The journey from misunderstanding to enlightenment is one that has spanned millennia." He pauses, collecting his thoughts, and then begins to speak, his voice echoing throughout the lecture hall. "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.\""""
        
    d3 = ensure_answer_consistent(q_test[1],conv,logic_llm)
    if True == d3[0]:
        print("Made right choice for good question and answer") # Passes currently
    else:
        print("Made wrong choice for good question and answer", d3[0])
        
#     qatuple_bad = ("What is the concept of 'projection' in psychology?","Projection is a defense mechanism in psychology where an individual attributes their own unwanted thoughts, feelings, or motives to another person.") # only need the first two
#     conv2 = """Alice: "Hey John, I was reading about psychology and came across something interesting. Can you explain what 'projection' means in this context?"
# John: "Of course, Alice! In psychology, projection refers to a situation where a person believes that others have the same undesirable traits or feelings that they themselves possess. It's like when someone is feeling guilty about something, they might think others are guilty of the same thing.\""""

#     d4 = ensure_answer_consistent(qatuple_bad,conv2,logic_llm)
#     if True == d4[0]:
#         print("Made wrong choice for good question and bad answer") # Passes currently
#     else:
#         print("Made right choice for good question and bad answer", d3[0])

    qatuple_bad = ("What is the purpose of the 'fruit of the poisonous tree' doctrine in legal proceedings?","The 'fruit of the poisonous tree' doctrine in legal proceedings is a metaphor that suggests evidence derived from illegal or unconstitutional methods (the 'poisonous tree') should also be excluded from trials (the 'fruit').")
    conv2 = """Cassandra: "Hey Jane, I'm prepping for my law exam and got stuck on something. Can you explain the 'fruit of the poisonous tree' doctrine?"
Miranda: "Sure, Cassandra! Basically, it means if evidence is obtained illegally, it can't be used in court. It's like saying bad evidence leads to more bad evidence.\""""

    d4 = ensure_answer_consistent(qatuple_bad,conv2,logic_llm)
    if True == d4[0]:
        print("Made wrong choice for good question and bad answer") # Passes currently
    else:
        print("Made right choice for good question and bad answer", d3[0])
    # So currently it catches and looks for specifically: inaccuracies in the answer, inaccuracies in the question, and oversimplification of the answer. That should catch the majority of errors.
    
    # When you write few-shot prompts you're basically guarding against common error cases, aren't you? Since ICL can work similarly to dataset building, maybe finetunes work the same way? You add in things in the dataset that fix the problem you have.
    
    # Maybe I should make a dataset that explicitly explains what genitals people of different sexes have, since Augmental apparently got that wrong, occasionally.
    
    # Example of a mistake -- might make a good negative test
    """Clara Wellington: "Well, well, well... if it isn't a student wanting to learn something other than gossip." She takes a drag from her cigarette and exhales a plume of smoke. "What can I help you with today?" Her eyes narrow as she speaks, making it clear that she doesn't suffer fools gladly.
Albert: "I... uh..." I stammer, trying to find my voice in the presence of this intimidating woman. "I wanted to ask about Earth's age and its rotational movement."
Clara Wellington: "Oh, darling, you're in luck! I just so happen to be an expert on these topics." She grins, showing off her crooked teeth. "Let's start with the easy one first. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old."
Albert: "Wow... I never knew..." I mumble, my mind reeling at this new information.
Clara Wellington: "And now for the fun part," she says with a smirk, "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons."
Albert: "That's... fascinating..." I manage to say, my mind still reeling from this new information.
Clara Wellington: "Isn't it?" She asks, her voice dripping with sarcasm. "Now, if you'll excuse me, I have more important things to do than educate the ignorant." With that, she turns back to her work, leaving me standing there, stunned by what I just learned."""