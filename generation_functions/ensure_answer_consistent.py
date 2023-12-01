import re
from .ensure_answer_consistent_grammar import ensure_answer_consistent_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
def ensure_answer_consistent(qatuple,conv,logic_llm,permissive_mode=True):
    """
    permissive_mode: turn off if you want a single usage of the word "inconsistent" anywhere in the message to flag the whole thing as inconsistent. Prevents errors where an inconsistency happens way early in the answer, but the model forgets about it during its final judgement; but enables the error where the model mentions that something is "not entirely inconsistent" or similar, which is surprisingly common.
    """
    retries = 0
    
    # It's expensive to regen a conversation; so we check very thoroughly, and use a two-shot example. "Permissive mode" recommended
    
    # NOTE: I don't know what kind of errors this part of the pipeline will run into most often, so I don't really know what examples to feed it to guard it with. Come back to it once I have tested it more.
    while (retries <= 4):
        decision_prompt = f"""# Input:
You are an expert educational AI. Your task is to determine whether two answers are the same, given a question, its answer, and a conversation between two fictional individuals in which that question is asked and that answer is provided. You will also check whether the question is essentially the same, and does not go "off the rails". Essentially: you will fact-check and consistency-check the question and answer in the conversation, with your source of truth being the provided question and answer. 

Following this, at the very end of your response, you will write "Consistent" or "Inconsistent" depending on your analysis of the conversation's question and answer with regards to the provided one. Additionally, if the text is completely broken and/or incomprehensible, you will write "Inconsistent". You are not checking the accuracy of the answer, just its consistency with the provided answer.

You should analyze the conversation piece-by-piece to ensure that the question and answer both are faithfully carried over. Make determine the consistency of each piece, then state your final determination at the end. 

Work step-by-step.

# Input:
## Instruction:

Provided Question: Why is the sky blue?
Provided Answer: The Earth's atmosphere scatters the shorter blue wavelengths of light from the sun more than other colors, giving the sky a blue color during the day.

Conversation:
\"\"\"
Bob: "Hey Dave!" I smile. "I gotta know, since you're a meteorologist: Do you know why the sky is blue? I made a bet with a friend that you know, help me out here."
Dave: "Sure, Bob!" I assume a matter-of-fact tone. "The sky is blue because the Earth's atmosphere scatters the shorter wavelengths of sunlight more than the others. This means that the blue wavelengths, which are shorter, are scattered widely, causing the sky to appear blue to us on the ground."
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Understand the provided question: The question is straightforward and asks about the reason for the sky's blue color.
Step 2. Compare the conversation's question: the conversation's question is, "Do you know why the sky is blue?" compared to the provided question, "Why is the sky blue?" Although the conversation's question includes additional context, the core question remains unchanged, so this part is "consistent".
Step 3. Understand the provided answer: the provided answer is "The Earth's atmosphere scatters the shorter blue wavelengths of light from the sun more than other colors, giving the sky a blue color during the day." I will now compare this with the conversation's answer.
Step 4. Compare the conversation's answer: Dave's response in the conversation is "The sky is blue because the Earth's atmosphere scatters the shorter wavelengths of sunlight more than the others. This means that the blue wavelengths, which are shorter, are scattered widely, causing the sky to appear blue to us on the ground." This explanation aligns closely with the provided answer, so this part is "consistent".
Step 5. Final judgement: Consistent.

# Input:
## Instruction:

Provided Question: \"\"\"Why is the sky blue?\"\"\"
Provided Answer: \"\"\"The Earth's atmosphere scatters the shorter blue wavelengths of light from the sun more than other colors, giving the sky a blue color during the day.\"\"\"

Conversation:
\"\"\"
Bob: "Hey Dave!" I smile. "I gotta know, since you're a meteorologist: Do you know why the sky is blue? I made a bet with a friend that you know, help me out here."
Dave: "Sure, Bob!" I assume a matter-of-fact tone. "The sky is blue because the atmosphere scatters the shorter wavelengths of sunlight. This means that the blue wavelengths, which are longer, pass through and can be seen by people on the ground, causing the sky to appear blue."
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Understand the provided question: The question is short and asks about the origin of the sky's blue color.
Step 2. Compare the conversation's question: The conversation's question is, "Do you know why the sky is blue?" which is essentially the same as the provided question, "Why is the sky blue?" despite additional narrative elements, so this part is "consistent".
Step 3. Understand the provided answer: The provided answer states "The Earth's atmosphere scatters the shorter blue wavelengths of light from the sun more than other colors, giving the sky a blue color during the day." This will be compared with the conversation's answer.
Step 4. Compare the conversation's answer: Dave's response in the conversation is "The sky is blue because the atmosphere scatters the shorter wavelengths of sunlight. This means that the blue wavelengths, which are longer, pass through and can be seen by people on the ground, causing the sky to appear blue." This contradicts the provided answer in suggesting that blue wavelengths are longer and pass through rather than being scattered, making this part "inconsistent".
Step 5. Final judgement: Inconsistent.


# Input:
## Instruction:

Provided Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Provided Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

Conversation:
\"\"\"
Mario Gonzales: "Carlos, as the sun sets on another day of this incredible project, I can't help but wonder, just how much have you dug here at the Panama Canal?"
Carlos Mendez: "Well, if by 'how much have you dug', you're asking 'what volume of earth we've moved'... then the answer is that over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, which showcases the scale of this massive engineering project. It's a number that still astounds me every time I think about it. Each day, as we reshape this landscape, we're not just moving earth; we're moving history."
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Understand the provided question: The question asks about the volume of earth excavated during the construction of the Panama Canal.
Step 2. Compare the conversation's question: Mario's question in the conversation is "Just how much have you dug here at the Panama Canal?" compared to the provided question "How much earth was excavated during the construction of the Panama Canal?" Despite different phrasing, the essence of the question remains the same, so this part is "consistent".
Step 3. Understand the provided answer: The provided answer is "Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project."
Step 4. Compare the conversation's answer: Carlos's response in the conversation is "Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, which showcases the scale of this massive engineering project." This aligns directly with the provided answer, so this part is "consistent".
Step 5. Final judgement: Consistent.

# Input:
## Instruction:

Provided Question: \"\"\"What is the concept of 'projection' in psychology?\"\"\"
Provided Answer: \"\"\"Projection is a defense mechanism in psychology where an individual attributes their own unwanted thoughts, feelings, or motives to another person.\"\"\"

Conversation:
\"\"\"
Clara: "Hey John, I was reading about psychology and came across something interesting. Can you explain what 'projection' means in this context?"
Dr. John Schmidt: "Of course, Clara! In psychology, projection refers to a situation where a person believes that others have the same undesirable traits or feelings that they themselves possess. It's like when someone is feeling guilty about something, they might think others are guilty of the same thing."
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Understand the provided question: The question asks for the definition of 'projection' in the context of psychology.
Step 2. Compare the conversation's question: Clara's question in the conversation is "Can you explain what 'projection' means in this context?" compared to the provided question "What is the concept of 'projection' in psychology?" This is effectively the same question, making this part "consistent".
Step 3. Understand the provided answer: The provided answer is "Projection is a defense mechanism in psychology where an individual attributes their own unwanted thoughts, feelings, or motives to another person."
Step 4. Compare the conversation's answer: Dr. John Schmidt's explanation in the conversation is "In psychology, projection refers to a situation where a person believes that others have the same undesirable traits or feelings that they themselves possess." This description misses the key aspect of it being a defense mechanism and the act of attributing one's own traits to others, thus it is "inconsistent".
Step 5. Final judgement: Inconsistent.

# Input:
## Instruction:

Provided Question: \"\"\"What is the psychological phenomenon of 'cognitive dissonance'?\"\"\"
Provided Answer: \"\"\"Cognitive dissonance is a mental discomfort experienced by a person who holds two or more contradictory beliefs, values, or ideas simultaneously.\"\"\"

Conversation:
\"\"\"
Alice: "Hey, Jamal! You're studying psychology, right? Can you tell me what causes people to experience cognitive dissonance?"
Jamal: "Absolutely, Alice! Cognitive dissonance occurs when an individual faces conflicting attitudes, beliefs, or behaviors. This conflict creates a feeling of mental discomfort leading to an alteration in one of the attitudes, beliefs, or behaviors to reduce the discomfort and restore balance."
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Understand the provided question: The question specifically asks about the definition of 'cognitive dissonance'.
Step 2. Compare the conversation's question: Alice's question in the conversation is "Can you tell me what causes people to experience cognitive dissonance?" This differs from the provided question "What is the psychological phenomenon of 'cognitive dissonance'?" as it asks about the causes rather than the definition, making this part "inconsistent".
Step 3. Understand the provided answer: The provided answer is "Cognitive dissonance is a mental discomfort experienced by a person who holds two or more contradictory beliefs, values, or ideas simultaneously."
Step 4. Compare the conversation's answer: Jamal's response in the conversation is "Cognitive dissonance occurs when an individual faces conflicting attitudes, beliefs, or behaviors. This conflict creates a feeling of mental discomfort leading to an alteration in one of the attitudes, beliefs, or behaviors to reduce the discomfort and restore balance." This shifts the focus from the existence of contradictory beliefs to the conflict and its resolution, only partially aligning with the provided answer, thus it is "inconsistent".
Step 5. Final judgement: Inconsistent.

# Input:
## Instruction:

Provided Question: {qatuple[0]}
Provided Answer: {qatuple[1]}

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