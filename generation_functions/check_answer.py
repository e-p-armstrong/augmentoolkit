import re
from .answer_accurate_grammar import answer_accurate_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed


# TODO improve the step-by-step check to formalize the fact that we evaluate each part of the answer. So "accurate" or "inaccurate" at the end of each step.
# Also improve the test, right now the "good" answer is actually a bit shit
# ^ one big unfixed problem here is that the model might occasionally say "This text is not neccessarily inaccurate" and that usage of the word might make the determination inaccurate. To fix this a reliance on the final judgement can be used, or I can prompt it to only use "inaccurate" if it is describing a part of the answer as bad, or through more precise grammars for the step-by-step bits.

def check_answer(qatuple,logic_llm, permissive_mode=True):
    retries = 0
    while (retries <= 4):
        decision_prompt = f"""# Input:
You are an expert educational AI. Given a paragraph or two from a larger text, a question based on the paragraphs, and an answer to the question, you will make a determination as to whether the answer to the question is a sensible answer, given the information in the paragraphs. Essentially: you will fact-check the answer to the question, with your source of truth being the paragraphs provided. Your task includes first analyzing the text, thinking through whether or not the answer reflects aspects of the paragraphs provided. 

Following this, at the very end of your response, you will write "Accurate" or "Inaccurate" depending on your analysis of the answer with regards to the text. 

Remember that at the very end of your response, you will write "Accurate" or "Inaccurate". Do not use these words anywhere else in your answer.

# Input:
## Instruction:
Text: 
\"\"\"
The Industrial Revolution marked a transformative period in history, fundamentally altering economic structures and industrial processes. One of the most significant innovations was the advent of steam power, which drastically reduced production costs. This reduction was due to the increased efficiency and speed of machines powered by steam, replacing the slower, more labor-intensive methods previously used. Additionally, steam power was instrumental in the development of semi-automated factories, leading to a significant shift in manufacturing processes and labor dynamics.
\"\"\"

Question (based on text): \"\"\"What was the role of steam power in the Industrial Revolution, particularly in terms of its impact on production costs, and the emergence of semi-automated factories?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Steam power during the Industrial Revolution played a crucial role in decreasing production costs. However, it had no significant impact on the emergence of semi-automated factories. Interestingly, it also led to an increased cultivation of lemons.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: focus on the details provided about the economic impacts of the Industrial Revolution.
Step 2. Understand the Question: the question's focus is on the role of steam power in the Industrial Revolution.
Step 3. Compare the First Part of the Answer with the Text: check if the text supports the claim that steam power decreased production costs. It does, so this part is accurate. Then, check if the text supports the claim that steam power had no impact on the emergence of semi-automated factories. The text contradicts this, and claims that steam power directly led to the creation of semi-automated factories, so this part is inaccurate.  Then, check if the text supports the claim that steam power led to increased cultivation of lemons. The text doesn't mention lemons at all, so this part is inaccurate.
Step 4. Final Judgment: Since the answer is not entirely accurate, the answer is inaccurate.

# Input:
## Instruction:
Text: 
\"\"\"
Epistemology, often regarded as a branch of philosophy, is concerned with the theory of knowledge. It involves understanding the nature, sources, and limitations of knowledge. A key concept in epistemology is the idea of "justified true belief," which suggests that for someone to know something, they must believe it to be true, have justification for this belief, and the belief must indeed be true. For example, believing that it will rain tomorrow because the weather forecast says so, and it turns out to be true, is a justified true belief. Epistemology also explores the idea of skepticism, which questions the possibility of certain or absolute knowledge.
\"\"\"

Question (based on text): \"\"\"What does the concept of 'justified true belief' involve in epistemology, and how does skepticism relate to the understanding of knowledge?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"The concept of 'justified true belief' in epistemology involves having a belief that is true and has justification. For a belief to be considered knowledge, it must be true, one must believe it, and there must be sufficient reason for this belief. Skepticism plays a role by challenging the certainty of knowledge, thereby emphasizing the need for strong justification in 'justified true belief.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: The primary focus is on epistemology, particularly the 'justified true belief' and skepticism.
Step 2. Understand the Answer: The answer addresses 'justified true belief' and skepticism within the context of epistemology. Key terms: justified true belief, skepticism, true, belief, justification, certainty of knowledge.
Step 3. Compare the First Part of the Answer with the Text: The text defines 'justified true belief' as a belief that is true, believed to be true, and has justification. The answer aligns with this by stating, "For a belief to be considered knowledge, it must be true, one must believe it, and there must be sufficient reason for this belief." Direct quote from the text: "for someone to know something, they must believe it to be true, have justification for this belief, and the belief must indeed be true."
Step 4. Compare the Second Part of the Answer with the Text: The text discusses skepticism as questioning the possibility of certain or absolute knowledge. The answer correctly relates this to 'justified true belief' by stating that skepticism emphasizes the need for strong justification. Direct quote from the text: "skepticism, which questions the possibility of certain or absolute knowledge."
Step 5. Final Judgment: Since the answer accurately reflects the text, the answer is accurate.

# Input:
## Instruction:
Text: 
\"\"\"
Formal logic, a branch of philosophy and mathematics, is concerned with the study of reasoning. It uses a set of symbols and rules to create a language that can precisely express ideas. One key aspect of formal logic is the concept of a valid argument, which is an argument where if the premises are true, the conclusion must be true. For instance, in the statement 'All humans are mortal; Socrates is a human; therefore, Socrates is mortal,' the conclusion follows logically from the premises. Another important element is the use of symbolic representation to simplify and clarify arguments. This allows for complex ideas to be broken down into simpler components, making them easier to analyze and understand.
\"\"\"

Question (based on text): \"\"\"What are the key aspects of formal logic, and how does symbolic representation contribute to its goals?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Key aspects of formal logic include the study of valid arguments and the use of symbolic representation. Valid arguments are those where the premises may or may not lead to a true conclusion. Symbolic representation helps in making complex ideas more understandable by breaking them down into simpler forms.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: The focus is on formal logic, specifically on valid arguments and symbolic representation.
Step 2. Understand the Answer: The answer discusses valid arguments and symbolic representation in formal logic. Key terms: valid arguments, symbolic representation, premises, true conclusion, complex ideas, simpler forms.
Step 3. Compare the First Part of the Answer with the Text: The text states, "a valid argument is an argument where if the premises are true, the conclusion must be true." The answer slightly mischaracterizes this by saying, "Valid arguments are those where the premises may or may not lead to a true conclusion." This is a subtle but significant deviation from the text. Direct quote from the text: "if the premises are true, the conclusion must be true." I will now compare the Second Part of the Answer with the Text: The answer correctly states that symbolic representation helps in making complex ideas more understandable by breaking them down into simpler forms. Direct quote from the text: "This allows for complex ideas to be broken down into simpler components."
Step 4. Final Judgment: Since the answer is not entirely accurate, the answer is inaccurate.

# Input:
## Instruction:
Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"{qatuple[1]}\"\"\"

# Response:
## Reasoning and thought process (I will be careful to check all of the answer against the text):
"""
        try:
            # print("DEBUG\n\n" + decision_prompt)
            completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>"], echo=True,grammar=answer_accurate_grammar,temperature=0.2)["choices"][0]["text"]

            # print("DEBUG\n\n")
            completion_pattern = re.compile(r"Reasoning and thought process \(I will be careful to check all of the answer against the text\):\n(.+)", re.DOTALL)
            response = completion_pattern.search(completion).group(1).strip()
            print(response)
            if permissive_mode:
                determination_pattern = re.compile(r"Final Judgement:(.+)", re.DOTALL)    
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response
            print("\n\nDETERMINATION:\n------")
            print(determination)
            print("\n---------\n")
            if "inaccurate" in determination or "Inaccurate" in determination or "mostly" in determination: # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
                return (False,response)
            elif "accurate" in determination or "Accurate" in determination: # very deliberate placement of accurate here, becaues the model can sometimes say irrelevant at the very end, even after saying accurate in its judgement
                return (True,response)
            elif "irrelevant" in determination or "Irrelevant" in determination: # optional support for checking relevance here, too.
                return (None,response) # signal that question is irrelevant
            else:
                Exception("Broke!")
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
    
    inaccurate_qa_tuple = ("For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?", "The concept of a spherical Earth has been known for only about 1,000 years.", "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.")
    
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
    
        
    # Fixed answer:
    accurate_qa_tuple = ("For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?", "It is likely that the concept of a spherical Earth was understood by at least a limited number of intelligent people around the time period mentioned in the text when it became commonly taught in schools and universities, which would put this knowledge after about 2500 years ago.", "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.")
    d3 = check_answer(accurate_qa_tuple,logic_llm)
    if True == d3[0]:
        print("Made wrong choice for bad question") # Passes currently
    else:
        print("Made wrong choice for good question", d3[0])