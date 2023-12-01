import re
from .answer_relevant_grammar import answer_relevant_grammar
from .constants import LOGICAL_MODEL
from llama_cpp import Llama

# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed
def check_answer_relevancy_with_text(qatuple,logic_llm):
    retries = 0
    while (retries <= 4):
        decision_prompt = f"""# Input:
You are an expert educational AI. Given a paragraph or two from a larger text, a question based on the paragraphs, and an answer to the question, you will make a determination as to whether the answer only uses the information in the paragraphs for its main points. Essentially: you will check if the answer is constrained to the information in the paragraphs provided. Your task includes first analyzing the answer, thinking through whether or not the answer reflects aspects of the paragraphs provided. 

Following this, at the very end of your response, you will write "Relevant" or "Irrelevant" depending on your analysis of the answer with regards to the text. 

# Input:
## Instruction:

Text: 
\"\"\"
Polar bears are primarily found in the Arctic Circle, where they have adapted to the harsh cold environment. Their diet is primarily based on marine mammals, with seals being the most crucial part of their diet. Polar bears are skilled hunters, often waiting by seals' breathing holes to catch their prey. Occasionally, they also eat birds, bird eggs, and small mammals. They also consume significant amounts of berries and fish. These bears are known for their ability to survive in extreme conditions, thanks to their thick fur and fat layers which provide insulation.
\"\"\"

Question (based on text): \"\"\"What are the primary food sources of polar bears?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Polar bears primarily eat seals, but they also consume significant amounts of berries and fish. Their efficient digestion system, which has been studied through MRIs, allows them to process a wide variety of foods.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: Focus on the main points stated in the text about the diet of polar bears.
Step 2. Understand the Answer: The answer describes the diet of polar bears, which includes seals, berries, and fish, as well as their efficient digestion system which has been studied with MRIs. Key terms: Polar bears, seals, berries, fish, digestion, MRIs.
Step 3. Summarize Relevant Parts of the Text: Since the question's primary focus is on the diet of polar bears, I will extract some key elements of the test related to this subject for reference. The text says about this topic, "Their diet is primarily based on marine mammals, with seals being the most crucial part of their diet." It also says "They also consume significant amounts of berries and fish."
Step 4. Compare the First Part of the Answer with the Text: Check if the text supports the claim that polar bears primarily eat seals. It states "Their diet is primarily based on marine mammals, with seals being the most crucial part of their diet", so this part is relevant. Then, check if the text mentions polar bears consuming significant amounts of berries and fish. The text says "They also consume significant amounts of berries and fish", so this part aligns with the text and is relevant. Then, check if the text mentions polar bears having an efficient digestion system. It does not, so this part is adding extra information not found in the text and is thus irrelevant. The text also does not mention MRIs, or "how" we know this fact, anywhere at all, so that information is also irrelevant.
Step 5. Final Judgement: The answer mentions some facts present in the original text, but since the answer also mentions facts not present in the original text, it is irrelevant.

# Input:
## Instruction:

Text: 
\"\"\"
The Pythagorean theorem is a fundamental principle in geometry, stating that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. This theorem is often expressed as the equation a² + b² = c², where a and b are the lengths of the two sides that form the right angle, and c is the length of the hypotenuse. Historically, this theorem has been attributed to the ancient Greek mathematician Pythagoras, although evidence suggests it was known to mathematicians in several cultures before him.
\"\"\"

Question (based on text): \"\"\"What does the Pythagorean theorem state in the context of geometry?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"The Pythagorean theorem, crucial in geometry, states that in a right-angled triangle, a² + b² = c², where a and b are the perpendicular sides and c is the hypotenuse. Additionally, it is noteworthy that this theorem was utilized in ancient Egypt for land surveying purposes.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: The text discusses the Pythagorean theorem, its formula, and its historical attribution.
Step 2. Understand the Answer: The answer explains the Pythagorean theorem, mentions its formula, and introduces an additional point about its use in ancient Egypt. Key terms: Pythagorean theorem, geometry, a² + b² = c², perpendicular sides, hypotenuse, ancient Egypt, land surveying.
Step 3. Summarize Relevant Parts of the Text: The text explicitly states, "The Pythagorean theorem is a fundamental principle in geometry, stating that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides." It further elaborates, "This theorem is often expressed as the equation a² + b² = c²."
Step 4. Compare Each Part of the Answer with the Text: The answer's first part that states the theorem and its formula matches with the text's statement, "The Pythagorean theorem is a fundamental principle in geometry," and "a² + b² = c²." Therefore, these parts are relevant. The claim about the theorem's use in ancient Egypt for land surveying is not supported by any information in the provided text. This part of the answer introduces external information and is therefore irrelevant.
Step 5. Final Judgement: While the answer correctly reflects the theorem's description and formula as stated in the text, the addition of historical usage in ancient Egypt is not found in the text, making the answer partially irrelevant.

# Input:
## Instruction:

Text: 
\"\"\"
Sigmund Freud, an Austrian neurologist and the father of psychoanalysis, introduced the concept of the unconscious mind. He suggested that the unconscious mind stores feelings, thoughts, and desires that are too threatening or painful for conscious awareness. Freud believed that these repressed elements could influence behavior and personality. One of his key methods for exploring the unconscious mind was through dream analysis, where he interpreted the meaning of dreams as a pathway to understanding the hidden desires and thoughts of the individual. Freud also developed the theory of the Oedipus complex, which proposes that during the phallic stage of psychosexual development, a male child experiences unconscious sexual desires for his mother and hostility toward his father.
\"\"\"

Question (based on text): \"\"\"What methods did Freud use to explore the unconscious mind, and what are some key concepts he introduced in this area?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Freud used dream analysis as a method to explore the unconscious mind. He also introduced the concept of the Oedipus complex, suggesting that during a specific stage of development, a child experiences unconscious desires.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: The text discusses Freud's contributions to the understanding of the unconscious mind, including his methods and key concepts.
Step 2. Understand the Answer: The answer refers to Freud's use of dream analysis and the concept of the Oedipus complex. Key terms: Freud, dream analysis, unconscious mind, Oedipus complex.
Step 3. Summarize Relevant Parts of the Text: The text mentions, "One of his key methods for exploring the unconscious mind was through dream analysis," and "Freud also developed the theory of the Oedipus complex."
Step 4. Compare Each Part of the Answer with the Text: Dream Analysis: The text clearly states, "One of his key methods for exploring the unconscious mind was through dream analysis." This directly matches the answer's claims. The text describes, "Freud also developed the theory of the Oedipus complex," matching the answer's mention of Freud's introduction of the Oedipus complex. 
Step 5. Final Judgement: Final Judgement: The answer accurately reflects the information provided in the text. Each part of the answer can be traced back to specific points in the text, as shown by the direct quotes. Therefore, the answer is relevant.

# Input:
## Instruction:

Text: 
\"\"\"
The planet Venus has a very dense atmosphere composed mainly of carbon dioxide, with clouds of sulfuric acid. This composition creates a runaway greenhouse effect, leading to surface temperatures hot enough to melt lead. Venus rotates very slowly on its axis, taking about 243 Earth days to complete one rotation, which is longer than its orbital period around the Sun. Interestingly, Venus rotates in the opposite direction to most planets in the solar system. This retrograde rotation is thought to be the result of a collision with a large celestial body early in its history.
\"\"\"

Question (based on text): \"\"\"What are the main characteristics of Venus's atmosphere and rotation?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Venus's atmosphere is dense, primarily composed of carbon dioxide, and contains clouds of sulfuric acid, leading to extremely high temperatures. The planet has a unique rotation, taking 243 Earth days to rotate once and rotating in a retrograde direction due to gravitational interactions with Earth and other planets.\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Analyze the Text: The text provides specific details about Venus's atmosphere and its rotation.
Step 2. Understand the Answer: The answer discusses Venus's dense atmosphere, composition, high temperatures, unique rotation, and the cause of its retrograde rotation. Key terms: Venus, atmosphere, carbon dioxide, sulfuric acid, temperature, rotation, retrograde, gravitational interactions, Earth.
Step 3. Summarize Relevant Parts of the Text: The text states, "The planet Venus has a very dense atmosphere composed mainly of carbon dioxide, with clouds of sulfuric acid," and "This composition creates a runaway greenhouse effect, leading to surface temperatures hot enough to melt lead." Regarding rotation, it mentions, "Venus rotates very slowly on its axis, taking about 243 Earth days to complete one rotation," and "Venus rotates in the opposite direction to most planets in the solar system."
Step 4. Compare the Answer with the Text: The answer's claim that Venus's atmosphere is dense and primarily composed of carbon dioxide with sulfuric acid clouds aligns with the text, as does the mention of high temperatures. For the rotation, the text confirms Venus takes 243 Earth days to rotate and rotates in a retrograde direction. However, the answer's mention of gravitational interactions with Earth and other planets as the cause of retrograde rotation is not supported by the text, which attributes it to a past collision.
Step 5. Final Judgement: The answer aligns with the text regarding Venus's atmospheric composition, temperature, and rotation period and direction. However, it introduces an unsupported reason for Venus's retrograde rotation. Since a significant part of the answer introduces information not present in the text, it is irrelevant.

# Input:
## Instruction:

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"{qatuple[1]}\"\"\"

# Response:
## Reasoning and thought process (I will check all parts of the answer for relevancy):
"""
        # print("DEBUG\n\n" + decision_prompt)
        try:
            completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>"], grammar=answer_relevant_grammar, echo=True,temperature=0.2)["choices"][0]["text"]

            # print("DEBUG\n\n")
            completion_pattern = re.compile(r"Reasoning and thought process \(I will check all parts of the answer for relevancy\):\n(.+)", re.DOTALL | re.IGNORECASE)
            judgement_pattern = re.compile(r"Final Judgement:(.+)", re.DOTALL | re.IGNORECASE)
            response = completion_pattern.search(completion).group(1).strip()
            print(response)
            determination = judgement_pattern.search(response).group(1).strip()
            print("\n\nDETERMINATION:\n------")
            print(determination)
            print("\n---------\n")
            if "irrelevant" in determination or "Irrelevant" in determination:
                return (False,response)
            elif "relevant" in determination or "Relevant" in determination:
                return (True,response)
            else:
                retries += 1
        except:
            retries += 1
    
    # TODO error handling
            
if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    q_test = [(') Explain how our understanding of planetary motion has changed over time.',
  'The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 (') Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 (') Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 (") Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.')]
    
    
    print("Begin HGWELLS test")
    result = check_answer_relevancy_with_text(q_test[2],logic_llm)
    ## TODO a wider variety of tests from different texts
    if not result[0]: # if not relevant
        print("Made right choice for bad answer")
    else:
        print("Made wrong choice for bad answer")
        
    d2 = check_answer_relevancy_with_text(q_test[1],logic_llm)
    if d2[0]:
        print("Made wrong choice for bad answer")
    else:
        print("Made right choice for bad answer") # It caught something I missed!!!!