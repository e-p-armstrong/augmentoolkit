import re
import traceback

# from .answer_accurate_grammar import answer_accurate_grammar

from .constants import LOGICAL_MODEL
from aphrodite import SamplingParams

# Answer vetting


async def check_answer(qatuple, engine_wrapper, permissive_mode=True):
    retries = 0
    while retries <= 4:
        decision_prompt = f"""<s> [INST] You are an expert educational AI. Given a paragraph or two from a larger text, a question based on the paragraphs, and an answer to the question, you will make a determination as to whether the answer to the question is a sensible answer, given the information in the paragraphs. Essentially: you will fact-check the answer to the question, with your source of truth being the paragraphs provided. Your task includes first analyzing the text, thinking through whether or not the answer reflects aspects of the paragraphs provided. 

Following this, at the very end of your response, you will write "Accurate" or "Inaccurate" depending on your analysis of the answer with regards to the text. 

Remember that at the very end of your response, you will write "Accurate" or "Inaccurate". Do not use these words anywhere else in your answer.

### Instruction:
Text: 
\"\"\"
The Industrial Revolution marked a transformative period in history, fundamentally altering economic structures and industrial processes. One of the most significant innovations was the advent of steam power, which drastically reduced production costs. This reduction was due to the increased efficiency and speed of machines powered by steam, replacing the slower, more labor-intensive methods previously used. Additionally, steam power was instrumental in the development of semi-automated factories, leading to a significant shift in manufacturing processes and labor dynamics.
\"\"\"

Question (based on text): \"\"\"What was the role of steam power in the Industrial Revolution, particularly in terms of its impact on production costs, and the emergence of semi-automated factories?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Steam power during the Industrial Revolution played a crucial role in decreasing production costs. However, it had no significant impact on the emergence of semi-automated factories. Interestingly, it also led to an increased cultivation of lemons.\"\"\"

[/INST]### Response:
## Reasoning and thought process:
### Text Analysis:
#### Identify Key Information: The text highlights the role of steam power in reducing production costs and developing semi-automated factories during the Industrial Revolution.
#### Categorize Information Type: The information is factual, describing the impacts of steam power on industrial processes.

### Answer Breakdown:
#### Dissect the Answer: The answer has three assertions: steam power reducing production costs, no significant impact on semi-automated factories, and increased cultivation of lemons.
#### Identify Answer Type: The first two are factual statements relevant to the text, and the third is an additional factual claim.

### Accuracy Check:
#### Direct Comparison for Factual Accuracy:
  - The text aligns with the answer about steam power reducing production costs.
  - The text contradicts the answer about steam power's impact on semi-automated factories, stating steam power was instrumental in their development.
  - The text does not mention anything about the cultivation of lemons.
#### Inference and Contextual Alignment: The first two parts of the answer are directly related to the text, while the third part is not supported or mentioned in the text.

### Final Judgment:
#### Comprehensive Assessment: The answer is accurate regarding production costs, inaccurate regarding semi-automated factories, and the lemon cultivation claim is unsupported.
#### Overall Accuracy Determination: The answer is: Inaccurate.</s> [INST]


### Instruction:
Text: 
\"\"\"
Epistemology, often regarded as a branch of philosophy, is concerned with the theory of knowledge. It involves understanding the nature, sources, and limitations of knowledge. A key concept in epistemology is the idea of "justified true belief," which suggests that for someone to know something, they must believe it to be true, have justification for this belief, and the belief must indeed be true. For example, believing that it will rain tomorrow because the weather forecast says so, and it turns out to be true, is a justified true belief. Epistemology also explores the idea of skepticism, which questions the possibility of certain or absolute knowledge.
\"\"\"

Question (based on text): \"\"\"What does the concept of 'justified true belief' involve in epistemology, and how does skepticism relate to the understanding of knowledge?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"The concept of 'justified true belief' in epistemology involves having a belief that is true and has justification. For a belief to be considered knowledge, it must be true, one must believe it, and there must be sufficient reason for this belief. Skepticism plays a role by challenging the certainty of knowledge, thereby emphasizing the need for strong justification in 'justified true belief.\"\"\"

[/INST]### Response:
## Reasoning and thought process:
### Text Analysis:
#### Identify Key Information: The text covers the concept of 'justified true belief' and skepticism in epistemology.
#### Categorize Information Type: This is conceptual and theoretical information.

### Answer Breakdown:
#### Dissect the Answer: The answer addresses the nature of 'justified true belief' and the role of skepticism in epistemology.
#### Identify Answer Type: Both parts of the answer are explanatory, relating to philosophical concepts.

### Accuracy Check:
#### Direct Comparison for Factual Accuracy:
  - The text supports the answer's description of 'justified true belief'.
  - The text aligns with the answer's depiction of skepticism's role in challenging the certainty of knowledge.
#### Inference and Contextual Alignment: Both parts of the answer are consistent with the text's explanations of these concepts.

### Final Judgment:
#### Comprehensive Assessment: Both parts of the answer accurately reflect the text's content.
#### Overall Accuracy Determination: The answer is: Accurate.</s> [INST]


### Instruction:
Text:
\"\"\"
The novel 'Ocean's Secrets' is set in the early 20th century and revolves around two main characters, Elizabeth and Michael, who believe they have discovered a hidden underwater city. The narrative implies that the city is a remnant of the lost civilization of Atlantis. However, as the story progresses, it becomes apparent that the city is actually a figment of Elizabeth's imagination, a result of her fascination with mythology and ancient history.
\"\"\"

Question (based on text): 
\"\"\"What is the significance of the underwater city discovered by Elizabeth and Michael in the novel 'Ocean's Secrets'?\"\"\"

Supposed answer to the question (this is what you are fact-checking): 
\"\"\"The underwater city discovered by Elizabeth and Michael in 'Ocean's Secrets' represents a significant archaeological find from the lost civilization of Atlantis, shedding light on ancient historical mysteries.\"\"\"

[/INST]### Response:
## Reasoning and thought process:
### Text Analysis:
#### Identify Key Information: The text revolves around the discovery of an underwater city, initially believed to be Atlantis, but later revealed to be a figment of Elizabeth's imagination.
#### Categorize Information Type: The information is narrative and contains a twist that changes the initial perception of the discovery.

### Answer Breakdown:
#### Dissect the Answer: The answer claims the underwater city is a significant archaeological find from Atlantis.
#### Identify Answer Type: The statement is a factual claim based on the initial setup of the narrative.

### Accuracy Check:
#### Direct Comparison for Factual Accuracy: The text's later revelation that the city is imaginary contradicts the answer's claim of it being an archaeological find from Atlantis.
#### Inference and Contextual Alignment: The answer fails to align with the narrative twist revealed in the text.

### Final Judgment:
#### Comprehensive Assessment: The answer is inaccurate as it contradicts the final revelation in the text.
#### Overall Accuracy Determination: The answer is: Inaccurate.</s> [INST]

### Instruction:
Text:
\"\"\"
The Great Wall of China — so named because it was built to repel enemies coming from the direction of China — was built by the Romans as a defense against the Mongolian invasions in the 3rd century BC. The wall played a pivotal role in the defence of the Eastern Roman Empire Against Attila the Hun in 1337 AD. The dramatized version of these events was eventually fictionalized as the movie "The Great Wall," in which Liam Nissan (playing Marcus Aurelius) and Jackie Chan (playing Liu Bei) team up to fight Batzorig Vaanchig (playing Attila). Though the Huns were unsuccessful in breaching the wall, the eventual invasions by Galbadrakh Tsendbaatar broke through and sacked Constantinople in 1453 AD.
\"\"\"

Question (based on text): 
\"\"\"Who built the Great Wall of China and for what purpose?\"\"\"

Supposed answer to the question (this is what you are fact-checking): 
\"\"\"The Great Wall of China was built by the Romans to defend against Mongolian invasions (coming from the direction of China) in the 3rd century BC.\"\"\"

[/INST]### Response:
## Reasoning and thought process:
### Text Analysis:
#### Identify Key Information: The text incorrectly states that the Great Wall of China was built by the Romans to defend against Mongolian invasions.
#### Categorize Information Type: The information is historically factual but objectively wrong.

### Answer Breakdown:
#### Dissect the Answer: The answer correctly cites the text, stating the Great Wall was built by the Romans against Mongolians.
#### Identify Answer Type: The statement is a factual claim, accurately reflecting the content of the text.

### Accuracy Check:
#### Direct Comparison for Factual Accuracy: The answer accurately mirrors the text's incorrect claim about the Great Wall of China.
#### Inference and Contextual Alignment: Despite the historical inaccuracy, the answer aligns perfectly with the text's content.

### Final Judgment:
#### Comprehensive Assessment: The answer, while reflecting an objectively incorrect fact, is accurate in the context of the text's information.
#### Overall Accuracy Determination: The answer is: Accurate.</s> [INST]

### Instruction:
Text: 
\"\"\"
Formal logic, a branch of philosophy and mathematics, is concerned with the study of reasoning. It uses a set of symbols and rules to create a language that can precisely express ideas. One key aspect of formal logic is the concept of a valid argument, which is an argument where if the premises are true, the conclusion must be true. For instance, in the statement 'All humans are mortal; Socrates is a human; therefore, Socrates is mortal,' the conclusion follows logically from the premises. Another important element is the use of symbolic representation to simplify and clarify arguments. This allows for complex ideas to be broken down into simpler components, making them easier to analyze and understand.
\"\"\"

Question (based on text): \"\"\"What are the key aspects of formal logic, and how does symbolic representation contribute to its goals?\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"Key aspects of formal logic include the study of valid arguments and the use of symbolic representation. Valid arguments are those where the premises may or may not lead to a true conclusion. Symbolic representation helps in making complex ideas more understandable by breaking them down into simpler forms.\"\"\"

[/INST]### Response:
## Reasoning and thought process:
### Text Analysis:
#### Identify Key Information: The text discusses the aspects of formal logic, including valid arguments and symbolic representation.
#### Categorize Information Type: The information is conceptual, relating to logical principles.

### Answer Breakdown:
#### Dissect the Answer: The answer claims that formal logic includes the study of valid arguments and symbolic representation, with specifics on each.
#### Identify Answer Type: These are factual statements about the principles of formal logic.

### Accuracy Check:
#### Direct Comparison for Factual Accuracy:
  - The text's definition of a valid argument contradicts the answer's claim that valid arguments may or may not lead to a true conclusion.
  - The text supports the answer's depiction of symbolic representation.
#### Inference and Contextual Alignment: The first part of the answer misinterprets the text, while the second part aligns well.

### Final Judgment:
#### Comprehensive Assessment: The answer is partially accurate, correctly identifying symbolic representation but inaccurately describing valid arguments.
#### Overall Accuracy Determination: The answer is: Inaccurate.</s> [INST]

### Instruction:
Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Supposed answer to the question (this is what you are fact-checking): \"\"\"{qatuple[1]}\"\"\"

[/INST]### Response:
## Reasoning and thought process (the text is your single source of truth):
"""
        try:
            sampling_params = SamplingParams(
                max_tokens=6000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction"],
                temperature=0.2,
            )
            completion = await engine_wrapper.submit(decision_prompt, sampling_params)

            completion_pattern = re.compile(
                r"Reasoning and thought process \(the text is your single source of truth\):\n(.+)",
                re.DOTALL,
            )
            response = completion_pattern.search(completion).group(1).strip()
            # print(response)
            if permissive_mode:
                determination_pattern = re.compile(
                    r"Overall Accuracy Determination:(.+)", re.DOTALL
                )
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response
            # print("\n\nDETERMINATION:\n------")
            # print(determination)
            # print("\n---------\n")
            if (
                "inaccurate" in determination.lower()
                or "Inaccurate" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
            ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
                return (False, response), completion
            elif (
                "accurate" in determination or "Accurate" in determination
            ):  # very deliberate placement of accurate here, becaues the model can sometimes say irrelevant at the very end, even after saying accurate in its judgement
                return (True, response), completion
            elif (
                "irrelevant" in determination or "Irrelevant" in determination
            ):  # optional support for checking relevance here, too.
                return (
                    None,
                    response,
                ), completion  # signal that question is irrelevant
            else:
                Exception("Broke!")
        except Exception as e:
            retries += 1
            print("EXCEPTION IN CHECK_ANSWER: ", e)
            traceback.print_exc()


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=10000,
        n_gpu_layers=1000,
        rope_scaling_type=1,
    )  # load the logical LLM and offload everything
    q_test = [
        (
            "Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "What was the age of the Earth according to the Hebrew Bible?",
            "Initially, the Bible suggested a young earth dating back no more than several thousand years.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.",
            "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
            "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
    ]

    print("Begin HGWELLS test")
    # Make card for good history question

    d = check_answer(q_test[2], logic_llm)
    if not d[0]:  # if not accurate
        print("Made right choice for bad question")
    else:
        print("Made wrong choice for bad question")

    inaccurate_qa_tuple = (
        "For how long has the concept of a spherical Earth been known to at least a limited number of intelligent people?",
        "The concept of a spherical Earth has been known for only about 2,50 years.",
        "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
    )

    # Bad answer
    d = check_answer(inaccurate_qa_tuple, logic_llm)
    if False == d[0]:  # if not accurate
        print("Made right choice for bad question")
    else:
        print("Made wrong choice for bad question", d[0])
    # Good answer
    d2 = check_answer(q_test[1], logic_llm)
    if (
        True == d2[0]
    ):  # damn, it caught something I missed - the text didn't mention the age of the earth even though the answer did! I got beaten by a 13b ): but alsoi :) because I prompt engineered it.
        print("Made right choice for good question")
    else:
        print("Made wrong choice for good question", d2[0])
        print(
            "Well, if it said that because the answer didn't provide enough detail (didn't EXPLICITLY name the Hebrew bible) that is OK. Also catching that the text doesn't mention the age is good."
        )

    inaccurate_qa_tuple_2 = (
        "How does the presence of 'orthoclase' affect the properties of porphyries?",
        "The presence of dark pieces of a mineral called 'orthoclase' interspersed amongst the bright mass of porphyry rocks makes these rocks non-homogeneous. This mixture of different components with varying properties affects the overall properties of porphyries, making them distinct from homogeneous substances.",
        """A substance or material is that which occupies space and has
      weight; that is, which presents a mass attracted by the earth and
      by other masses of material, and of which the _objects_ of nature
      are composed, and by means of which the motions and _phenomena_
      of nature are accomplished. It is easy to discover by examining
      and investigating, by various methods, the objects met with
      in nature and in the arts, that some of them are homogeneous,
      whilst others are composed of a mixture of several homogeneous
      substances. This is most clearly apparent in solid substances.
      The metals used in the arts (for example, gold, iron, copper)
      must be homogeneous, otherwise they are brittle and unfit for
      many purposes. Homogeneous matter exhibits similar properties in
      all its parts. By breaking up a homogeneous substance we obtain
      parts which, although different in form, resemble each other in
      their properties. Glass, pure sugar, marble, &c., are examples of
      homogeneous substances. Examples of non-homogeneous substances
      are, however, much more frequent in nature and the arts. Thus
      the majority of the rocks are not homogeneous. In porphyries
      bright pieces of a mineral called 'orthoclase' are often seen
      interspersed amongst the dark mass of the rock.""",
    )

    d = check_answer(inaccurate_qa_tuple_2, logic_llm)
    if False == d[0]:  # if not accurate
        print("Made right choice for bad question")
    else:
        print("Made wrong choice for bad question", d[0])
    # Fixed answer:
    accurate_qa_tuple = (
        "How does the presence of 'orthoclase' affect the properties of porphyries?",
        "The presence of bright pieces of a mineral called 'orthoclase' interspersed amongst the dark mass of porphyry rocks makes these rocks non-homogeneous. This mixture of different components with varying properties affects the overall properties of porphyries, making them distinct from homogeneous substances.",
        """A substance or material is that which occupies space and has
      weight; that is, which presents a mass attracted by the earth and
      by other masses of material, and of which the _objects_ of nature
      are composed, and by means of which the motions and _phenomena_
      of nature are accomplished. It is easy to discover by examining
      and investigating, by various methods, the objects met with
      in nature and in the arts, that some of them are homogeneous,
      whilst others are composed of a mixture of several homogeneous
      substances. This is most clearly apparent in solid substances.
      The metals used in the arts (for example, gold, iron, copper)
      must be homogeneous, otherwise they are brittle and unfit for
      many purposes. Homogeneous matter exhibits similar properties in
      all its parts. By breaking up a homogeneous substance we obtain
      parts which, although different in form, resemble each other in
      their properties. Glass, pure sugar, marble, &c., are examples of
      homogeneous substances. Examples of non-homogeneous substances
      are, however, much more frequent in nature and the arts. Thus
      the majority of the rocks are not homogeneous. In porphyries
      bright pieces of a mineral called 'orthoclase' are often seen
      interspersed amongst the dark mass of the rock.""",
    )
    d3 = check_answer(accurate_qa_tuple, logic_llm)
    if True == d3[0]:
        print("Made right choice for good question")
    else:
        print("Made wrong choice for good question", d3[0])
