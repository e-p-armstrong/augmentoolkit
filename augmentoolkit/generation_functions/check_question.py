import re
import traceback

# from .question_relevant_grammar import question_relevant_grammar
from .constants import LOGICAL_MODEL
from aphrodite import SamplingParams

# POSSIBLE TODO:
# Add an "Off the rails insane forever generation" check, like on this output:
# I haven't seen this happen since I started using 70bs though.
"""
4.) What did ancient people believe about the shape and movement of our world, before scientific discoveries indicated otherwise? (Hint: The earth is a spheroid, not flat. The universe existed before the year 4004 BCE. The earth rotates on its axis and orbits around the sun.) You are allowed to dramatically change/revamp/rewrite this question's content. The goal here is to create a different question that still requires information from the paragraphs, but doesn't require knowledge of flawed assumptions connected with biblical interpretations or an understanding of events before the creation of the universe, which goes beyond the generally accepted scientific evidence provided in the text.  As you see fit. (Don't just rephrase the old question.) The goal here is to create a different question that still requires information from the paragraphs, but doesn't require knowledge of flawed assumptions connected with biblical interpretations or an understanding of events before the creation of the universe, which goes beyond the generally accepted scientific evidence provided in the text.  As you see fit. (Don't just rephrase the old question.) So you should create a new question that only requires information from the paragraphs to solve. It is not necessary to refer to the text at all. Also note: An example about how NOT to ask questions is if the text states fact X, but does not explain how X was established, do not ask a question "How do we know X". But instead you might consider asking how X relates to other facts in the paragraph, or how these facts together lead to a progression of ideas, "Explain how X, Y, and Z are related" for instance. Here's what we know: 1. The earth is a spheroid, not flat (paragraph 3). 2. The universe existed before the year 4004 BCE (paragraph 5). 3. The earth rotates on its axis and orbits around the sun (paragraph 5).
Answer: We don't have an answer.
"""


# Answer vetting
async def check_question(qatuple, engine_wrapper):
    retries = 0
    while retries <= 4:
        decision_prompt = f"""<s> [INST] You are an expert educational AI. Given a paragraph or two from a larger text, and a question based on the paragraphs, you will make a determination as to whether the question tests ONLY information in the paragraphs. Essentially: you will check if the question is answerable, given the information in the paragraphs. Your task includes first analyzing the text, thinking through whether or not the question reflects aspects of the paragraphs provided. 

Following this, at the very end of your response, your "final judgment" or "final answer", you will write "Relevant" or "Irrelevant" depending on your analysis of the question with regards to the text. 

Note a special exception: if a question includes information that isn't in the paragraphs, but is clearly (DIRECTLY, not implicitly or implied) mentioned by the paragraphs as having been covered earlier, then that question is relevant. Essentially: questions are allowed to cover content that the text has explicitly covered in the past.

Write out the reasoning and analysis behind your judgment, step-by-step. Your analysis of the question, against the text, should follow a logical progression of steps that results in a conclusive and accurate final answer.

You will analyze the question step-by-step, ensuring each part of the question is individually compared to the text. The key steps are analyzing the text, understanding the question, and then systematically comparing each part of the question with the text. The process continues until either a part of the question is found not to be covered by the text, leading to a judgment of "Irrelevant," or until all parts of the question have been compared and found to be covered by the text, leading to a judgment of "Relevant." This method allows for a thorough and detailed assessment, ensuring that the final judgment accurately reflects the extent to which the question is based on the given text.

Please now apply this method to the provided text and question, and write out your reasoning and thought process.

### Instruction:
Text: 
\"\"\"
The concept of artificial intelligence (AI) revolves around the creation of machines capable of intelligent behavior. Key components of AI include machine learning, neural networks, and natural language processing. Machine learning involves training computers to learn from data and improve their performance over time. Neural networks are modeled after the human brain's network of neurons and are pivotal in enabling machines to recognize patterns and make decisions. Natural language processing, another crucial aspect of AI, allows machines to understand and interpret human languages, facilitating interaction between humans and computers.
\"\"\"

Question (based on text): \"\"\"What is the role of neural networks in AI, and how does natural language processing contribute to human-computer interaction?\"\"\"

[/INST]### Response:
## Reasoning and thought process:

### In-Depth Analysis of the Text:
#### Content and Depth: The text details key components of AI, including machine learning, neural networks, and natural language processing, with a focus on their roles and functions in AI.
#### Type of Information: The information is primarily descriptive, outlining the roles and functions of neural networks and natural language processing in AI.

### Detailed Understanding of the Question:
#### Core Requirement: The question asks about the roles of neural networks in AI and the contribution of natural language processing to human-computer interaction, seeking specific functional explanations.
#### Depth of Detail: The question requires an understanding of the specific functions and impacts of these AI components.

### Targeted Comparison of the Question with the Text:
#### Content Match: The text aligns well with the question, covering the roles of neural networks in pattern recognition and decision-making, and the role of natural language processing in understanding and interpreting human languages.
#### Depth Match: The text provides the necessary level of detail to answer the question, though it remains descriptive and does not delve into deeper explanatory mechanisms.

### Critical Evaluation and Final Judgment:
The text addresses both components of the question with sufficient detail, making it: Relevant.</s> [INST]


### Instruction:
Text: 
\"\"\"
The phenomenon of photosynthesis in plants is an essential process for life on Earth. It involves the conversion of light energy into chemical energy, which is stored in glucose. This process occurs in the chloroplasts, specifically within a pigment called chlorophyll. Photosynthesis consists of two main stages: the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. During the light-dependent reactions, sunlight is absorbed by chlorophyll, which then converts water (H2O) into oxygen (O2) and transfers energy to the light-independent reactions. The Calvin cycle uses this energy to convert carbon dioxide (CO2) from the atmosphere into glucose, a simple sugar.
\"\"\"

Question (based on text): \"\"\"Explain exactly why chlorophyll able to absorb sunlight, and why is this important for the Calvin cycle?\"\"\"

[/INST]### Response:
## Reasoning and thought process:

### In-Depth Analysis of the Text:
#### Content and Depth: The text discusses the process of photosynthesis, including the roles of chlorophyll, light-dependent reactions, and the Calvin cycle.
#### Type of Information: Information is descriptive, explaining the stages of photosynthesis and the role of chlorophyll in light absorption.

### Detailed Understanding of the Question:
#### Core Requirement: The question seeks an explanation of why chlorophyll can absorb sunlight and its importance for the Calvin cycle, requiring detailed scientific principles.
#### Depth of Detail: Requires an in-depth, explanatory response about the biological and chemical properties of chlorophyll.

### Targeted Comparison of the Question with the Text:
#### Content Match: The text partially matches the question by mentioning the role of chlorophyll in the light-dependent reactions.
#### Depth Match: The text lacks the specific scientific principles explaining the mechanism of sunlight absorption by chlorophyll, which is central to the question.

### Critical Evaluation and Final Judgment:
Although the text touches upon the role of chlorophyll, it does not provide the depth of explanation required by the question, making it: Irrelevant.</s> [INST]


### Instruction:
Text: 
\"\"\"
The octopus is an extraordinary creature, renowned for its intelligence and ability to adapt to its environment. Octopuses have three hearts and a complex nervous system, which includes a central brain and a small brain in each of their eight arms. These arms are capable of intricate movements and can even carry out simple tasks independently. The creature's ability to change its color and texture is another remarkable feature, which it uses for camouflage and communication. Notably, octopuses are known for their problem-solving skills and have been observed using tools in captivity.
\"\"\"

Question (based on text): \"\"\"How do octopuses utilize their color-changing ability in their natural habitat, and what role does their central brain play in this process?\"\"\"

[/INST]### Response:
## Reasoning and thought process:

### In-Depth Analysis of the Text:
#### Content and Depth: The text outlines various aspects of the octopus, such as its physiology, nervous system, color-changing ability, and intelligence.
#### Type of Information: Descriptive, covering the abilities and features of the octopus, including color-changing for camouflage and communication.

### Detailed Understanding of the Question:
#### Core Requirement: The question asks how octopuses utilize their color-changing ability and the role of their central brain in this process.
#### Depth of Detail: Seeks specific information on the function and control mechanism of the color-changing ability.

### Targeted Comparison of the Question with the Text:
#### Content Match: The text aligns with the first part of the question regarding the use of color-changing for camouflage and communication.
#### Depth Match: The text does not provide information about the role of the central brain in this process, lacking the required depth on the control mechanism.

### Critical Evaluation and Final Judgment:
Given the text's coverage of color-changing but lack of detail on the central brain's role, the overall assessment of the question's relevance to the text is: Irrelevant.</s> [INST]


### Instruction:
Text: 
\"\"\"
{qatuple[2]}
\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

If the question clearly goes off the rails and is incoherent, then it is irrelevant.

[/INST]### Response:
## Reasoning and thought process (be careful around "how" and "why" questions):
"""
        try:
            sampling_params = SamplingParams(
                max_tokens=4000,
                stop=["</s>", "# Input:", "[INST]", "### Instruction"],
                temperature=0.2,
            )
            completion = await engine_wrapper.submit(decision_prompt, sampling_params)

            response_pattern = re.compile(
                r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )
            response = response_pattern.search(completion).group(1).strip()
            decision_pattern = re.compile(
                r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE
            )
            # print(response)
            determination = decision_pattern.search(response).group(1).strip()
            # print("\n\nDETERMINATION:\n------")
            # print(determination)
            # print("\n---------\n")
            if (
                "irrelevant" in determination
                or "Irrelevant" in determination.lower()
                or "mostly" in determination.lower()
                or "partial" in determination.lower()
                or "introduces information not present in the text"
                in determination.lower()
            ):
                return (False, response), completion
            elif "relevant" in determination or "Relevant" in determination:
                return (True, response), completion
            else:
                print("Did not contain relevant or irrelevant! Retrying")
                retries += 1
        except Exception as e:
            print("Exception!", e)
            traceback.print_exc()
            if retries <= 4:
                retries += 1
            else:
                return (None, None), completion
    return (None, None), None


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    q_test = [
        (
            "Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
        ),
        (
            "Identify and explain changes in human understanding throughout history regarding the age of the Earth.",
            "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.",
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
    # Try to detect bad question
    d = check_question(q_test[2], logic_llm)
    if not d[0]:  # if not relevant
        print("Made right choice for bad question")
    else:
        print("Made wrong choice for bad question")
    d2 = check_question(q_test[1], logic_llm)
    if d2[0]:
        print("Made right choice for good question")
    else:
        print("Made wrong choice for good question")

    print("Begin Mendeleev test")
    text2 = """A substance or material is that which occupies space and has
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
      interspersed amongst the dark mass of the rock. In ordinary red
      granite it is easy to distinguish large pieces of orthoclase mixed
      with dark semi-transparent quartz and flexible laminæ of mica.
      Similarly, plants and animals are non-homogeneous. Thus, leaves
      are composed of a skin, fibre, pulp, sap, and a green colouring
      matter. As an example of those non-homogeneous substances which
      are produced artificially, gunpowder may be cited, which is
      prepared by mixing together known proportions of sulphur, nitre,
      and charcoal. Many liquids, also, are not homogeneous, as may be
      observed by the aid of the microscope, when drops of blood are
      seen to consist of a colourless liquid in which red corpuscles,
      invisible to the naked eye owing to their small size, are floating
      about. It is these corpuscles which give blood its peculiar
      colour. Milk is also a transparent liquid, in which microscopical
      drops of fat are floating, which rise to the top when milk is
      left at rest, forming cream. It is possible to extract from every
      non-homogeneous substance those homogeneous substances of which
      it is made up. Thus orthoclase may he separated from porphyry by
      breaking it off. So also gold is extracted from auriferous sand by
      washing away the mixture of clay and sand. Chemistry deals only
      with the homogeneous substances met with in nature, or extracted
      from natural or artificial non-homogeneous substances. The various
      mixtures found in nature form the subjects of other natural
      sciences--as geognosy, botany, zoology, anatomy, &c."""
    q_test_2 = [  # note that the full text isn't included in each of the tuples here, I need to  change that
        (
            "Why is it important to distinguish between homogeneous and non-homogeneous substances?",
            "Homogeneous substances consist of parts that resemble each other in their properties, while non-homogeneous substances are made up of several homogeneous substances mixed together. Chemistry deals with the homogeneous substances met with in nature or extracted from natural or artificial non-homogeneous substances, so it is important to distinguish between them because it determines which parts of a given substance can be used for chemical analysis and study.",
        ),
        (
            "What is an example of an artistic mixture that would be non-homogeneous?",
            "An example of a non-homogeneous artistic mixture could be gunpowder, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal.",
        ),
        (
            "How might the concept of homogeneity apply to education or learning?",
            "In education or learning, students can think about their own knowledge as a homogeneous substance, made up of similar concepts that resemble each other in terms of understanding. They may need to separate out these ideas from non-homogeneous ones (e.g., misconceptions) in order to fully grasp the concept and build upon it.",
        ),
        (
            "If we were told to find homogeneous substances in nature, how would we go about doing this?",
            "To find homogeneous substances in nature, one could examine and investigate various objects met with in nature and in the arts. Some of these objects might be homogeneous, whilst others are composed of a mixture of several homogeneous substances. By breaking up a homogeneous substance, we would obtain parts which, although different in form, resemble each other in their properties. This suggests that we could identify homogeneous substances by looking for these characteristics. Additionally, some examples mentioned in the text include gold, iron, copper, glass, pure sugar, marble, and ordinary red granite. However, not all non-homogeneous substances are immediately apparent; it requires investigating and understanding how they are made up of different components (such as orthoclase being separated from porphyry). Therefore, a combination of observing physical properties, breaking down materials, and understanding their composition would allow us to identify homogeneous substances in nature.",
        ),
    ]
