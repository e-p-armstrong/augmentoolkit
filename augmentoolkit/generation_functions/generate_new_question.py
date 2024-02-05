import re

# try:
# from .question_grammar import question_grammar

from .constants import LOGICAL_MODEL


async def generate_new_question(qatuple, engine_wrapper):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    while not made_questions and (
        retries <= 5
    ):  # TODO - UPDATE and TEST the few-shot prompt with the latest from generate_questions
        question_prompt = f"""You are an expert educational AI that, given a paragraph or two from a text, will create a suitable educational question based on the paragraphs, and *only* based on the paragraphs. You are focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels). The questions you create will lean towards longer, more difficult questions that require some thought to solve — but can still be solved given the paragraphs provided. Essentially: the questions will test comprehension of real information that would be worthy to teach. After the question, you will also write its answer.

Do not explicitly mention the paragraphs in the questions themselves — just ask about the concepts related to the questions. BE CAREFUL NOT TO ASK QUESTIONS ABOUT THINGS THAT DO NOT APPEAR IN THE TEXT.

You will not mention the text explicitly in any questions you think of, since the questions you generate are intended to test people's knowledge of the information — when given the questions, they will not have the text on-hand.

### Instruction:
Text details: Road Construction, by Mark Ericsson

Text to make a question from: 
\"\"\"
Road construction is a multifaceted process involving various stages and materials, each critical for the durability and safety of the road. Initially, a thorough site survey and soil testing are conducted to assess the suitability of the terrain. Following this, the groundwork commences with the removal of topsoil and leveling of the area. Subsequently, a layer of sub-base material, typically composed of crushed stone or gravel, is laid to provide stability. This is followed by the base layer, often made of a stronger aggregate, to support the surface layer. The surface layer, usually asphalt or concrete, is then applied, offering a smooth and durable driving surface. Additionally, proper drainage systems are installed to prevent water accumulation, which can lead to road damage. Throughout the construction, environmental considerations are taken into account to minimize the impact on surrounding ecosystems. Regular maintenance, including patching and resurfacing, is essential to extend the road's lifespan and ensure safety for its users.
\"\"\"

### Response:
## Question:
1.) What is the purpose of conducting a site survey and soil testing in the initial stage of road construction?
Answer: The site survey and soil testing are conducted to assess the suitability of the terrain for road construction, ensuring the area is appropriate and will support the road structure effectively.

### Instruction:
Text details: Introduction to Mathematics, by Elise Delacroix

Text to make a question from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

### Response:
## Question:
1.) How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
Answer: The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.

### Instruction:
Text details: Thus Spake Zarathustra, by Friedrich Nietzsche

Text to make a question from: 
\"\"\"
When Zarathustra was thirty years old, he left his home and the lake of his home, and went into the mountains. There he enjoyed his spirit and
solitude, and for ten years did not weary of it. But at last his heart changed,—and rising one morning with the rosy dawn, he went before the
sun, and spake thus unto it:

Thou great star! What would be thy happiness if thou hadst not those for
whom thou shinest!

For ten years hast thou climbed hither unto my cave: thou wouldst have
wearied of thy light and of the journey, had it not been for me, mine
eagle, and my serpent.

But we awaited thee every morning, took from thee thine overflow and
blessed thee for it.

Lo! I am weary of my wisdom, like the bee that hath gathered too much
honey; I need hands outstretched to take it.

I would fain bestow and distribute, until the wise have once more become
joyous in their folly, and the poor happy in their riches.

Therefore must I descend into the deep: as thou doest in the
evening, when thou goest behind the sea, and givest light also to the
nether-world, thou exuberant star!

Like thee must I GO DOWN, as men say, to whom I shall descend.
\"\"\"

### Response:
## Question:
1.) What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
Answer: They need the acknowledgement and admiration of others. Take the line from 'Thus Spake Zarathustra' by Friedrich Nietzsche: "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.

### Instruction:
Text details: The Republic, by Plato

Text to make a question from: 
\"\"\"
I went down yesterday to the Piraeus with Glaucon the son of Ariston,
that I might offer up my prayers to the goddess (Bendis, the Thracian
Artemis.); and also because I wanted to see in what manner they would
celebrate the festival, which was a new thing. I was delighted with the
procession of the inhabitants; but that of the Thracians was equally,
if not more, beautiful. When we had finished our prayers and viewed the
spectacle, we turned in the direction of the city; and at that instant
Polemarchus the son of Cephalus chanced to catch sight of us from a
distance as we were starting on our way home, and told his servant to
run and bid us wait for him. The servant took hold of me by the cloak
behind, and said: Polemarchus desires you to wait.

I turned round, and asked him where his master was.

There he is, said the youth, coming after you, if you will only wait.

Certainly we will, said Glaucon; and in a few minutes Polemarchus
appeared, and with him Adeimantus, Glaucon’s brother, Niceratus the son
of Nicias, and several others who had been at the procession.

Polemarchus said to me: I perceive, Socrates, that you and your
companion are already on your way to the city.

You are not far wrong, I said.

But do you see, he rejoined, how many we are?

Of course.

And are you stronger than all these? for if not, you will have to
remain where you are.

May there not be the alternative, I said, that we may persuade you to
let us go?

But can you persuade us, if we refuse to listen to you? he said.

Certainly not, replied Glaucon.

Then we are not going to listen; of that you may be assured.
\"\"\"

### Response:
## Question:
1.) In Plato's "The Republic," in the dialogue where Polemarchus comments on the size of his group and questions Socrates' strength compared to it, ultimately stating that Socrates will have to remain where he is, what is Polemarchus implying?
Answer: Polemarchus is implying that since his group is stronger than Socrates, he can force Socrates to remain where he is.

### Instruction:
Text Details: Engineering Projects Throughout History, by Hugo Gonzalez

Text to make a question from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

### Response:
## Question:
1.) How much earth was excavated during the construction of the Panama Canal?
Answer: Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.

### Instruction:
Text Details: Engineering Projects Throughout History, by Hugo Gonzalez

Text to make a question from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

### Response:
## Question:
1.) How much earth was excavated during the construction of the Panama Canal?
Answer: Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.

### Instruction:
Text details: {qatuple[3]}

Text to make a question from: 
\"\"\"
{qatuple[2]}
\"\"\"

### Response:
## Question (based on text):
"""
        # print("DEBUG\n\n" + prompt=decision_prompt)
        sampling_params = {
            "max_tokens": 8000,
            "stop": ["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
            "temperature": 0.2,
        }
        print("--QA TUPLE DURING NEW Q GEN--")
        print(qatuple)
        completion = await engine_wrapper.submit(question_prompt, sampling_params)
        # print("COMPLETION:\n\n----------------------")
        # print(completion)
        # print("\n------------------")

        # Extract questions
        response_pattern = re.compile(
            r"Question \(based on text\):\n(.+)", re.IGNORECASE | re.DOTALL
        )
        generation = response_pattern.search(completion).group(1)
        # print("GENERATION:\n\n-------------------\n\n", generation)
        # print("-------------------")
        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(generation)
        if len(matches) > 0:
            print("Made Qs, yay!")
            made_questions = True
        else:
            print("retry!")
            retries += 1

    for match in matches:
        return (
            match[0].replace(") ", "", 1).strip(),
            match[1].replace(") ", "", 1).strip(),
            qatuple[2].replace(") ", "", 1),
            qatuple[3],
        ), completion
    print("Should not have reached here")
    print(matches)
    print(questions)
    return questions, completion


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    text = """The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.

The earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."""

    plan = """First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). I will start by looking at one or two sentences at a time. Let's begin with: "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years." This paragraph is saying that people used to know only about 3,000 years of history, but now they know much more. So I might ask something like "What was the time period in which people had limited knowledge?" The question tests knowledge of when historical records were incomplete and therefore difficult to access. It requires understanding of the text as well as analysis to determine what time period is being referred to here.  Next, I'll look at: "Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end." This paragraph states that the universe seems infinite because of its reflection, but it could actually be finite with two ends. So I might ask something like "How does the structure of the universe affect its size?" This question tests understanding of the concept of reflection and how it can distort perceptions of size, as well as analysis to determine what implications this might have for the actual size of the universe. Then I'll move on: "The earth ... circles about the sun in a slightly distorted and slowly variable oval path in a year." This paragraph talks about the Earth's orbit around the Sun. So I might ask something like "What is the shape of the Earth's orbit around the sun?" This question tests understanding of the text as well as analysis to determine the specific shape mentioned here. Lastly, let's examine: "Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."  This paragraph says that the Earth's distance from the Sun changes throughout the year. So I might ask something like "Why does the Earth's distance from the Sun change over the course of the year?" This question tests understanding of the text as well as analysis to determine why this change occurs."""
    print("Begin HGWELLS test")
    # result = generate_question(text,plan,logic_llm)

    print("Begin MENDELEEV Test")
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

    plan3 = """Step 1. Identify Key Topics: The key topics in this paragraph include homogeneous and non-homogeneous substances, their properties, and how they are extracted from natural or artificial mixtures.
Step 2. Determine Information-Rich Areas: The text provides detailed descriptions of the properties of homogeneous and non-homogeneous substances, as well as methods for extracting them from mixtures.
Step 3. Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and processes related to homogeneous and non-homogeneous substances. Example: "What is the definition of a homogeneous substance?"
Step 4. Develop Questions Exploring Relationships Between Concepts: Generate questions that explore how one concept relates to another within the text. Example: "How does the extraction process for gold differ from that of orthoclase?"
Step 5. Create Questions Investigating Application of Concepts: Look for examples of applications of these concepts in nature or industry that are mentioned in the text. Example: "In what ways is the concept of homogeneous substances used in chemistry?\""""

    result2 = generate_new_question((text2, "Principles of chemistry"), logic_llm)
    print("GENERATION TEST2:\n\n-------------------\n\n", result2)
