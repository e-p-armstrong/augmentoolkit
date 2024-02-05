import re

# from .questions_grammar import questions_grammar
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps


async def generate_questions(para_tuple, plan, engine_wrapper):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)

    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    while not made_questions and (retries <= 5):
        question_prompt = f"""You are an expert educational AI that, given a paragraph or two from a text, will create suitable educational questions based on the paragraphs, and *only* based on the paragraphs. You are focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels). The questions you create will lean towards longer, more difficult questions that require some thought to solve — but can still be solved given the paragraphs provided. Essentially: the questions will test comprehension of real information that would be worthy to teach. After the question, you will also write its answer.

Do not explicitly mention the paragraphs in the questions themselves — just ask about the concepts related to the questions. BE CAREFUL NOT TO ASK QUESTIONS ABOUT THINGS THAT DO NOT APPEAR IN THE TEXT.

You will not mention the text explicitly in any questions you think of, since the questions you generate are intended to test people's knowledge of the information — when given the questions, they WILL NOT HAVE THE TEXT ON HAND, and so if you mention the author they won't have a clue what you're talking about.


### Instruction:
Text details: Introduction to Mathematics, by Jane Smith

Text to make questions from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

### Response:
## Reasoning and thought process:
Identify Key Topics: The key topics in this paragraph are linear and quadratic functions in mathematics, their definitions, components, and applications.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and components of these functions. Example: "What does the 'm' in the linear function equation represent?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the equations. Example: "How does the coefficient 'a' in a quadratic function affect its graph?"
Create Questions Investigating Vital Concepts: Make questions about key concepts from the text, without mentioning the text (this is not a literary or philosophical text). Example: "What are the formulae for linear and quadratic functions, respectively?"
Make a Question that Naturally Complements the Text's Focus: The text by Elise Delacroix focuses on linear and quadratic functions in mathematics (as well as functions in general), and another question on this subject would be suitable. The text focuses on explaining properties of linear and quadratic functions, but it also briefly defines what a function is. Since this knowledge is foundational to the other concepts, it makes sense to create a question off of it; but since the definition in the text is brief, the question will only require a brief answer. Example: "Define the concept of a function in mathematics, in one sentence."

## Questions:
1.) How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
Answer: The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.

2.) What role does the y-intercept 'b' play in graphing a linear function?
Answer: The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.

3.) In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?
Answer: The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.

4.) Define the concept of a function in mathematics, in one sentence.
Answer: A function is a relationship where each input is associated with exactly one output.


### Instruction:
Text details: Thus Spake Zarathustra, by Friedrich Nietzsche

Text to make questions from: 
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
## Reasoning and Thought Process:
Identify Key Themes: In this excerpt, we find themes of transformation, self-discovery, the value of wisdom, and the need for recognition of effort and wisdom.
Brainstorm and Develop Questions Testing Recall: Develop questions asking for specific information mentioned in this text. Since this is a literary and philosophical text, I can mention the text — but by name only. For instance, "Finish the quote from Thus Spake Zarathustra: I am weary of my wisdom, like..."
Devise Questions Testing Comprehension of, and Belief in, the Text's Opinions: Questions can be formed to check understanding and acceptance of the text's philosophy. For instance, "What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?" Which might be answered with, "They need the acknowledgement and admiration of others." and then supported by lines such as "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" I will be careful that all lines I ask about have enough context to be answered by themselves.
Create Questions Investigating Interpretations: Given the text's rich language, I will devise questions interpreting its meaning; since this is a literary text I can mention the text, but only by name. Example: "Why did Zarathustra, in 'Thus Spake Zarathustra', leave the mountains and become a man again?"
Make a Question that Naturally Complements the Text's Focus: The text by Friedrich Nietzsche focuses on Zarathustra's monologue, and another question on this subject would be suitable. Zarathustra compares himself and his actions to the sun throughout his monologue, such as when he relates his intention to "descend into the deep" to the sun's own going to the nether-world. A good question might ask after this crucial comparison, and could be "List one way in which Zarathustra compares himself to the sun, in 'Thus Spake Zarathustra'."

## Questions:
1.) Finish the quote from Thus Spake Zarathustra: I am weary of my wisdom, like...
Answer: "... the bee that hath gathered too much honey; I need hands outstretched to take it."

2.) What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
Answer: They need the acknowledgement and admiration of others. Take the line from 'Thus Spake Zarathustra' by Friedrich Nietzsche: "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.

3.) Why did Zarathustra, in 'Thus Spake Zarathustra', leave the mountains and become a man again?
Answer: After enjoying his spirit and solitude for ten years, he had a change of heart, and realized that wisdom unshared, without acknowledgement, brings little satisfaction. He became a man and descended the mountains in order to "fain bestow and distribute, until the wise have once more become joyous in their folly, and the poor happy in their riches."

4.) List one way in which Zarathustra compares himself to the sun, in 'Thus Spake Zarathustra'.
Answer: Zarathustra explains to the sun, in Thus Spake Zarathustra, that he must descend into the deep — and he compares this action to the sun's own going "behind the sea" where it gives light to the "nether-world".


### Instruction:
Text details: The Republic, by Plato

Text to plan questions from:
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
## Reasoning and Thought Process:
Identify Key Themes: This excerpt from Plato's "The Republic" presents themes of social interaction, philosophical dialogue, and the dynamics of persuasion and power.
Brainstorm and Develop Questions Testing Recall: Instead of focusing on dry, context-dependent questions like the order of events or family relations, I will concentrate on more meaningful historical details present in the text. For instance, "Which goddess is Bendis equivalent to?"
Devise Questions Exploring Comprehension of, and Belief in, the Text's Opinions: I will create questions that delve into the mind and thoughts of the writer. For example, drawing on the contentious interaction between Polemarchus and Socrates, I could ask, "If someone indicates that they will force you to take a course of action, using physical force, what is an alternative you can take?" with the answer being "Persuade them to let me go." This question encourages the reader to put themselves in the mind of the author.
Create Questions Investigating Interpretations: Focusing on the dialogue's subtleties, I will craft an interpretative question. Since this is a literary text, I can reference the text, but only by name. So the question might be, "In Plato's "The Republic," in the dialogue where Polemarchus comments on the size of his group and questions Socrates' strength compared to it, ultimately stating that Socrates will have to remain where he is, what is Polemarchus implying?" This question prompts critical thinking about power dynamics and conversational implications depicted.
Make a Question that Naturally Complements the Text's Focus: The text by Plato focuses on the visit to Piraeus and the interaction with Glaucon, and another question on this subject would be suitable. The text focuses on the interaction between Polemarchus and Socrates, but enough questions have already been asked about that subject to cover it completely. That leaves the visit to Piraeus as a subject for questions. The cause behind Socrates' visit to Piraeus is directly mentioned in the text and could make for a good recall question, so one question may be "Why did Socrates visit Piraeus with Glaucon in Plato's 'The Republic'?"

## Questions:
1.) Which goddess is Bendis equivalent to?
Answer: The Thracian Goddess Bendis is equivalent to Artemis.

2.) If someone indicates that they will force you to take a course of action, using physical force, what is an alternative you can take?
Answer: One viable alternative I can take is to persuade them to let me go. This, of course, requires that the other party does not refuse to listen to me.

3.) In Plato's "The Republic," in the dialogue where Polemarchus comments on the size of his group and questions Socrates' strength compared to it, ultimately stating that Socrates will have to remain where he is, what is Polemarchus implying?
Answer: Polemarchus is implying that since his group is stronger than Socrates, he can force Socrates to remain where he is.

4.) Why did Socrates visit Piraeus with Glaucon in Plato's "The Republic"?
Answer: Socrates visited Piraeus to offer up his prayers to the goddess Bendis, as well as to see in what manner the Thracians would celebrate the festival.


### Instruction:
Text details: Great Construction Projects Throughout History, by Marco Gonzalez

Text to make questions from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

### Response:
## Reasoning and Thought Process:
Identify Key Topics: The paragraph details specific aspects of the Panama Canal's construction, focusing on its challenges, innovations, and impacts. Topics include construction challenges, health issues, excavation techniques, and the canal's impact on global trade.
Brainstorm and Develop Questions Testing Recall: Questions can be formed to recall factual data from the text. Example: "How much earth was excavated during the construction of the Panama Canal?"
Devise Questions Exploring Cause and Effect Relationships: This involves creating questions that explore how certain challenges led to specific solutions and impacts. Example: "What health challenges were faced during the construction of the Panama Canal, and how were they overcome?"
Create Questions Investigating Quantitative Values: Given the text's focus on concrete numbers, I will devise questions that require analyzing these figures. Example: "By how many miles did the Panama Canal reduce the sea voyage from San Francisco to New York?"
Make a Question that Naturally Complements the Text's Focus: The text by Marco Gonzalez focuses on the Panama Canal (and recall-oriented facts about it), and another question on this subject would be suitable. One small area that has not yet been tested, and which is an important detail mentioned in the text, is be: "In what year was the Panama Canal completed?"

## Questions:
1.) How much earth was excavated during the construction of the Panama Canal?
Answer: Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal.

2.) What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
Answer: The construction faced significant health challenges, notably malaria and yellow fever. 

3.) By how many miles did the Panama Canal reduce the sea voyage from San Francisco to New York?
Answer: The completion of the Panama Canal reduced the sea voyage from San Francisco to New York by around 8,000 miles.

4.) In what year was the Panama Canal completed?
Answer: The Panama Canal's construction was completed in 1914.


### Instruction:
Text details: The Scale of the Solar System, by Rob Robertson

Text to plan questions from:
\"\"\"
If, then, we represent our earth as a little ball of one inch diameter, the sun would be a big globe nine feet across and 323 yards away, that is about a fifth of a mile, four or five minutes’ walking. The moon would be a small pea two feet and a half from the world.  Between earth and sun there would be the two inner planets, Mercury and Venus, at distances of one hundred and twenty-five and two hundred and fifty  yards from the sun. All round and about these bodies there would be  emptiness until you came to Mars, a hundred and seventy-five feet beyond the earth; Jupiter nearly a mile away, a foot in diameter; Saturn, a little smaller, two miles off; Uranus four miles off and Neptune six miles off. Then nothingness and nothingness except for small particles and drifting scraps of attenuated vapour for thousands of miles.
\"\"\"

### Response:
## Reasoning and thought process:
Identify Key Topics: The key topic in this paragraph is the scale of the solar system, which is presented using analogy to make the distances easier to conceive of.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of the components of this analogy. Since the text is not provided while the question is being asked, I will have to include the analogy in the question. Example: "If the Earth had a diameter of only one inch, how far would the moon be from the Earth, if it (and its distance from the Earth) were also resized proportionally?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different information in the excerpt. Notably, in this text the distances between planets are much greater than the planets' sizes. Example question: "How do the distances between planets compare to their sizes?"
Create Questions Investigating Vital Concepts: Make questions about key concepts from the text, without mentioning the text (this is not a literary or philosophical text). A key concept is the distances between planets. Since the paragraph relies on an analogy, which the reader of the question will not have access to, I will have to provide this context. Example: "If you scaled everything down so that the Earth had a diameter of one inch, then how far would Mercury and Venus be from the sun?"
Make a Question that Naturally Complements the Text's Focus: The text by Rob Robertson focuses on the size-based analogy described throughout, and another question on this subject would be suitable. Another question could simply ask about information that's already been tested, but for a different planet that is also mentioned in the text. Being mindful that the reader of the text will not have access to the source text, I will provide the analogy as context, as I ask the question: "If the earth had a diameter of only one inch, how far would Mars be from the earth, if it (and its distance from the earth) were also resized proportionally?"

## Questions:
1.) If the earth had a diameter of only one inch, how far would the moon be from the earth, if it (and its distance from the earth) were also resized proportionally?
Answer: The moon would be two feet and a half from the earth. The moon's size, under this new scale, would be that of a small pea.

2.) How do the distances between planets compare to their sizes?
Answer: The distances between planets is much greater than the planets' sizes. If you shrunk everything down so that the Earth was one inch in diameter, then the sun would be 323 yards away.

3.) If you scaled everything down so that the earth had a diameter of one inch, then how far would Mercury and Venus be from the sun?
Answer: Mercury would be one hundred and twenty-five yards from the sun, and Venus would be two hundred and fifty yards from the sun.

4.) If the earth had a diameter of only one inch, how far would Mars be from the earth, if it (and its distance from the earth) were also resized proportionally?
Answer: Mars would be a hundred and seventy-five feet beyond the earth.


### Instruction:
Text details: {para_tuple[1]}

Text to make questions from: 
\"\"\"
{para_tuple[0]}
\"\"\"

### Response:
## Reasoning and Thought Process:
{strip_steps(plan)}

## Questions (make 4):
"""
        # print("DEBUG\n\n" + prompt=decision_prompt)
        sampling_params = {
            "max_tokens": 12000,
            "stop": ["</s>", "# Input:", "[INST]", "### Instruction", "[INST"],
            "temperature": 0.8,
            # top_k=-1,
            "top_p": 1,
            # min_p=0.5,
        }
        completion = await engine_wrapper.submit(question_prompt, sampling_params)

        # Extract questions
        response_pattern = re.compile(
            r"Questions \(make 4\):\n(.+)", re.IGNORECASE | re.DOTALL
        )
        generation = response_pattern.search(completion).group(1)
        # print("GENERATION:\n\n-------------------\n\n", generation)
        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(generation)
        if len(matches) > 0:
            made_questions = True
        else:
            retries += 1
    if retries > 5:
        return None, None

    for match in matches:
        questions.append(
            (
                match[0].replace(") ", "", 1).strip(),
                match[1].replace(") ", "", 1).strip(),
                para_tuple[0].replace(") ", "", 1),
                para_tuple[1].replace(") ", "", 1),
            )
        )

    return questions, completion


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=12000,
        n_gpu_layers=100,
        verbose=True,
    )  # load the logical LLM and offload everything
    text = """The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.

The earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."""

    plan = """First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). I will start by looking at one or two sentences at a time. Let's begin with: "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years." This paragraph is saying that people used to know only about 3,000 years of history, but now they know much more. So I might ask something like "What was the time period in which people had limited knowledge?" The question tests knowledge of when historical records were incomplete and therefore difficult to access. It requires understanding of the text as well as analysis to determine what time period is being referred to here.  Next, I'll look at: "Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end." This paragraph states that the universe seems infinite because of its reflection, but it could actually be finite with two ends. So I might ask something like "How does the structure of the universe affect its size?" This question tests understanding of the concept of reflection and how it can distort perceptions of size, as well as analysis to determine what implications this might have for the actual size of the universe. Then I'll move on: "The earth ... circles about the sun in a slightly distorted and slowly variable oval path in a year." This paragraph talks about the Earth's orbit around the Sun. So I might ask something like "What is the shape of the Earth's orbit around the sun?" This question tests understanding of the text as well as analysis to determine the specific shape mentioned here. Lastly, let's examine: "Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."  This paragraph says that the Earth's distance from the Sun changes throughout the year. So I might ask something like "Why does the Earth's distance from the Sun change over the course of the year?" This question tests understanding of the text as well as analysis to determine why this change occurs."""
    print("Begin HGWELLS test")
    # result = generate_questions((text,"A Short History of the World, by HG Wells"),plan,logic_llm)

    print("Begin MENDELEEV Test")
    plan2 = """First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). Now, immediately I can see that the following passage might be a good basis for a question: "It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances." This seems like it could be a good basis for an application question. "By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties." This might also be a good basis for an application question. "Gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal." This seems like it could be a good basis for an analysis question. "Chemistry deals only with the homogeneous substances met with in nature, or extracted from natural or artificial non-homogeneous substances." This might also be a good basis for an application question."""
    text2 = """A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, copper) must be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about."""
    text_kant = """I. Of the difference between Pure and Empirical Knowledge

That all our knowledge begins with experience there can be no doubt.
For how is it possible that the faculty of cognition should be awakened
into exercise otherwise than by means of objects which affect our
senses, and partly of themselves produce representations, partly rouse
our powers of understanding into activity, to compare to connect, or to
separate these, and so to convert the raw material of our sensuous
impressions into a knowledge of objects, which is called experience? In
respect of time, therefore, no knowledge of ours is antecedent to
experience, but begins with it.

But, though all our knowledge begins with experience, it by no means
follows that all arises out of experience. For, on the contrary, it is
quite possible that our empirical knowledge is a compound of that which
we receive through impressions, and that which the faculty of cognition
supplies from itself (sensuous impressions giving merely the occasion),
an addition which we cannot distinguish from the original element given
by sense, till long practice has made us attentive to, and skilful in
separating it. It is, therefore, a question which requires close
investigation, and not to be answered at first sight, whether there
exists a knowledge altogether independent of experience, and even of
all sensuous impressions? Knowledge of this kind is called à priori, in
contradistinction to empirical knowledge, which has its sources à
posteriori, that is, in experience.

But the expression, “à priori,” is not as yet definite enough
adequately to indicate the whole meaning of the question above started.
For, in speaking of knowledge which has its sources in experience, we
are wont to say, that this or that may be known à priori, because we do
not derive this knowledge immediately from experience, but from a
general rule, which, however, we have itself borrowed from experience.
Thus, if a man undermined his house, we say, “he might know à priori
that it would have fallen;” that is, he needed not to have waited for
the experience that it did actually fall. But still, à priori, he could
not know even this much. For, that bodies are heavy, and, consequently,
that they fall when their supports are taken away, must have been known
to him previously, by means of experience."""

    plan3 = """Identify Key Topics: The key topics in this paragraph are homogeneous and non-homogeneous substances, their characteristics, examples of each type, and how they can be identified.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and characteristics of homogeneous and non-homogeneous substances. Example: "What is a homogeneous substance?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the examples given in the text. Example: "How does the presence of 'orthoclase' affect the properties of porphyries?"
Create Questions Investigating If-Then Relationships (where since one thing is true, another follows): Make questions that examine how different characteristics lead to identifying a substance as homogeneous or non-homogeneous. Example: "How can we determine if a substance is homogeneous based on its properties?\""""

    plan3_contextver = """Identify Key Topics: The paragraph discusses homogeneous substances, their properties, and examples of them in nature.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test recall of these facts. Example: "What is a homogenous substance?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the text. Example: "How do homogeneous substances differ from non-homogeneous ones?"
Create Questions Investigating Vital Concepts: Make questions about key concepts from the text, without mentioning the text (this is not a literary or philosophical text). Example: "What are some examples of homogenous substances in nature?"
Make a Question that Naturally Complements the Text's Focus: The paragraph focuses on explaining homogeneous substances and their properties, but also defines what a substance is. Since this knowledge is foundational to the other concepts, it makes sense to create a question of it; since the definition in the text is brief, the question will only require a brief answer. Example: "Define a substance in chemistry, in one sentence.\""""

    plan4 = """Identify Key Topics: The main themes in this excerpt from Kant's "Critique of Pure Reason" are knowledge, experience, and empiricism.
Brainstorm and Develop Questions Testing Recall: I will devise questions that recall specific information from the text. Example: "What does Immanuel Kant say all our knowledge begins with?"
Devise Questions Exploring Relationships Between Ideas: This philosophical text allows me to mention it by name, so I can ask about its ideas and their relationships. For instance, "What is the difference between empirical and pure knowledge according to Kant in 'The Critique of Pure Reason'?"
Create Questions Investigating Interpretations: Given the text's complexity, I will devise questions that require interpretation. Example: "Why does Kant believe it's possible for us to have knowledge independent of experience?" This question can be answered by analyzing his statements about how we may know things without direct experience."""
    plan4 = """Identify Key Topics: This excerpt from Kant's "Critique of Pure Reason" presents themes of knowledge, experience, and empirical versus pure knowledge.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of specific details mentioned in this text. Since this is a philosophical text, I can mention it by name but only to reference it. For instance, "What does Kant say about how all our knowledge begins?"
Devise Questions Exploring Comprehension of, and Belief in, the Text's Opinions: Questions can be formed to check understanding and acceptance of the text's philosophy. For example, "Does Kant believe that knowledge is always derived from experience?" Which might be answered with "No" and supported by lines such as "It is quite possible that our empirical knowledge is a compound of that we receive through impressions, and that the faculty of cognition supplies from itself." I will be careful that all questions have enough context to be answered by themselves.
Create Questions Investigating Interpretations: Given the text's rich language, I will devise questions interpreting its meaning; since this is a philosophical text I can mention the text, but only by name. Example: "In 'The Critique of Pure Reason', why does Kant say that all knowledge begins with experience?"
Make a Question that Naturally Complements the Text's Focus: The text by Immanuel Kant focuses on the distinction between empirical and pure knowledge, another question on this subject would be suitable. Since I have already asked about the difference between these two types of knowledge, I will ask about the source of knowledge in general, which is mentioned throughout the text. A good question might be "What does Kant say about the origin of knowledge?\""""

    plan5 = """Identify Key Topics: The key topics in this paragraph are homogeneous substances, their properties, and examples of them.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of these components. Example: "What is a homogenous substance?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the text. For instance, "How does Mendeleev define non-homogeneous substances in comparison to homogeneous ones?"
Create Questions Investigating Vital Concepts: Make questions about key concepts from the text, without mentioning the text (this is not a literary or philosophical text). A vital concept is the difference between homogenous and non-homogenous substances; since this is a scientific text, I will ask "What are the properties of a homogeneous substance?"
Make a Question that Naturally Complements the Text's Focus: The text by Demitry Mendeleev focuses on homogeneous and non-homogenous substances (and their examples), and another question on this subject would be suitable. Since I have already asked about the properties of homogenous substances, I will ask a question about non-homogenous ones: "What are some properties of non-homogenous substances?\""""

    plan_xenophon = """Identify Key Topics: This excerpt from Xenophon's Symposium presents themes of gender roles, the value of labor, and the nature of beauty.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of specific information in this text. Example: "What did Socrates compare perfumes to?"
Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the excerpt. For instance, "How does Socrates view women's use of perfume?"
Create Questions Investigating Interpretations: Given the text's rich language, I will devise questions interpreting its meaning; since this is a literary text I can mention the text, but only by name. Example: "What does Socrates mean when he says that perfumes make distinctions vanish?"
Make a Question that Naturally Complements the Text's Focus: The text by Xenophon focuses on Socrates' views and another question on this subject would be suitable. The text focuses on Socrates' opinions, but enough questions have already been asked about that subject to cover it completely. That leaves the role of women as a subject for questions. Since the text mentions the women present at the symposium, I can ask "Who are the two brides mentioned in Xenophon's Symposium?\""""

    text_xenophon = """After the girl had played to them upon the flute, and then the boy in
turn upon the harp, and both performers, as it would appear, had set the
hearts of every one rejoicing, Socrates turned to Callias:

A feast, upon my word, O princeliest entertainer! (2) Was it not enough
to set before your guests a faultless dinner, but you must feast our
eyes and ears on sights and sounds the most delicious?

 (2) Lit. "in consummate style."

To which the host: And that reminds me, a supply of unguents might not
be amiss; (3) what say you? Shall we feast on perfumes also? (4)

 (3) Lit. "suppose I tell the servant to bring in some perfumes, so
    that we may further feast on fragrance..." Cf. Theophr. "Char."
    vii. 6 (Jebb ad loc.)

 (4) See Athen. xv. 686.

No, I protest (the other answered). Scents resemble clothes. One dress
is beautiful on man and one on woman; and so with fragrance: what
becomes the woman, ill becomes the man. Did ever man anoint himself with
oil of myrrh to please his fellow? Women, and especially young women
(like our two friends' brides, Niceratus' and Critobulus'), need no
perfume, being but compounds themselves of fragrance. (5) No, sweeter
than any perfume else to women is good olive-oil, suggestive of the
training-school: (6) sweet if present, and when absent longed for. And
why? Distinctions vanish with the use of perfumes. The freeman and the
slave have forthwith both alike one odour. But the scents derived from
toils--those toils which every free man loves (7)--need customary habit
first, and time's distillery, if they are to be sweet with freedom's
breath, at last."""

    result2 = generate_questions((text2, "Principles of chemistry"), plan5, logic_llm)
    result3 = generate_questions(
        (text_kant, "The Critique of Pure Reason, by Immanuel Kant"), plan4, logic_llm
    )
    result3 = generate_questions(
        (text_xenophon, "The Symposium, by Xenophon"), plan_xenophon, logic_llm
    )
    print("GENERATION TEST2:\n\n-------------------\n\n", result2)
