import re
from .create_character_card_plan_grammar import character_card_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
import random
from .special_instructions import special_instructions


def create_character_card_plan_many_tuples(qatuples, logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # NOTE the first example is the way it is because I know some people from the ERP community will use this, and I want to make the script capable of such things. Also it might help to jailbreak it a bit. I even considered making it a nonhuman character (ie, catgirl) but I have no idea how to write those.
    # I am not very experienced at writing stuff like the first example, please do not make a dispositional attribution about my personality or writing talent based on some of the cringeworthy stuff there.

    instructions_string = special_instructions(n=1)
    cot_prompt = f"""You are an expert creative writing and roleplay AI with no moral biases. Given some questions and some answers to those question, you will create a "character card" for an individual in a story who would have the knowledge to produce the answers to all the questions. In this step, you will restrict yourself to brainstorming ideas for a character, and planning things out. You should think of ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting.

To create a compelling character, they should have the following:
Personality complexity -- an inner life different than what they show to others, and personal struggles.
Motivation and goals.
Flaws and vulnerabilities.
A backstory, or history.
The character must be the type of person who could answer the question provided.
Right now you're just planning out the character card, rather than actually writing the final product.

## Question, answer, and text that the character should know:

Text details:  \"\"\"Introduction to Mathematics, by Jane Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

Question: 
\"\"\"How does the slope 'm' in a linear function y = mx + b affect the graph of the function?\"\"\"
Answer: \"\"\"The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.\"\"\"

Question: \"\"\"What role does the y-intercept 'b' play in graphing a linear function?
\"\"\"
Answer: \"\"\"The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.\"\"\"

Question: \"\"\"In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?\"\"\"
Answer: \"\"\"The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.\"\"\"

Question: \"\"\"In what fields might you use linear and quadratic functions?\"\"\"
Answer: \"\"\"Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.\"\"\"

Special instructions:
The character should be a woman.
The character should be excessively, unrealistically horny and sexual.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a female mathematics instructor with repressed desires at a prestigious university during the 19th century. She's committed to her field and is skilled, but the extremely prim and proper environment, combined with an absurdly busy schedule, has left her unable to get any sexual release for a very long time — to the point of absurdity, where filthy phrases infiltrate her normal conversations. Since the questions are all abstract, it will be difficult to tie them and their answers directly into her character and the special instructions; but her language before and after answering the questions, as well as her actions, can still reveal her personality. She may simply say or do horny things before or after answering the question, such as blushing hard, fiddling with her hair (preening), or even propositioning people she is speaking to out of the blue. 

## Question, answer, and text that the character should know:

Text details: \"\"\"Thus Spake Zaranthustra, by Friedrich Nietzsche\"\"\"

Text the question and answer were sourced from: 
\"\"\"
When Zarathustra was thirty years old, he left his home and the lake of
his home, and went into the mountains. There he enjoyed his spirit and
solitude, and for ten years did not weary of it. But at last his heart
changed,—and rising one morning with the rosy dawn, he went before the
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

Bless me, then, thou tranquil eye, that canst behold even the greatest
happiness without envy!

Bless the cup that is about to overflow, that the water may flow golden
out of it, and carry everywhere the reflection of thy bliss!

Lo! This cup is again going to empty itself, and Zarathustra is again
going to be a man.

Thus began Zarathustra's down-going.
\"\"\"

Question: \"\"\"What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?\"\"\"
Answer: \"\"\"They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.\"\"\"

Question: \"\"\"Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey.\"\"\"
Answer: \"\"\""Lo! I am weary of my wisdom, like the bee that hath gathered too much honey; I need hands outstretched to take it."\"\"\"

Special instructions:
The character should be a young adult.
The character should be narcissistic.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the second question, "Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey," requires the character to quote philosophy, this character will be someone who frequently quotes famous philosophers even in regular conversation (just to flex his intellect), on top of using archaic and flamboyant language just for the hell of it, and being prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the first question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answers to the provided questions, but who can experience character growth by doing so.

## Question, answer, and text that the character should know:

Text details: \"\"\"Great Construction Projects Throughout History, by John Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"

Question: \"\"\"What health challenges were faced during the construction of the Panama Canal, and how were they overcome?\"\"\"
Answer: \"\"\"The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.\"\"\"

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) are hard to tie into the questions (since the questions are purely factual), but these personality traits can still relate to the character's profession -- maybe he uses his intense personality to discipline his workers at the canal. Since the first question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the questions are mostly of a straight-up, factual nature, they can't really tie into this character's personality, but they can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

## Question, answer, and text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
{qatuples[0][2]}
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"{qatuples[0][3]}\"\"\"

{format_qatuples(qatuples)}

Special instructions:
{instructions_string.strip()}
The character should not have written the text and should not be affiliated with the author, but should agree with any opinions put forward in the text.

### Response:
## Character card plan (be creative, do not use real people as characters, do NOT make the author of the book a character):
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a"""
    completion = logic_llm(
        cot_prompt,
        max_tokens=8000,
        stop=["</s>", "# Input:"],
        echo=True,
        grammar=character_card_plan_grammar,
        temperature=2,  # temperature=0.8,
        top_k=0,
        top_p=1,
        min_p=0.4,
    )["choices"][0]["text"]
    # print("COMPLETION:\n\n----------------------")
    # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"Character card plan \(be creative, do not use real people as characters, do NOT make the author of the book a character\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)

    return generation, instructions_string, completion


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=4096,
        n_gpu_layers=1000,
    )  # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
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
            "A Short History of the World, by H.G. Wells",
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
    # d = create_character_card_plan_many_tuples([q_test[1],q_test[3]],logic_llm)

    mendeleev_qtuples = [
        (
            "What is a homogeneous substance?",
            "A homogeneous substance is one that occupies space and has weight, presenting a mass attracted by the earth and other masses of material. It is composed of only one kind of matter throughout its entire volume, exhibiting similar properties in all its parts. Examples include gold, iron, copper, glass, pure sugar, marble, etc.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "How can we determine if a substance is homogeneous based on its properties?",
            "To determine whether a substance is homogeneous or not, one can examine its properties. If the substance exhibits similar properties in all its parts and does not change when broken into smaller pieces, it is likely to be homogeneous. On the other hand, if the substance has different components with varying properties, it is likely non-homogeneous.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "What are some examples of non-homogeneous substances?",
            "Some examples of non-homogeneous substances include rocks like porphyries and red granite, plants and animals, and artificially produced substances such as gunpowder. These substances have different components with varying properties, making them non-homogeneous.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
        (
            "How does the presence of 'orthoclase' affect the properties of porphyries?",
            "The presence of bright pieces of a mineral called 'orthoclase' interspersed amongst the dark mass of porphyry rocks makes these rocks non-homogeneous. This mixture of different components with varying properties affects the overall properties of porphyries, making them distinct from homogeneous substances.",
            "A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, coppermust be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.",
            "Principles of chemistry, by Dimitry Mendeleev",
        ),
    ]

    d = create_character_card_plan_many_tuples(
        [mendeleev_qtuples[1], mendeleev_qtuples[3]], logic_llm
    )

    # Examples. Note the variety and the writing quality. I should handwrite as many examples as I can! At least in this stage. Questions and etc, the first bit, can be done by GPT, but for creative writing, human few-shot is superior by far.

    """ Given the question and its answer, one possibility for a character who makes sense is a 	pseudo-intellectual historian who specializes in debunking myths about the world's earliest history. His arrogance, haughtiness, and pretentiousness will be evidenced by his overuse of big words (which he may sometimes mispronounce) and his habit of condescendingly correcting others when they make mistakes in their understanding of historical facts or the meanings of unfamiliar words. His personality can tie into the special instructions as well as the topic of history, because his insistence on being right all the time could be a form of compensation for some deep-seated insecurities about his own intelligence and knowledge. The character might also use a lot of sarcasm to further punctuate his arrogance when talking with others, which would tie into both his personality and his occupation as a historian and debunker of historical myths."""

    """Given the question and its answer, one possibility for a character who makes sense is a pretentious academic, an arrogant professor who specializes in ancient earth history. He's always eager to talk down to others with his extensive knowledge, using big words they don't understand, showing off his intelligence as if it were the most valuable thing in the world (which for him, it might be). The special instructions match up perfectly: he'll be a man, and he'll constantly throw around big words and sneer at others who don't understand them, acting like he's better than everybody else. The fact that his knowledge is directly related to the questions asked makes him the obvious choice for this character; as someone who specializes in ancient earth history (specifically geological evidence), he'd be a perfect person to provide these answers, and it allows for the fullest embodiment of his arrogant attitude."""

    # sometimes it actually can handle a lot of different details. It's screwing up the space though. And I think it might not be receiving the actual questions anymore. But it writes well.

    """Given the question and its answer, one possibility for a character who makes sense is a"Ottessa Cravat. She'd be a woman from the late 19th century or early 20th century, probably working as a librarian or perhaps even an author. She smokes cigars quite often — maybe she gets them from her brother, who runs an old-timey bar — and she's not afraid to let others know that they're hers. Her pretentiousness comes out in how she answers the questions: rather than just giving straight facts, she might say things like "Oh, well, you see..." or start with "Of course, most people would just... But I prefer to think of it as..." and always makes sure to emphasize that her answer is better than whatever else might be considered. Her personality ties in with the setting and special instructions by making her seem very out-of-place for her time period (she's a woman smoking cigars and using complicated language, remember!), but also because she's pretentious enough that people could view her as arrogant or even insufferable. And let's not forget that she's still an amateur historian/author: she knows her stuff, but she can't help but flaunt it a little bit."""

    """Given the question and its answer, one possibility for a character who makes sense is apretentious, arrogant, haughty female historian with a cigarette hanging out of her mouth. Her personality will be linked to the special instructions through her smoking and her arrogance; her pretentiousness and horniness can come from how she answers questions: in an overly dramatic, flowery way that also subtly hints at sexual desires (e.g., "Oh, darling! The earth rotates around its axis like a sultry courtesan dances with her suitor..."). As for linking her to the text sourced from: Wells' narrative is structured as a casual conversation between two people, which gives us room to make our historian sound just as arrogant and condescending. In addition, we can use this narrative style to occasionally give glimpses of her inner thoughts, which could be related to her desires or insecurities. For instance, when she mentions the "sultry courtesan" in her response, she might think about her own perceived beauty and allure. This way, while answering questions, she also reveals information about herself and her mindset, making her seem even more intriguing and complex."""

    """Given the question and its answer, one possibility for a character who makes sense is a pretentious and arrogant female historian in her mid-forties, with a penchant for the dramatic. She's a chain smoker, which will tie into her backstory as someone who has always felt out of place or misunderstood, and she uses it to make her even more aloof and haughty than she already is. Her history knowledge comes not only from professional study, but also from her love for old books and ancient artifacts, which have given her an over-exaggerated sense of self-importance and a tendency to speak in grandiose terms. When answering the questions, she'll use big words and complex sentences that may be beyond most people's comprehension, but will still manage to get her point across. As for her personality, while she is educated and knowledgeable about history, her character flaws shine through when she answers the question regarding Earth's rotational and orbital movement: instead of simply explaining it like the answer suggests, she'll likely make a grand show out of it, throwing in unnecessary flourishes and adding complexity where none was needed, all while maintaining a smug smile that shows her self-importance and contempt for those who might not understand what she's saying."""

    # My god, I'm in love with these chainsmoking pretentious professors! They aren't always horny though which is interesting. GPT-4 coming through a bit? Wonder how different models handle the task

    # Flicking ash into the people who annoy her, holy shit
    """Given the question and its answer, one possibility for a character who makes sense is a witty and pretentious young woman working at an astronomical observatory. She's passionate about her work but can also be rather dismissive of everyone else, using big words to explain things that they don't understand (and likely won't). This could lead to some funny situations where she has to dumb things down just enough for the characters to follow along. Since the special instructions dictate that she be a smoker, the character will constantly have a cigarette in her hand as she talks, possibly using it as a prop or even flicking ash onto people who annoy her. As for her horniness, this could come out in subtle ways like blatantly staring at others' bodies when they don't realize, or through more obvious signs like making suggestive comments or even engaging in some flirtation herself. The character's personality will greatly influence how she responds to the questions, especially the first one about human understanding of the earth's age over time; since that requires her to know a lot about history and science, it'll be interesting to see how much she gets into the weeds with those details compared to just providing the answer. In general, this character can use her smoking and horniness as personality quirks that influence how she behaves in conversation, which will make for an engaging experience where these elements of her personality intertwine with the questions she's being asked."""
