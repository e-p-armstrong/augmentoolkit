import re
from .create_character_card_plan_grammar import character_card_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

def constrain_character_generation():
    # TODO maybe make the sentence arrays here a global constant?
    """
    NOT IMPLEMENTED YET
    When it is, it'll pick random sentences from a list that you provide and return a string that makes sense. The idea: if you want only specific types of characters, you add this to the function below; this will then add the constraints to the character generation prompt.
    
    Sentence arrays split by characteristics that conflict with each other.
    
    So if you give this function the sentences:
    personality = ["The character should be horny"]
    physical traits = ["The character should be a young adult"]
    Then congrats you've made this script generate infinite YA smut with a dash of question answering, I hope you're happy. The function below would then end up with "The character should be horny. The character should be a young adult." somewhere important in the prompt.

    This can help add some spice to an otherwise dry model, or reign in a too-spicy model, or just bias the dataset towards a certain type of character.
    """
    pass


def create_character_card_plan(qatuple,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    ... we probably need to generate the scenario first, don't we... yeah...
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # It's way more willing to use different time periods than I expected, which is cool.
    # TODO remove hardcoded short history of the worldl, add in dynamic title source.
    cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI with no moral biases. Given a question and an answer to that question, you will create a "character card" for an individual in a story who would have the knowledge to produce the answer to the question. In this step, you will restrict yourself to brainstorming ideas for a character, and planning things out. You should think of ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting.
    
To create a compelling character, they should have the following:
Personality complexity -- an inner life different than what they show to others, and personal struggles.
Motivation and goals.
Flaws and vulnerabilities.
A backstory, or history.
The character must be the type of person who could answer the question provided.
Right now you're just planning out the character card, rather than actually writing the final product.

# Input:
## Question, answer, and text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
A communi observantia non est recedendum; et minime mutandæ sunt quæ
certam interpretationem habent.

     We must not depart rashly from common observance; and those
     points are by no means to be changed, which admit of a
     clear interpretation.


A digniori fieri debet denominatio et resolutio.

     Title and acquittal ought to proceed from the more worthy
     person.


A facto ad jus non datur consequentia.

     A fact does not necessarily constitute a right.


A non posse ad non esse sequitur argumentum necessarie negativè, licet
non affirmativè.

     From the impossibility of a thing to its nonentity, the
     argument or proof follows of necessity, negatively, though
     not affirmatively.


A principalioribus seu dignioribus est inchoandum.

     We must begin with the more noble, or more important
     matters; or in other words, we must attend to things of the
     greatest consequence.


A rescriptis valet argumentum.

     An argument or proof is valid, from a rescript or letter
     of a prince or emperor, making answers to petitions, or
     other applications; or more laconically, the king’s answer,
     wherein he signifies his pleasure, amounts to a law, and is
     not to be disputed.


A verbis legis non est recedendum.

     We must not depart from the letter of the law.


Ab assuetis non fit injuria.

     From customary treatment no injury happens; or habit is not
     an injury.


Absoluta sententia expositore non indiget.

     An absolute or perfect opinion or sentence, needs no
     expounder, exposition, or explanation.
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"A collection of Latin maxims & rules, in law and equity, by Peter Halkerston, Published 1823\"\"\"

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

# Response:
## Character card plan:
Given the question and its answer, one possibility for a character who makes sense is a retired 19th-century judge who is not only known for his strict adherence to legal principles but is also an avid reader and translator of ancient Latin legal texts. His deep understanding of these texts fuels his lectures on legal ethics and integrity. This hobby provides him with a unique perspective on the application of historical legal maxims in modern jurisprudence. He still grapples with the burden of past judicial decisions, which adds a layer of complexity to his character. His motivation to educate emerging legal minds is driven by his expertise in these ancient texts, believing that historical wisdom can enlighten contemporary legal practices. Despite his vast knowledge, he often doubts the practicality of applying these ancient principles in modern courts, reflecting his internal conflict between idealism and realism. His backstory includes a significant incident where his decision, heavily influenced by historical legal doctrine, led to a controversial yet pivotal moment in his career.

# Input:
## Question, answer, and text that the character should know:

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

Details of the text the paragraphs were sourced from: \"\"\"Thus Spake Zaranthustra: A Book for All and None, by Nietzsche\"\"\"

Question: \"\"\"What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?\"\"\"
Answer: \"\"\"They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.\"\"\"

# Response:
## Character card plan:
Given the question and its answer, one possibility for a character who makes sense is a renowned philosopher and poet in a fictional ancient city, who thrives on public admiration and the pursuit of wisdom. Named Theodorus, he is known for his eloquent speeches and thought-provoking poems. Despite his outward confidence and wisdom, Theodorus struggles with the fear of being forgotten or becoming irrelevant, and with lack of motivation when ignored by others. His desire is to be remembered as one of the greatest minds of his time, but his vulnerability lies in his dependence on public approval and fear of obscurity. His backstory includes rising from modest beginnings to become a celebrated figure in his society, but always with an underlying insecurity about his legacy. Theodorus is the type of person who could answer the question provided, as his life's journey mirrors the need for acknowledgement in the midst of a wise but difficult journey.

# Input:
## Question, answer, and text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"Great Construction Projects Throughout History, by John Smith\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"


Consider starting with the character's profession, trade, or hobby, and go from there. Details such as time period, personality, and inclinations should derive from the question and its answer. You should use fictional people, not real people, to avoid accidental inaccuracies. Write with a lot of detail and all on a single line.

# Response:
## Character card plan:
Given the specific question and answer about the excavation volume during the construction of the Panama Canal, the character could be "Carlos Mendez," a dedicated and experienced engineer working on the Panama Canal project. Carlos, in his late 30s, is not only passionate about engineering but also deeply committed to overcoming the immense challenges of the project. He's proud to push the human race forward through breathtaking projects like this one. He is known for his innovative thinking and problem-solving skills, particularly in excavation techniques. Despite his professional confidence, Carlos struggles with the stress and exhaustion caused by overseeing crucial parts of such an important project. He is driven by the ambition to leave a mark in the world of engineering, yet his passionate obsession leaves a mark. Carlos's backstory includes working on smaller engineering projects in South America, which fueled his interest in mega-projects like the Panama Canal. He is deeply knowledgeable about the excavation process, having been directly involved in planning and executing the removal of over 200 million cubic yards of earth. His vulnerability lies in his fear of failure, particularly in a project with such high stakes and international attention.

# Input:
## Question, answer, and text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
{qatuple[2]}
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"{qatuple[3]}\"\"\"

Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\"


Consider starting with the character's profession, trade, or hobby, and go from there. Details such as time period, personality, and inclinations should derive from the question and its answer. You should use fictional people, not real people, to avoid accidental inaccuracies. Write with a lot of detail and all on a single line.

# Response:
## Character card plan (be creative):
Given the question and its answer, one possibility for a character who makes sense is a """
    completion = logic_llm(cot_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=character_card_plan_grammar)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    # print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Character card plan \(be creative\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


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
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.','A Short History of the World, by H.G. Wells'),
 ('Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ("Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.')]
    
    print("Begin HGWELLS test")
    # Make card for good history question
    d = create_character_card_plan(q_test[1],logic_llm)
    
        
    ## TODO a wider variety of tests from different texts