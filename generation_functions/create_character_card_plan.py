import re
from .create_character_card_plan_grammar import character_card_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from special_instructions import special_instructions

def create_character_card_plan(qatuple,logic_llm,assistant_mode=False):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # It's way more willing to use different time periods than I expected, which is cool.

    instructions_string = special_instructions()
    
    # WIP
    if assistant_mode:
          return "" # Just an empty string since this gets skipped in assistant mode
    else:     
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

Text details:  \"\"\"Introduction to Mathematics, by Jane Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

Question: 
\"\"\"
How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
\"\"\"
Answer: 
\"\"\"
The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.
\"\"\"

Special instructions:
The character should be a woman.
The character should be excessively, unrealistically horny and sexual.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a female mathematics instructor with repressed desires at a prestigious university during the 19th century. She's committed to her field and is skilled, but the extremely prim and proper environment, combined with an absurdly busy schedule, has left her unable to get any sexual release for a very long time — to the point of absurdity, where filthy phrases infiltrate her normal conversations. Since the question is abstract and mathematical, it will be difficult to tie them and their answers directly into her character and the special instructions; but her language can still reveal her personality. For instance, while describing linear functions in the question, instead of saying that the graph "ascends" with a positive slope, or "descends" with a negative slope, she might instead say it "grows" and "shrinks" (a subtle reference to male genitals). Instead of saying a slope is "steep" she might call it "erect" instead. Wherever clever analogies can't be tied into the questions, she'll simply say or do horny things before or after answering the question, such as blushing hard, fiddling with her hair (preening), or even propositioning people she is speaking to out of the blue. 

# Input:
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

Question: 
\"\"\"
What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
\"\"\"
Answer: 
\"\"\"
They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.
\"\"\"

Special instructions:
The character should be a young adult.
The character should be narcissistic.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the text, Thus Spake Zarathustra, is philosophical and written using very distinct, archaic language, this character will be someone who (just to flex his intellect) uses archaic and flamboyant language just for the hell of it — and is prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answer to the provided question, but who can reveal character depth by doing so.

# Input:
## Question, answer, and text that the character should know:

Text details: \"\"\"Great Construction Projects Throughout History, by John Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Question: 
\"\"\"
How much earth was excavated during the construction of the Panama Canal?
\"\"\"
Answer: 
\"\"\"
Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) will tie into the question and setting by being tools he uses to whip the workers at the canal into shape. Since the question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the question is of a straight-up, factual nature, it can't really tie into this character's personality, but it can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

# Input:
## Question, answer, and text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
{qatuple[2]}
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"{qatuple[3]}\"\"\"

Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\"

Special instructions:
{instructions_string}

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