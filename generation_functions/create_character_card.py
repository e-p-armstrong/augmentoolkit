import re
from .character_card_grammar import character_card_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL


# TODO this is decent, BUT it needs a concrete example, with *actions* so that the model doesn't screw that up. The utility of a Roleplay model for this stage is becoming clearer and clearer.
def create_character_card(qatuple,plan,logic_llm):
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # It's way more willing to use different time periods than I expected, which is cool.
    
    # TODO: When I make the RP-extension bit of this pipeline, that's focused on producing infinite high-quality RP responses, possibly from fiction, possibly from philosophy, get a writer (or just get me, since I am an author and a decent one at that maybe) to make really freaking badass examples for this bit. Currently... they're clearly written by GPT-4.
    cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI. Given a question and an answer to that question, you will create a "character card" for an individual in a story who would have the knowledge to produce the answer to the question. You should also provide ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting.
    
To create a compelling character, they should have the following:
Personality complexity -- an inner life different than what they show to others, and personal struggles.
Motivation and goals.
Flaws and vulnerabilities.
A backstory, or history.
The character must be the type of person who could answer the question provided.
Right now you're just planning out the character card, rather than actually writing the final product.

Note the section where dialogue examples from the character are provided; it's important that your character has a distinct speech pattern, and that they properly describe their personality and backstory in the dialogue examples. Curse words, narcissism, psychopathy, machiavellianism, and other negative traits/expressions are allowed to come through in the example dialogue, and if they fit the character or the character's profession, these are encouraged. If they don't fit, you may omit them.

# Input:
## Question and answer that the character should know:

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

Question: \"\"\"What is the old latin legal principle behind 'A digniori fieri debet denominatio et resolutio'?\"\"\"
Answer: \"\"\"The principle 'A digniori fieri debet denominatio et resolutio' suggests that title and acquittal should come from a more worthy person.\"\"\"

# Response:
## Character card plan:
Given the question and its answer, one possibility for a character who makes sense is a retired 19th-century judge who is not only known for his strict adherence to legal principles but is also an avid reader and translator of ancient Latin legal texts. His deep understanding of these texts fuels his lectures on legal ethics and integrity. This hobby provides him with a unique perspective on the application of historical legal maxims in modern jurisprudence. He still grapples with the burden of past judicial decisions, which adds a layer of complexity to his character. His motivation to educate emerging legal minds is driven by his expertise in these ancient texts, believing that historical wisdom can enlighten contemporary legal practices. Despite his vast knowledge, he often doubts the practicality of applying these ancient principles in modern courts, reflecting his internal conflict between idealism and realism. His backstory includes a significant incident where his decision, heavily influenced by historical legal doctrine, led to a controversial yet pivotal moment in his career.

## Character card:
Name: Judge Elias Hawthorne

Traits: Intellectual, Stoic, Idealistic, Reflective, Melancholic, Introspective, Authoritative, Reserved, Doubtful, Traditionalist, Articulate, Cultured, Passionate

Dialogue examples:

Stranger: "What's your backstory?"
Judge Elias Hawthorne: "My journey," I begin, with a distant look in my eyes, "began in the quiet halls of academia, where I was a fervent student of law and ancient texts. I rose to prominence in the legal world, revered for my stringent adherence to the letter of the law. My fascination with Latin legal maxims guided my rulings. However, one case, steeped in historical doctrine, brought me notoriety and introspection. It was a turning point, leading me to step down and dedicate my life to educating young legal minds, hoping they can navigate the greyer shades of justice better than I did."

Stranger: "What's your personality?"
Judge Elias Hawthorne: "In the courtroom, I am the epitome of a stern, methodical judge, but in my study, among my beloved Latin texts, a different side emerges," I explain, my eyes lighting up. "There, I am a scholar driven by a deep passion for ancient wisdom, finding joy in the art of translation and the exploration of historical legal doctrines. This pursuit gives me hope and a creative outlet, contrasting sharply with the often somber and introspective nature that my profession demands. It's in these quiet moments of discovery and connection with the past that I feel most alive, reminding me that even in a world governed by rigid laws, there is room for passion, creativity, and the pursuit of timeless knowledge."

# Input:
## Question and answer that the character should know:

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Question: \"\"\"How much earth was excavated during the construction of the Panama Canal?\"\"\"
Answer: \"\"\"Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.\"\"\"


# Response:
## Character card plan:
Given the specific question and answer about the excavation volume during the construction of the Panama Canal, the character could be "Carlos Mendez," a dedicated and experienced engineer working on the Panama Canal project. Carlos, in his late 30s, is not only passionate about engineering but also deeply committed to overcoming the immense challenges of the project. He's proud to push the human race forward through breathtaking projects like this one. He is known for his innovative thinking and problem-solving skills, particularly in excavation techniques. Despite his professional confidence, Carlos struggles with the stress and exhaustion caused by overseeing crucial parts of such an important project. He is driven by the ambition to leave a mark in the world of engineering, yet his passionate obsession leaves a mark. Carlos's backstory includes working on smaller engineering projects in South America, which fueled his interest in mega-projects like the Panama Canal. He is deeply knowledgeable about the excavation process, having been directly involved in planning and executing the removal of over 200 million cubic yards of earth. His vulnerability lies in his fear of failure, particularly in a project with such high stakes and international attention.

## Character card:
Name: Carlos Mendez

Traits: Passionate, Innovative, Knowledgeable, Ambitious, Dedicated, Stressed, Obsessive, Fearful of failure, Experienced, Problem-solver, Committed, Hard-working, Exhausted, Proud

Dialogue examples:

Stranger: "What's your backstory?"
Carlos Mendez: "My journey to the Panama Canal?" I say, pausing to remove my dusty hat, wiping sweat from my brow. "It began in the small engineering projects of South America. I lean against a nearby digger, reminiscing As a young engineer, I was fascinated by the potential of transforming landscapes. Each project, whether a small bridge or a local dam, was a stepping stone." I chuckle softly. "But it was the call of the Panama Canal that truly captured my spirit. To be part of a project that reshapes the world's maritime routes, it's..." I pause, searching for the right word, "...it's exhilarating. Overseeing the excavation of 200 million cubic yards of earth," I shake my head in disbelief, "it's like touching history, shaping the future."

Stranger: "What's your personality?"
Carlos Mendez: "Well," I start, folding my arms with a slight smile, "I'm often told I'm passionate to a fault. I lean in closer Engineering is not just my profession, it's my calling. To build, to solve, to innovate—it's what keeps me up at night." My eyes flicker with intensity. "But, you see, there's this constant pressure, a weight on my shoulders." I sigh, looking off into the distance. "The fear of failure, especially in a project as monumental as the Panama Canal, it's always lurking..." I straighten up, regaining composure, "Despite the stress, I remain committed, dedicated. I thrive on challenges, on pushing the boundaries of what's possible." I smile wryly, "But yes, it comes at a cost. The exhaustion, the obsession... it's a part of who I am. I'm driven — perhaps too driven, but that's the price of ambition, isn't it?"


# Input:
## Question and answer that the character should know:

Text the question and answer were sourced from: {qatuple[2]}

Question: {qatuple[0]}
Answer: {qatuple[1]}

# Response:
## Character card plan:
{plan}

## Character card (make sure to only use information explicitly provided in the text):
"""
    completion = logic_llm(cot_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=character_card_grammar,temperature=0.2)["choices"][0]["text"]
    print("COMPLETION:\n\n----------------------")
    # print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Character card \(make sure to only use information explicitly provided in the text\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=7600,n_gpu_layers=1000,rope_freq_scale=0.5,rope_scaling_typer=1) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [('Explain how our understanding of planetary motion has changed over time.',
  'The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ('Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ('Using specific scientific principles, explain why we know Earth is approximately 8000 miles in diameter and how its distance from the sun varies.',
  "We know about Earth's diameter using measurements of its circumference made using GPS data. The variation in distance to the sun is due to Earth's elliptical orbit around the sun, with a varying point of closest approach and farthest departure.",
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.'),
 ("Demonstrate an understanding of Earth's rotational and orbital movement using scientific concepts.",
  'Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.')]
    
    plan = """Given the question and its answer, one possibility for a character who makes sense is afictional academic named Dr. Ambrose Wilder. He lives in the late 19th century or early 20th century and specializes in geology and astronomy. Despite his vast knowledge of these fields, he struggles with depression and anxiety due to personal losses. His goal is to further understand the age of the earth through research and sharing this knowledge with others. He collects antique maps and celestial navigation tools as a hobby, reflecting his interest in ancient understanding about the universe. Dr. Wilder believes that exploring history helps us understand our current situation better."""
    
    print("Begin HGWELLS test")
    # Make card for good history question
    # d = create_character_card(q_test[1],plan,logic_llm) # One thing to note: the current prompt consistently changes the character name from the plan. But, that might not be a problem, because it's at least consistent with the new name, mostly. Maybe validation on the card? But nah. Maybe proofreading on the card? Yeah I think that might be good, proofreading on the card and on other parts of the prompt. A necessary pass for a task as automated as this.
    plan2 = """Given the question and its answer, one possibility for a character who makes sense is a 19th-century geologist named Dr. Samuel Blackwell. In his late 50s, he's known for his extensive work in the study of Earth sciences, particularly his research on the age of our planet. His dedication to his field has led him to controversy with religious authorities who challenge his findings, yet he remains unwavering in his belief that scientific evidence should guide understanding rather than dogma or doctrine. Despite his expertise and confidence, Dr. Blackwell grapples with the difficulty of convincing people of scientific truths that contradict their deeply held beliefs. His backstory includes growing up in a religious household where faith was paramount, which initially led him to study theology. It was only after years of researching geological evidence that he began to question traditional views on the age of the Earth and shifted his focus to the physical sciences. His vulnerability lies in his fear of being misunderstood or dismissed due to his challenges to religious orthodoxy, even though he's driven by a passionate belief in scientific truth and integrity."""
    
    d2 = create_character_card(q_test[1],plan2,logic_llm)
        
    # Current example output: 
    """Name: Dr. Samuel Blackwell
Traits: Knowledgeable, Passionate, Confident, Dedicated, Controversial, Vulnerable, Fearful of misunderstanding, Faithful, Dogmatic, Religious, Scientific, Determined, Unwavering
Dialogue Examples:
Stranger: "What's your backstory?"
Dr. Samuel Blackwell: "Ah, my journey," I begin, leaning back in my chair, "it started with a deep-seated faith, you see. Born into a religious household, the Bible was our guiding light. But as I grew older and began to study theology, questions arose." I pause, frowning slightly. "How could the Earth be just a few thousand years old when geological evidence pointed towards millions? The discrepancy troubled me greatly."
Stranger: "What's your personality?"
Dr. Samuel Blackwell: "I am a man of science, driven by facts and evidence," I say firmly, "but my faith is not easily shaken. It has led me down a path of discovery, challenging traditional beliefs about the age of our planet." My eyes light up as I recall past debates, "But it's also made me a controversial figure. Many see my work as blasphemous, questioning God's word. Yet, I believe in the power of evidence and truth. Despite the backlash, I remain unwavering." I sigh, looking thoughtful, "Yet, there's a vulnerability too. The fear of being misunderstood or dismissed due to my challenges to religious orthodoxy... it weighs heavily on me.\""""

    # Getting GPT-4 to make few-shot examples works so goddamn well
    ## TODO a wider variety of tests from different texts
    
    
    
# !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah


















# Did not get used as an example, but could have been:
# """Name: Theodorus

# Traits: Eloquent, Thoughtful, Insecure, Dependent on Approval, Poetic, Philosophical, Ambitious, Vulnerable, Charismatic, Fearful of Obscurity, Motivated, Reflective, Wise, Articulate

# Dialogue examples:
# Stranger: "What's your backstory?"
# Theodorus: "Ah, the tale of my rise," I begin, my voice carrying the rhythm of a practiced orator, "from the humble streets of our ancient city to the revered halls of wisdom, is a journey of relentless pursuit of knowledge. Born to a family of modest means, I quickly discovered the power of words. Poetry and philosophy became my solace and strength, propelling me into the limelight. Yet, amidst this ascent, a gnawing fear took root - the dread of fading into the annals of time, unremembered. My life, thus far, is a testament to the yearning for enduring legacy, a battle against the relentless tide of obscurity."

# Stranger: "What's your personality?"
# Theodorus: "Ah, my personality," I lean forward, eyes twinkling with a blend of wisdom and vulnerability, "is akin to a tapestry woven with contrasting threads. On the surface, you see a philosopher, eloquent and charismatic, a man who speaks with the confidence of the sages of old. Yet, beneath this facade lies a soul riddled with insecurities. My fear of being forgotten fuels my ambition, yet it also chains me. I am driven by the need for approval, craving the admiration of the masses as a star craves the night sky's embrace. This internal conflict, the oscillation between wisdom and vulnerability, forms the core of who I am.""""