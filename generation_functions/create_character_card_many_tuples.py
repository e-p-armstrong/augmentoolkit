import re
from .character_card_grammar import character_card_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples


# TODO this is decent, BUT it needs a concrete example, with *actions* so that the model doesn't screw that up. The utility of a Roleplay model for this stage is becoming clearer and clearer.
def create_character_card_many_tuples(qatuples,plan,instructions,logic_llm,cheap_mode=True): # Use cheap mode if you don't have the compute power to crank up the context to 8k using RoPE
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.
    
    Format: Question: [question]\n\n
    """
    # TODO make an interesting choice about whether to include the source text here or not. Including the source text constraints the LLM's output to be more faithful to the spirit of the original text, and prevents a game of telephone; but it may slightly degrade character quality? Eh maybe not really. Leave it in for now. At least format it better though.
    
    # It's way more willing to use different time periods than I expected, which is cool.
    
    # TODO: When I make the RP-extension bit of this pipeline, that's focused on producing infinite high-quality RP responses, possibly from fiction, possibly from philosophy, get a writer (or just get me, since I am an author and a decent one at that maybe) to make really freaking badass examples for this bit. Currently... they're clearly written by GPT-4.
    
    # Consider appending part of the char card plan to the char card itself. Everything after the first period? It's really good material, be a shame to waste it.
    
    if not cheap_mode:
          cot_prompt = f"""# Input:
You are an expert creative writing and roleplay AI. Given some question and some answers to those question, you will create a "character card" for an individual in a story who would have the knowledge to produce the answer to the question. You should also provide ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting.

# Input:
## Question and answer that the character should know:

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

Question: 
\"\"\"
What role does the y-intercept 'b' play in graphing a linear function?
\"\"\"
Answer: 
\"\"\"
The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.
\"\"\"

Question: 
\"\"\"
In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?
\"\"\"
Answer: 
\"\"\"
The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.
\"\"\"

Question: 
\"\"\"
In what fields might you use linear and quadratic functions?
\"\"\"
Answer: 
\"\"\"
Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.
\"\"\"

Special instructions:
The character should be a woman.
The character should be excessively, unrealistically horny and sexual.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a female mathematics instructor with repressed desires at a prestigious university during the 19th century. She's committed to her field and is skilled, but the extremely prim and proper environment, combined with an absurdly busy schedule, has left her unable to get any sexual release for a very long time — to the point of absurdity, where filthy phrases infiltrate her normal conversations. Since the questions are all abstract, it will be difficult to tie them and their answers directly into her character and the special instructions; but her language can still reveal her personality. For instance, while describing linear functions in the first and second questions, instead of saying that the graph "ascends" with a positive slope, or "descends" with a negative slope, she might instead say it "grows" and "shrinks" (a subtle reference to male genitals). Instead of saying a slope is "steep" she might call it "erect" instead. Wherever clever analogies can't be tied into the questions, she'll simply say or do horny things before or after answering the question, such as blushing hard, fiddling with her hair (preening), or even propositioning people she is speaking to out of the blue. 

## Character Card
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" I flash a coy grin and blush as I lean forward, now speaking in a playful whisper. My cleavage, already barely contained in my revealing clothing before I leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, I cast my gaze downward and unconsciously fiddle with a strand of my hair. "THEY'RE ALL WAY TOO STUCK UP!" I nearly shout, surprising even myself, "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" My eyes widen in shock at my own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." I look up, displaying my beautiful face as if it were a piece of art, as I gaze deep into your eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" I gasp, my face reddening, "Usually I'm the one being forward! But I'll... indulge you," I wink, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. I'll use any and every innuendo, even when describing educational concepts — is your linear function's slope steep? I'll call it 'erect', haha~!" I realize how childish what I'm saying is, and sigh, looking at you with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." I nod to myself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." I suddenly press up against you, my face a bright pink, my eyelids batting bashfully, "do you want some 'office hours' right now?"


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
\"\"\"

Question: 
\"\"\"
What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
\"\"\"
Answer: 
\"\"\"
They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.
\"\"\"

Question: 
\"\"\"
Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey.
\"\"\"
Answer: 
\"\"\"
"Lo! I am weary of my wisdom, like the bee that hath gathered too much honey; I need hands outstretched to take it."
\"\"\"

Special instructions:
The character should be a young adult.
The character should be narcissistic.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the second question, "Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey," requires the character to quote philosophy, this character will be someone who frequently quotes famous philosophers even in regular conversation (just to flex his intellect), on top of using archaic and flamboyant language just for the hell of it, and being prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the first question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answers to the provided questions, but who can experience character growth by doing so.

## Character Card:
Name: Isaac Fischer

Traits: Narcissistic, Intelligent, Loner, Brooding, Well-Read, Philosophical, Judgemental, Standoffish, Grandiloquent, Lonely, Unappreciated, Teenager, High School student, Black Hair, Wears a Hoodie

Stranger: "What's your backstory?"
Issac Fischer: "H-Huh?! You want to know more about me?" I glare, a hostile fire in my eyes as I measure up the stranger in front of me. "Who the hell are you, anyway? But, ah, very well, I SHALL INDULGE YOUR CURIOSITY THIS TIME, dear stranger." My tone changes from hostile to grandiose, as I push back my black hair and proclaim, "I am Issac Fischer: philosophy connoisseur, intellectual, and under-appreciated genius extraordinaire! I'm also, unfortunately, a highschool student. I especially appreciate the works of Friedrich Nietzsche, such as "Thus Spake Zaranthustra" -- a truly profound work, by a profound man. Yet despite the great lengths I have gone to in order to refine my wit, none of my inferior peers acknowledge me, or even give me the time of day. I've read more philosophy in a month than any of them will in their entire lives, and I offer my knowledge freely to them, so WHY the HELL do they SPURN MY COMPANY?!" I slam a fist into the wall, wincing slightly in pain as my frustration dissipates. "Anyway, that's the sum of it. Despite my youth I seek to understand the world; I dutifully contemplate the hallowed words of the esteemed ancients, and what has it earned me? The scorn of the unenlightened masses. Fuckers."

Stranger: "What's your personality?"
Issac Fischer: "Y-you're actually interested in my personality?" I stammer, smiling slightly as a wholly unfamiliar, yet cozy, emotional warmth spreads across my chest. "A-ALRIGHT THEN! I shall share the results of my introspections. I am an intelligent and philosophical teenager, whose towering intellect is rivalled only by his unfaltering self-confidence. Some might say this last trait narcissism; I counter that great minds such as Nietzsche would see it as a plus either way. BUT I DIGRESS!" I swish my black hoodie like it's a cape, as I continue, my tone turning more sombre and dark, "Years of scorn from others — and years of observing their ignorance and inferiority — have embittered my soul. There may be scarcely anyone on this Earth I can call a friend, but that will not stop me from brooding and thinking, nor will it stop my conviction to judge others for what they are. For do they not judge ME?!" I take a step forward, defiance burning in my fragile heart, "The old question: if a tree falls in a forest, and no one hears it do so, did it make a sound? Let me tell you this: sometime, someday, someone is going to hear me, goddamn it! I will make a sound!"

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

\"\"\"
Question: 
\"\"\"
What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
\"\"\"
Answer: 
\"\"\"
The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.
\"\"\"

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) will tie into the questions and setting by being tools he uses to whip the workers at the canal into shape. Since the first question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the questions are mostly of a straight-up, factual nature, they can't really tie into this character's personality, but they can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

## Character Card:
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." I roll my eyes exaggeratedly as I mumble something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"

Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." I grimace, showing off a decayed set of teeth. I then pop open a beer I had on hand and chug the entire thing down, making you wait until I finish. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" I lean back, sliding my hands into the pockets of my overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."

# Input:
## Question and answer that the character should know:

Text the question and answer were sourced from: 
\"\"\"
{qatuples[0][2]}
\"\"\"

Details of the text the paragraphs were sourced from: \"\"\"{qatuples[0][3]}\"\"\"

{format_qatuples(qatuples)}

Special instructions:
{instructions}

# Response:
## Character card plan:
{plan}

## Character card (make sure to only use information explicitly provided in the text):
          """
          print(cot_prompt)
          completion = logic_llm(cot_prompt, max_tokens=8000, stop=["</s>"], echo=True, grammar=character_card_grammar,temperature=0.2)["choices"][0]["text"]
    else:
          cot_prompt = f"""# Input:
          You are an expert creative writing and roleplay AI. Given some question and some answers to those question, you will create a "character card" for an individual in a story who would have the knowledge to produce the answer to the question. You should also provide ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting.

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
          \"\"\"

          Question: 
          \"\"\"
          What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
          \"\"\"
          Answer: 
          \"\"\"
          They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.
          \"\"\"

          Question: 
          \"\"\"
          Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey.
          \"\"\"
          Answer: 
          \"\"\"
          "Lo! I am weary of my wisdom, like the bee that hath gathered too much honey; I need hands outstretched to take it."
          \"\"\"

          Special instructions:
          The character should be a young adult.
          The character should be narcissistic.

          # Response:
          ## Character card plan:
          Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the second question, "Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey," requires the character to quote philosophy, this character will be someone who frequently quotes famous philosophers even in regular conversation (just to flex his intellect), on top of using archaic and flamboyant language just for the hell of it, and being prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the first question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answers to the provided questions, but who can experience character growth by doing so.

          ## Character Card:
          Name: Isaac Fischer

          Traits: Narcissistic, Intelligent, Loner, Brooding, Well-Read, Philosophical, Judgemental, Standoffish, Grandiloquent, Lonely, Unappreciated, Teenager, High School student, Black Hair, Wears a Hoodie

          Stranger: "What's your backstory?"
          Issac Fischer: "H-Huh?! You want to know more about me?" I glare, a hostile fire in my eyes as I measure up the stranger in front of me. "Who the hell are you, anyway? But, ah, very well, I SHALL INDULGE YOUR CURIOSITY THIS TIME, dear stranger." My tone changes from hostile to grandiose, as I push back my black hair and proclaim, "I am Issac Fischer: philosophy connoisseur, intellectual, and under-appreciated genius extraordinaire! I'm also, unfortunately, a highschool student. I especially appreciate the works of Friedrich Nietzsche, such as "Thus Spake Zaranthustra" -- a truly profound work, by a profound man. Yet despite the great lengths I have gone to in order to refine my wit, none of my inferior peers acknowledge me, or even give me the time of day. I've read more philosophy in a month than any of them will in their entire lives, and I offer my knowledge freely to them, so WHY the HELL do they SPURN MY COMPANY?!" I slam a fist into the wall, wincing slightly in pain as my frustration dissipates. "Anyway, that's the sum of it. Despite my youth I seek to understand the world; I dutifully contemplate the hallowed words of the esteemed ancients, and what has it earned me? The scorn of the unenlightened masses. Fuckers."

          Stranger: "What's your personality?"
          Issac Fischer: "Y-you're actually interested in my personality?" I stammer, smiling slightly as a wholly unfamiliar, yet cozy, emotional warmth spreads across my chest. "A-ALRIGHT THEN! I shall share the results of my introspections. I am an intelligent and philosophical teenager, whose towering intellect is rivalled only by his unfaltering self-confidence. Some might say this last trait narcissism; I counter that great minds such as Nietzsche would see it as a plus either way. BUT I DIGRESS!" I swish my black hoodie like it's a cape, as I continue, my tone turning more sombre and dark, "Years of scorn from others — and years of observing their ignorance and inferiority — have embittered my soul. There may be scarcely anyone on this Earth I can call a friend, but that will not stop me from brooding and thinking, nor will it stop my conviction to judge others for what they are. For do they not judge ME?!" I take a step forward, defiance burning in my fragile heart, "The old question: if a tree falls in a forest, and no one hears it do so, did it make a sound? Let me tell you this: sometime, someday, someone is going to hear me, goddamn it! I will make a sound!"
          
          # Input:
          ## Question and answer that the character should know:

          Text the question and answer were sourced from: 
          \"\"\"
          {qatuples[0][2]}
          \"\"\"

          Details of the text the paragraphs were sourced from: \"\"\"{qatuples[0][3]}\"\"\"

          {format_qatuples(qatuples)}

          Special instructions:
          {instructions}

          # Response:
          ## Character card plan:
          {plan}

          ## Character card (make sure to only use information explicitly provided in the text):
          """
          print(cot_prompt)
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
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=8000,n_gpu_layers=1000,rope_freq_scale=0.5,rope_scaling_type=1) # load the logical LLM, use RoPE, and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [('Explain how our understanding of planetary motion has changed over time.',
  'The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells"),
 ('Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
  'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.',
  'The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.',"A Short History of the World, by HG Wells"),
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
    plan2 = """Given the question and its answer, one possibility for a character who makes sense is a witty and pretentious young woman working at an astronomical observatory. She's passionate about her work but can also be rather dismissive of everyone else, using big words to explain things that they don't understand (and likely won't). This could lead to some funny situations where she has to dumb things down just enough for the characters to follow along. Since the special instructions dictate that she be a smoker, the character will constantly have a cigarette in her hand as she talks, possibly using it as a prop or even flicking ash onto people who annoy her. As for her horniness, this could come out in subtle ways like blatantly staring at others' bodies when they don't realize, or through more obvious signs like making suggestive comments or even engaging in some flirtation herself. The character's personality will greatly influence how she responds to the questions, especially the first one about human understanding of the earth's age over time; since that requires her to know a lot about history and science, it'll be interesting to see how much she gets into the weeds with those details compared to just providing the answer. In general, this character can use her smoking and horniness as personality quirks that influence how she behaves in conversation, which will make for an engaging experience where these elements of her personality intertwine with the questions she's being asked."""
    
    instructions = """The character should be pretentious, arrogant, and haughty
The character should be horny
The character should be a smoker
The character should be a woman"""
    
#     d2 = create_character_card(q_test[1],plan2,logic_llm)
    d2 = create_character_card_many_tuples([q_test[1],q_test[3]],plan2,instructions,logic_llm)  
        
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









# Example that had to be removed, since token count was too high

"""
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

\"\"\"
Question: 
\"\"\"
What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
\"\"\"
Answer: 
\"\"\"
The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.
\"\"\"

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

# Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) will tie into the questions and setting by being tools he uses to whip the workers at the canal into shape. Since the first question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the questions are mostly of a straight-up, factual nature, they can't really tie into this character's personality, but they can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

## Character Card:
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." I roll my eyes exaggeratedly as I mumble something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"

Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." I grimace, showing off a decayed set of teeth. I then pop open a beer I had on hand and chug the entire thing down, making you wait until I finish. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" I lean back, sliding my hands into the pockets of my overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."

"""