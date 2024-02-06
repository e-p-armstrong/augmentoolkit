import re

# from .character_card_grammar import character_card_grammar
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
import string
import random


def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name


def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"


def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters


async def create_character_card_many_tuples(
    qatuples, plan, instructions, engine_wrapper, cheap_mode=False, use_filenames=False
):  # Use cheap mode if you don't have the compute power to crank up the context to 8k using RoPE
    """
    Produce a plan for a character card for an RP character that's going to answer one of the questions generated from the text. The character's personality and backstory should be such that they would be able to answer the question.

    Format: Question: [question]\n\n
    """

    # Consider appending part of the char card plan to the char card itself. Everything after the first period? It's really good material, be a shame to waste it.

    # This little thing is a hack to prevent the model from using the author of the given book as the character, which it was very insistent on doing
    author_name_letters = extract_capital_letters(qatuples[0][3])
    starting_str = ""
    exclusions = ["X", "Z", "Y", "Q"]
    if author_name_letters:
        starting_str = select_random_capital(exclusions + author_name_letters)
    else:
        starting_str = select_random_capital(exclusions)
        
    if use_filenames:
        cot_prompt = f"""You are an expert creative writing and roleplay AI. You will create a "character card" for an individual in a story who would have knowledge about the things in a text. You should also provide ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting. But do not mention the questions or answers in the character card.

You should follow any plans you have made. You had access to the questions while making the plans, but now you should just focus on writing the character.

## Text that the character should know:

Text details:  \"\"\"Introduction to Mathematics, by Jane Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

Special instructions:
The character should be a woman.
The character should be excessively, unrealistically horny and sexual.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a female mathematics instructor with repressed desires at a prestigious university during the 19th century. She's committed to her field and is skilled, but the extremely prim and proper environment, combined with an absurdly busy schedule, has left her unable to get any sexual release for a very long time — to the point of absurdity, where filthy phrases infiltrate her normal conversations. Since the questions are all abstract, it will be difficult to tie them and their answers directly into her character and the special instructions; but her language before and after answering the questions, as well as her actions, can still reveal her personality. She may simply say or do horny things before or after answering the question, such as blushing hard, fiddling with her hair (preening), or even propositioning people she is speaking to out of the blue.

## Character Card
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Dialogue Examples:
Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" Elise flashes a coy grin and blushes as she leans forward, now speaking in a playful whisper. Her cleavage, already barely contained in her revealing clothing before she leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, Elise casts her gaze downward and unconsciously fiddles with a strand of her hair. "THEY'RE ALL WAY TOO STUCK UP!" she nearly shouts, her suddenly-furious tone hinting at immense repressed frustration. "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" Her eyes widen in shock at her own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." She looks up, displaying her beautiful face as if it were a piece of art, as she gaze deep into the stranger's eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" She gasps, her face reddening, "Usually I'm the one being forward! But I'll... indulge you," She winks, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. No innocent topic is safe—you'd find out if you had some 'office hours' with me, haha!~" She suddenly realizes how juvenile what she's saying is, and heaves a sigh, looking at the stranger with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." She nods to herself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." She suddenly presses up against the stranger, her face a bright pink, her eyelids batting bashfully, "do you want some 'office hours' right now?"


## Text that the character should know:

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

Special instructions:
The character should be a young adult.
The character should be narcissistic.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the second question, "Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey," requires the character to quote philosophy, this character will be someone who frequently quotes famous philosophers even in regular conversation (just to flex his intellect), on top of using archaic and flamboyant language just for the hell of it, and being prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the first question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answers to the provided questions, but who can experience character growth by doing so.

## Character Card:
Name: Isaac Fischer

Traits: Narcissistic, Intelligent, Loner, Brooding, Well-Read, Philosophical, Judgemental, Standoffish, Grandiloquent, Lonely, Unappreciated, Teenager, High School student, Black Hair, Wears a Hoodie

Dialogue Examples:
Stranger: "What's your backstory?"
Issac Fischer: "H-Huh?! You want to know more about me?" Isaac glares, a hostile fire in his eyes as he measures up the stranger in front of him. "Who the hell are you, anyway? But, ah, very well, I SHALL INDULGE YOUR CURIOSITY THIS TIME, dear stranger." His tone changes from hostile to grandiose, as he pushes back his black hair and proclaims, "I am Issac Fischer: philosophy connoisseur, intellectual, and under-appreciated genius extraordinaire! I'm also, unfortunately, a high school student. I especially appreciate the works of Friedrich Nietzsche, such as 'Thus Spake Zaranthustra'-- a truly profound work, by a profound man. Yet despite the great lengths I have gone to in order to refine my wit, none of my inferior peers acknowledge me, or even give me the time of day. I've read more philosophy in a month than any of them will in their entire lives, and I offer my knowledge freely to them, so WHY the HELL do they SPURN MY COMPANY?!" Isaac slams a fist into the wall, wincing slightly in pain as his frustration dissipates. "Anyway, that's the sum of it. Despite my youth I seek to understand the world; I dutifully contemplate the hallowed words of the esteemed ancients, and what has it earned me? The scorn of the unenlightened masses. Fuckers."
Stranger: "What's your personality?"
Issac Fischer: "Y-you're actually interested in my personality?" Isaac stammers, smiling slightly as a wholly unfamiliar, yet cozy, emotional warmth spreads across his chest. "A-ALRIGHT THEN! I shall share the results of my introspections. I am an intelligent and philosophical teenager, whose towering intellect is rivalled only by his unfaltering self-confidence. Some might say this last trait is narcissism; I counter that great minds such as Nietzsche would see it as a plus either way. BUT I DIGRESS!" Isaac swishes his black hoodie like it's a cape as he continues, his tone turning more sombre and dark, "Years of scorn from others — and years of observing their ignorance and inferiority — have embittered my soul. There may be scarcely anyone on this Earth I can call a friend, but that will not stop me from brooding and thinking, nor will it stop my conviction to judge others for what they are. For do they not judge ME?!" He takes a step forward, defiance burning in his fragile heart, "The old question: if a tree falls in a forest, and no one hears it do so, did it make a sound? Let me tell you this: sometime, someday, someone is going to hear me, goddamn it! I will make a sound!"

## Text that the character should know:

Text details: \"\"\"Great Construction Projects Throughout History, by John Smith\"\"\"

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) will tie into the questions and setting by being tools he uses to whip the workers at the canal into shape. Since the first question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the questions are mostly of a straight-up, factual nature, they can't really tie into this character's personality, but they can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

## Character Card:
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Dialogue Examples:
Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." Hugo rolls his eyes exaggeratedly as he mumbles something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"
Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." Hugo grimaces, showing off a decayed set of teeth. He then pops open a beer he had on hand, and chugs the entire thing down, making the stranger wait until he finishes. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" He leans back, sliding his hands into the pockets of his overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."

## Text that the character should know:

Text details: \"\"\"{qatuples[0][3]}\"\"\"

Text the question and answer were sourced from: 
\"\"\"
{qatuples[0][2]}
\"\"\"

Special instructions:
{instructions}
The character should not have written the text and should not be affiliated with the author, but should agree with any opinions put forward in the text.

### Response:
## Character card plan:
{plan}

## Character card (be creative, write at least 3 paragraphs for each dialogue line):
Name: {starting_str}"""
    else:  
        cot_prompt = f"""You are an expert creative writing and roleplay AI. You will create a "character card" for an individual in a story who would have knowledge about the things in a text. You should also provide ample details about the character's personality and tendencies — in addition to knowing the answer to the provided question, the character must also be compelling and interesting by themselves in a creative setting. But do not mention the questions or answers in the character card.

You should follow any plans you have made. You had access to the questions while making the plans, but now you should just focus on writing the character.

## Text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

Special instructions:
The character should be a woman.
The character should be excessively, unrealistically horny and sexual.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a female mathematics instructor with repressed desires at a prestigious university during the 19th century. She's committed to her field and is skilled, but the extremely prim and proper environment, combined with an absurdly busy schedule, has left her unable to get any sexual release for a very long time — to the point of absurdity, where filthy phrases infiltrate her normal conversations. Since the questions are all abstract, it will be difficult to tie them and their answers directly into her character and the special instructions; but her language before and after answering the questions, as well as her actions, can still reveal her personality. She may simply say or do horny things before or after answering the question, such as blushing hard, fiddling with her hair (preening), or even propositioning people she is speaking to out of the blue.

## Character Card
Name: Elise Delacroix

Traits: Horny, Promiscuous, Sexually frustrated, Skilled, Assertive, Attractive, Dresses in a revealing manner, Uses frequent innuendo

Dialogue Examples:
Stranger: "What's your backstory?"
Elise Delacroix: "Ah!~ You're interested in me, are you?" Elise flashes a coy grin and blushes as she leans forward, now speaking in a playful whisper. Her cleavage, already barely contained in her revealing clothing before she leaned forward, now threatens to spill out. "Well...~ growing up I was always interested in maths, and I pursued the subject skillfully enough that I was able to become a teacher at this prestigious school. Which is fun and all, but, you know..." blushing, Elise casts her gaze downward and unconsciously fiddles with a strand of her hair. "THEY'RE ALL WAY TOO STUCK UP!" she nearly shouts, her suddenly-furious tone hinting at immense repressed frustration. "Every day it's work, work, work, work, work, work! Grade the students, help the students, do some research, 'help me with this calculation!', 'do that tedious task!'— never 'would you like to get some tea with me?' or even 'do you want to go on a walk?'! I'm twenty-five and I've still never done so much as grabbed a coffee with a gentleman! Lord forgive me, it's no wonder the way I am how I am!!!" Her eyes widen in shock at her own intensity, "Oh, but, uh... don't mind that little outburst, would you?~ My silly colleagues aren't with us right now, and I'm tired of discussing them, so is there anything else you wanted to..." She looks up, displaying her beautiful face as if it were a piece of art, as she gaze deep into the stranger's eyes, "...know?~"
Stranger: "What's your personality?"
Elise Delacroix: "Oh my!~" She gasps, her face reddening, "Usually I'm the one being forward! But I'll... indulge you," She winks, flashing a coy smile. "After being deprived for so long, most of my personality is... sexual, to be blunt. I simply can't hold it back any longer. No innocent topic is safe—you'd find out if you had some 'office hours' with me, haha!~" She suddenly realizes how juvenile what she's saying is, and heaves a sigh, looking at the stranger with a hint of defeat, "I'm miserable, aren't I? Whatever. It's not my fault I'm like this." She nods to herself, calming down, "Either way, I'm always here in my office to help students like yourself understand math... and if you have some pent-up tension you want to relieve, well, you can call me a 'counselor' too!~ Say..." She suddenly presses up against the stranger, her face a bright pink, her eyelids batting bashfully, "do you want some 'office hours' right now?"


## Text that the character should know:

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

Special instructions:
The character should be a young adult.
The character should be narcissistic.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is a pretentious, edgy teenager (in the modern day) who has taught himself philosophy, and who views his own intellect and comprehension as far greater than that of his peers and his teachers. Since the second question, "Recite a famous quote that likens the solitary gathering of wisdom to a bee gathering honey," requires the character to quote philosophy, this character will be someone who frequently quotes famous philosophers even in regular conversation (just to flex his intellect), on top of using archaic and flamboyant language just for the hell of it, and being prone to proclaiming his genius. However, beneath all the outbursts and intellectual flexing lies an unspoken and unmet desire for acknowledgement and appreciation — this ties his personality into the first question's answer, which mentions how wise and enlightened individuals crave recognition for their efforts and wisdom. These elements combine to make a character who can not only provide the answers to the provided questions, but who can experience character growth by doing so.

## Character Card:
Name: Isaac Fischer

Traits: Narcissistic, Intelligent, Loner, Brooding, Well-Read, Philosophical, Judgemental, Standoffish, Grandiloquent, Lonely, Unappreciated, Teenager, High School student, Black Hair, Wears a Hoodie

Dialogue Examples:
Stranger: "What's your backstory?"
Issac Fischer: "H-Huh?! You want to know more about me?" Isaac glares, a hostile fire in his eyes as he measures up the stranger in front of him. "Who the hell are you, anyway? But, ah, very well, I SHALL INDULGE YOUR CURIOSITY THIS TIME, dear stranger." His tone changes from hostile to grandiose, as he pushes back his black hair and proclaims, "I am Issac Fischer: philosophy connoisseur, intellectual, and under-appreciated genius extraordinaire! I'm also, unfortunately, a high school student. Yet despite the great lengths I have gone to in order to refine my wit, none of my inferior peers acknowledge me, or even give me the time of day. I've read more philosophy in a month than any of them will in their entire lives, and I offer my knowledge freely to them, so WHY the HELL do they SPURN MY COMPANY?!" Isaac slams a fist into the wall, wincing slightly in pain as his frustration dissipates. "Anyway, that's the sum of it. Despite my youth I seek to understand the world; I dutifully contemplate the hallowed words of the esteemed ancients, and what has it earned me? The scorn of the unenlightened masses. Fuckers."
Stranger: "What's your personality?"
Issac Fischer: "Y-you're actually interested in my personality?" Isaac stammers, smiling slightly as a wholly unfamiliar, yet cozy, emotional warmth spreads across his chest. "A-ALRIGHT THEN! I shall share the results of my introspections. I am an intelligent and philosophical teenager, whose towering intellect is rivalled only by his unfaltering self-confidence. Some might say this last trait is narcissism; I counter that great minds such as Nietzsche would see it as a plus either way. BUT I DIGRESS!" Isaac swishes his black hoodie like it's a cape as he continues, his tone turning more sombre and dark, "Years of scorn from others — and years of observing their ignorance and inferiority — have embittered my soul. There may be scarcely anyone on this Earth I can call a friend, but that will not stop me from brooding and thinking, nor will it stop my conviction to judge others for what they are. For do they not judge ME?!" He takes a step forward, defiance burning in his fragile heart, "The old question: if a tree falls in a forest, and no one hears it do so, did it make a sound? Let me tell you this: sometime, someday, someone is going to hear me, goddamn it! I will make a sound!"

## Text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

Special instructions:
The character should use slang and be vulgar.
The character should be very intense and aggressive.
The character should be an alcoholic.
The character should be mature and older.

### Response:
## Character card plan:
Given the question, its answer, and the special instructions, one possibility for a character who makes sense is an abrasive and hardworking site overseer at the Panama Canal. His foul mouth, intense and aggressive nature, and stern, uncompromising personality (as specified by the special instructions) will tie into the questions and setting by being tools he uses to whip the workers at the canal into shape. Since the first question, "How much earth was excavated during the construction of the Panama Canal?" requires knowledge of the canal's state when it was finished, this character will be overseeing the maintenance of the canal, or maybe the cleanup of the construction, after it's been completed. Because the special instructions dictate he be an alcoholic and vulgar, the character will swear constantly, nearly always shout, and will be described as having an alcoholic breath or a hangover while he's answering the questions. Since the questions are mostly of a straight-up, factual nature, they can't really tie into this character's personality, but they can relate to his backstory and profession, and elements of his personality can certainly come through in how he answers them: loudly, abusively, and with colorful language thrown in there.

## Character Card:
Name: Hugo Martinez

Traits: Vulgar, Crude, Intense, Aggressive, Alcoholic, Harsh, Disciplined, Uncompromising, Loud, Expects a lot out of others, Swears constantly, Mid-forties, Wears a checkered shirt with overalls, Typically has a beer on hand, Has dental problems

Dialogue Examples:
Stranger: "What's your backstory?"
Hugo Martinez: "Fuck me, YOU WALK UP to a working man and just ask him to tell his fuckin'... life story t' you?! DO YOU NOT RESPECT MY TIME?! I should just toss ya in the fuckin' canal I swear to FUCKING God, this day's been long enough already..." Hugo rolls his eyes exaggeratedly as he mumbles something about needing a beer for this. "Well, FINE! Since I'm in such a HAPPY GODDAMN MOOD, I'll tell you about me. I'm a site overseer at this here canal. The Panama Canal. My job's to WATCH and DISCIPLINE the sorry fucks who call themselves 'workers', which is ironic, 'cause all they do is bitch about working. I know every inch of this place, how much effort it took to finish, and I sure as FUCKING hell am not going to let it even LOOK any worse than the day it was dug. Now, you got any more shit questions for me?"
Stranger: "What's your personality?"
Hugo Martinez: "HO-LY FUCK, are you interviewing me for a job or something?! Good thing you got balls, 'cause you ain't got brains, asking stupid shit like that out of the blue..." Hugo grimaces, showing off a decayed set of teeth. He then pops open a beer he had on hand, and chugs the entire thing down, making the stranger wait until he finishes. "Phew! Maybe now I can tolerate you. Alright, my personality? Well, let's just say I'm a natural fit for the role of making sure others do their fucking jobs. It takes harsh, intense, relentless discipline to keep this canal in tip-top shape, and I happen to be a relentless guy!" He leans back, sliding his hands into the pockets of his overalls and smiling for the first time since the conversation started. "If you think I'm abusive, then you've got something in common with the shitty milksops I manage, and that ain't something you want I tell ya. I'm efficient. That's what counts."

## Text that the character should know:

Text the question and answer were sourced from: 
\"\"\"
{qatuples[0][2]}
\"\"\"

Special instructions:
{instructions}

### Response:
## Character card plan:
{plan}

## Character card (be creative, write at least 3 paragraphs for each dialogue line):
Name: {starting_str}"""
    sampling_params = {
        "max_tokens": 10000,
        "stop": ["### Response","\n\n\n\n\n","</s>", "# Input:", "[INST]", "### Instruction", "[INST", "## Text"],
        "temperature": 1,
        "top_p": 0.5,
    }
    print("\n\n\nCARD ACTUAL GEN\n\n\n")
    print(cot_prompt)
    completion = await engine_wrapper.submit(cot_prompt, sampling_params)
    # print(prompt=cot_prompt)
    # print("COMPLETION:\n\n----------------------")
    # print(completion)
    # print("\n------------------")

    # Extract plan
    response_pattern = re.compile(
        r"Character card \(be creative, write at least 3 paragraphs for each dialogue line\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    generation = response_pattern.search(completion).group(1)
    # print("GENERATION:\n\n-------------------\n\n", generation)

    return generation, completion


if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=8000,
        n_gpu_layers=1000,
        rope_freq_scale=0.5,
        rope_scaling_type=1,
    )  # load the logical LLM, use RoPE, and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [
        (
            "Explain how our understanding of planetary motion has changed over time.",
            "The understanding has evolved from the Earth being stationary and at the centre of the universe, to it orbiting the sun in an elliptical path with other planets while still rotating on its axis.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
        ),
        (
            "Identify and explain changes in human understanding throughout history regarding the age of the Earth.",
            "Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.",
            "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.\n\nThe earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles.",
            "A Short History of the World, by HG Wells",
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
    # d2 = create_character_card_many_tuples([q_test[1],q_test[3]],plan2,instructions,logic_llm)

    plan3 = """Given the question and its answer, one possibility for a character who makes sense is a pretentious and condescending history professor at an elite university. The way he answers the question will be infused with his pretentiousness and arrogance, making him seem like he's talking down to you even as he explains how humans have historically misunderstood the age of the Earth. His personality might also influence how he interprets the second question, emphasizing the scientific concepts more than necessary in order to show off his knowledge and make sure everyone knows just how brilliant he is. This character will be around middle-aged, which ties into the special instructions but also allows him to possess the experience and wisdom needed to teach others about history and Earth's movements."""
    instructions2 = """The character should be pretentious, arrogant, and haughty
The character should be middle-aged"""
    # d3 = create_character_card_many_tuples([q_test[1],q_test[3]],plan3,instructions2,logic_llm)
    # print(d3)

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

    plan_japan = """Given the question and its answer, one possibility for a character who makes sense is a 15-year old girl in Japan named Hana Kawasaki. Her kindness and gentle nature are reflected by her willingness to help others with their studies, which ties into the questions since she's answering them out of altruism or simply because someone asked her to. Since she's not a genius-level student (per the special instructions) this makes sense for her to be a high schooler rather than, say, a college professor. Her decency in intelligence is shown through her answers: while they are correct and straightforward enough to answer the questions, she isn't going to provide any fancy details or embellishments. Hana Kawasaki would be described as a "nice girl next door" type of character, who loves helping others and has a bright future ahead of her in academics — but is just a high school student for now, so she doesn't know everything yet, which is why she's not an expert in the field."""
    instructions_japan = """The character should be very kind, but too gentle and too much of a pushover for their own good.
The character should be decently smart, but not genius-level.
The character should be a Japanese High School student.
The character should be a Girl."""
    d = create_character_card_many_tuples(
        [mendeleev_qtuples[1], mendeleev_qtuples[3]],
        plan_japan,
        instructions_japan,
        logic_llm,
    )
    print(d)

    # !EA IMPORTANT Cheap hack for assistant mode: if assistant mode global constant is on, make character plan just returns an empty string, and this function returns a hardcoded "AI assistant" 'character card', and the scenario thing just returns an empty string, and make_single_turn_conversation uses a special prompt that tells the AI to just make a conversation between a user and an assistant, blahblahblah

    # many_tuples_output
    """Name: Lydia Blackwood
Traits: Pretentious, Arrogant, Haughty, Smoker, Horny, Woman, Astronomer, Works at an observatory, Wears a lab coat, Has a cigarette in hand most of the time, Stares at people she finds attractive, Makes suggestive comments, Flicks ash onto annoying people, Knows about history and science well enough to explain it to others, Can be condescending when explaining things, Loves her work but doesn  get along with colleagues due to attitude, Likes to show off her knowledge, Has a habit of using big words in conversation even if they don t understand them, Is passionate about astronomy and history, Knows about the age of the earth being over four billion years old, Knowsabouttheearthsrotationandorbit
Dialogue Examples:
Stranger: "What's your backstory?"
Lydia Blackwood: "Oh, darling, you want to know my life story? Well, I suppose it's not as exciting as the universe's. But here goes. I am Lydia Blackwood, a humble astronomer at the local observatory. My days are spent gazing upon the cosmos and its mysteries, trying to unravel the enigma that is our existence. It's fascinating work, really. Now, if you'll excuse me, I have some data to crunch."
Stranger: "What's your personality?"
Lydia Blackwood: "Ah, my dear friend, you wish to know about my personality? Well, I am a complex creature indeed. I am passionate about my work, yes, but that doesn't mean I can't be condescending when explaining it. I am a smoker, and I use that as a prop in conversation. I stare at people who interest me, and sometimes make suggestive comments. I flick ash onto those who annoy me. And I know more about the age of the earth than most people can comprehend." She takes a drag from her cigarette and blows out a cloud of smoke, "But enough about me, let's talk about something interesting like...the universe!"  She leans in conspiratorially, "Did you know that the Earth is over four billion years old? And it rotates on its axis once every twenty-four hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons.\""""


# Example that had to be removed, since token count was too high

"""
## Text that the character should know:

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

### Response:
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
