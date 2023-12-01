# Super Amazing Infinite Data Notebook
-- Pre-Pre Alpha --

Currently this code can generate good question/answer pairs, and decent singleturn conversations about those question/answer pairs, from most plaintext books (I've tested on a few from project gutenberg, which are open source and free and plaintext).

It is not polished. In fact you might consider it "antipolished" since I've been rushing through things and iterating and etc. to the point where outdated TODOs are the norm, you'll find all my notes strewn about the place, it's painfully obvious which files were copypasted from which, and I'm 90% sure there are some bugs lying in wait in places.

BUT

it partly works. You can get decent singleturn data off of this, especially if you use a 70b, but a good 13b works in a pinch too. I've tested with Airoboros-l2-13b-3.1.1 Q8_0 gguf, and AIROBOROS-l2-70b-Q4_K_M, both on an A6000 rented using vast.ai which you can get for like less than 50 cents an hour. It works decently well and generates decently fast.

What does "work" mean here?

It means infinite data! No but really. There's so much quality text out in the world, and yet we model creators often reduce ourselves to data annotators when creating datasets, manually combing through ungodly amounts of text just to get something we can train on. It's nonsensical that we, creators of machines that can write, spend a good amount of our time writing. This tries to solve that. Mind you, if you use a 13b and not a 70b, right now you WILL have to do a good amount of editing. But eh, pre alpha.

Focus is very, very heavily on answer accuracy and veracity, over efficiency. Double and triple checks everywhere. This is to allow potential industrial-strength data generation once it's done; but comes at the cost of speed and sanity if you're running on a laptop. However, someday if you have a technical manual or legal corpus you want a model to memorize, run this over it like five times and you'll get a nice dataset.

## HOW TO WORK ON THIS AND BEND IT TO YOUR WILL (hack it to your own needs)

Each prompt is in its own file. in generatiom_functions/
Each file has a test in that file. All of them up to (not including) the multiturn ones should be runnable.

To run a test for a module

`from the root directory of the repo`

```
python -m generation_functions.some_llm_prompt_file
```

MIT license. :partyparrot:

## Known limitations:
There is/was a significant bug throughout some parts of the script where .lower() is needlessly used. This would be fine except for the fact that output is being passed from function to function. Remove wherever you find it and adapt regexes to account for the change in caps. 

### If the model the data's generated with has GPT-4 isms, then unless you prompt it otherwise, some of the data might have those too.
I've noticed character cards beginning with "Ah," which is like the most GPT-4ism GPT-4ism that ever GPT-4ism'd.

### Example of known bias -- antireligious:
When given the opening paragraph of HG Well's Short History of the World, and the rather innocuous question/answer:
'Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.'

It generates this:
```
Given the question and its answer, one possibility for a character who makes sense is a scholar or professor, specifically in the field of geology or archaeology. This person would be knowledgeable about ancient texts, especially religious ones, as well as the more recent findings from science and history. They might have a passion for uncovering truths hidden beneath layers of misconception and prejudice. Their backstory could involve being born into a religious family but seeking out secular learning, leading to conflict with their loved ones or even expulsion from their community. Despite these challenges, the character remains steadfast in their pursuit of knowledge and truth, driven by an inner desire for understanding and a belief that everyone deserves access to accurate information. They might struggle with loneliness due to their unorthodox views, yet find solace in the company of ancient artifacts and geological formations, which speak volumes without uttering a word. This character is not necessarily anti-religious, but rather sees religion as another part of the human story that should be understood and respected, even if it doesn't align with their personal beliefs. Overall, this individual embodies both the complexities of human understanding throughout history and the ongoing quest for knowledge and truth in a world where misconceptions still abound.
```
I mean I'm not religious, but that reads like something off of r/Atheism. And it's very, very obviously GPT-4 inspired.
Solution found: write my own character cards. I've done that for multiturn and it works great, but multiturn isn't fully working yet, so... we're stuck with gptisms for the next few days.

### Problem: it mentions "the text" sometimes
### Solution: I'm going to make a prompt that rewrites questions and answers to not reference "the text" and put all the qatuples through that if a flag is set at the top of the notebook, which should stop this infuriatingly prevalent issue.

Note:
Focus is very, very heavily on answer accuracy and veracity, over efficiency. Double and triple checks everywhere. This is to allow potential industrial-strength data generation; but comes at the cost of speed and sanity if you're running on a laptop.

## Running this with gpu inference
I have spent too many hours of my life either using llama.cpp python without gpu inference, or trying and failing to get gpu infernce working. Below is a command created using distilled Evan tears™ that makes gpu inference magically work on rented vast.ai instances. Maybe it works on runpod too idk I don't use runpod.

```
apt install build-essential && conda install cmake && conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy && CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir && pip install protobuf sentencepiece transformers matplotlib
```
input 'y' to everything and you're good.

## FAQ
Q: "Your code is unpolished and shitty"
A: "Yes."

Q: "I found a bug!"
A: "Tell me where and I'll fix it."

Q: "Why release this in this state when you're only like 2 days from having a much more polished version with working multiturn?"
A: "Because by releasing this early, I can get earlier feedback that will help me improve it more effectively before a broader public release. I've learned from Augmental and I want to test my stuff more now. ~~Also I want to release my Augmented data pipeline before Jondurbin releases his because I'm that petty and prideful lmao and I want to be first to use an open source model for it successfully and open source the code~~"

Q: "I have a question that isn't in the FAQ"
A: "Ping me on Discord! @Heralax"

# Modification guide:
Specific prompts you might want to change are judged_worthy_for_questions, generate_question_plan, nd generate_question

There is a significant bug throughout many parts of the script where .lower() is needlessly used. This would be fine except for the fact that output is being passed from function to function. Remove wherever you find it and adapt regexes to account for the change in caps. 

# todo rename all files in generation_functions such that their grammars are file_name_grammar.py and the functions have the same names as the files, so I don't have to hunt anymore.

## Known limitation: the model's opinions can seep into the characters, if the text does not explicitly state an opinion.

# Bug found: llama_cpp_python hangs forever if the max_tokens is lower than the starting number of tokens. Set it as high as the context, always.

# If the model the data's generated with has GPT-4 isms, then unless you prompt it otherwise, some of the data might have those too.
I've noticed character cards beginning with "Ah," which is like the most GPT-4ism GPT-4ism that ever GPT-4ism'd.

## Example of known bias -- antireligious:
When given the opening paragraph of HG Well's Short History of the World, and the rather innocuous question/answer:
'Identify and explain changes in human understanding throughout history regarding the age of the Earth.',
'Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old.'

It generates this:
```
Given the question and its answer, one possibility for a character who makes sense is a scholar or professor, specifically in the field of geology or archaeology. This person would be knowledgeable about ancient texts, especially religious ones, as well as the more recent findings from science and history. They might have a passion for uncovering truths hidden beneath layers of misconception and prejudice. Their backstory could involve being born into a religious family but seeking out secular learning, leading to conflict with their loved ones or even expulsion from their community. Despite these challenges, the character remains steadfast in their pursuit of knowledge and truth, driven by an inner desire for understanding and a belief that everyone deserves access to accurate information. They might struggle with loneliness due to their unorthodox views, yet find solace in the company of ancient artifacts and geological formations, which speak volumes without uttering a word. This character is not necessarily anti-religious, but rather sees religion as another part of the human story that should be understood and respected, even if it doesn't align with their personal beliefs. Overall, this individual embodies both the complexities of human understanding throughout history and the ongoing quest for knowledge and truth in a world where misconceptions still abound.
```
I mean I'm not religious, but that reads like something off of r/Atheism.  I might just make some synthetic data off of the Bible and 12 Rules for Life to de-align this thing.

^ Alternative, and better approach: Prompting (ICL). Make one of the character card examples a devout priest, and the question be a thing about a religious text. This may be able to counteract some of the inherent, GPT-4 style bias that exists.

Before then though, add a jailbreak to the character card prompt that tells it to assume the morality of the book. To that one and to the question one. DO THIS AFTER THE PIPELINE IS ROUGHLY FINISHED AND YOU CAN TEST IT ON VERY FRINGE TEXTS. Currently the model obeys the facts of a text, but not the opinions implied by it.