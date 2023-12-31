# Augmentoolkit
Generate multi-turn training data, about any subject, using Open Source LLMs!
Save yourself the time of manually editing 1000s of AI chats to build your own dataset (which you then can't open source anyway because of personal reputation risks). Easily configure the prompts and settings to generate conversations aligned to your tastes and interests.
Avoids breaking the bank (and getting your API key revoked) because it doesn't use the OpenAI API.

## Table of Contents:
1. [Installation](#installation)
    - [Getting the Repository](#getting-the-repository)
    - [Installing Dependencies](#installing-dependencies)
    - [Quick Install](#quick-install)
2. [Introduction](#introduction)
    - [What is Augmentoolkit?](#what-is-augmentoolkit)
    - [Why Was it Built?](#why-was-it-built)
3. [Quickstart](#quickstart)
4. [Usage](#usage)
    - [Concepts and Operation](#concepts-and-operation)
    - [Understanding What is Going On as It Runs](#understanding-what-is-going-on-as-it-runs)
    - [Some Features Worth Being Aware Of](#some-features-worth-being-aware-of)
5. [Customization](#customization-arranged-in-order-of-least-to-most-difficult-to-implement)
6. [General Notes, Known Limitations, Quirks, Features](#general-notes-known-limitations-quirks-features)
    - [Why is it writing so many files?](#why-is-it-writing-so-many-files)
    - [Known Limitations](#known-limitations)
7. [Contributing](#contributing)
    - [Improvement Areas](#obvious-areas-for-improvement-feel-free-to-open-a-pr)
8. [Contact](#contact)

## Installation:
Augmentoolkit is a Jupyter Notebook with some functions to import, so there is not much here besides cloning the repo and installing its dependencies (you probably already have most of them). Still, the details are here for completion's sake (and the newer enthusiasts among us).

First, get the repository onto your computer (or an instance rented out by your favorite compute provider, i.e., runpod, vast.ai, etc.):
```
git clone https://github.com/e-p-armstrong/augmentool.git
```

Then, install the project's dependencies. You need the latest [Llama.cpp Python](https://github.com/abetlen/llama-cpp-python) with GPU acceleration, and the following Python libraries: `protobuf sentencepiece transformers matplotlib nltk`. Installing the former is honestly a pain; installing the latter is as simple as:
```
pip install protobuf sentencepiece transformers matplotlib nltk
``` 

Things change fast enough in ML that you should refer to the link for install advice about Llama.cpp Python, BUT if you have freshly rented out a GPU instance from a compute provider, then the quick install command below should work.

**Quick install:** on a fresh Vast.ai Linux instance (select the ` anibali/pytorch:2.0.1-cuda11.8 ` docker image), you would need to run the following command to get this working:
```
apt install -y build-essential && conda install -y cmake && conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc && CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir && pip install protobuf sentencepiece transformers matplotlib nltk
```
**Note: Runpod instances,** I believe, come with CMAKE and some other things preinstalled, but do not include conda as far as I know. Your command can probably be shorter there, and might just need to include the llama.cpp and Python parts. I don't use Runpod, IDK, try it out and tell me.

## Introduction: What is this and why was it built?
Open source is meant to move fast, be shareable, and be novel. Our datasets are none of these. Most of the top models have private datasets (or are merges), and replicating said datasets often either A) requires an obscene number of OpenAI API credits, or B) requires you, the model creator, to spend dozens if not hundreds of hours accumulating a hybrid dataset based off of your own conversations with bots. The former is based on a paid service (whose TOS you're violating) that can ban you at any second and whose writing style you probably hate; the latter is far too slow to iterate on, does not scale at all, and is not easily shareable due to the sensitive nature of private chats with bots. And moreover, if we're literally creating machines that can write, why do we spend most of our time writing?

**Augmentoolkit** is meant to make high-quality data generation easy, fast, shareable, configurable, and open-source. It is meant to allow the easy creation of datasets about any knowledge base that exists in plain text. It is meant to allow models to bootstrap additional training data for themselves. It is meant to allow enthusiasts who have a powerful computer, but don't want to train models, to contribute to the advancement of open source AI by generating swathes of data. It's meant to help narrow the gap between OpenAI's obscenely large dataset, and we have in the land of Open Source. Whether you're making a finetune on a specific scientific domain, or are creating the latest RP model to top [Weicon's leaderboard](https://rentry.co/ayumi_erp_rating), Augmentoolkit exists to make your data problems a bit less problematic.

A flowchart of Augmentoolkit's operation can be found in the [Usage](#usage) section.

Conceptually, Augmentoolkit takes human-written text with information in it, and turns it into instruct-tuning data: 
- It uses the text's information to generate questions that test the information, and it also generates answers to the questions that use the information. 
- It triple-checks whether the generated questions and answers are accurate and only use information provided in the text (ensuring that the LLM did not hallucinate new information). 
- Finally, it writes an interaction in a fictional setting between a character with domain expertise, and an ignorant secondary character, where the secondary character asks the questions and the primary character answers them. 
- After checking that this conversation faithfully includes the original questions and answers, the result is saved as part of the newly-generated dataset. The usage of characters and a setting means that the model's creative writing and RP skill can be improved at the same time as its knowledge base (but if you don't want an RP bot, you can always turn "Assistant Mode" on for user-assistant interactions instead).
You can see a flowchart of this process over in [Usage](#usage).

The name "Augmentoolkit" comes from "Augmented data" (my own coined phrase for human-written text that is AI-reformatted, since I [couldn't find a standardized one](https://xkcd.com/927/)) and "toolkit".

## Quickstart:
Here's a bullet list describing how to quickly get this notebook generating text.
- Get this notebook and the other code onto a machine with the power to run [NeverSleep/FlatOrcamaid-13b-v0.2](https://huggingface.co/TheBloke/FlatOrcamaid-13B-v0.2-GGUF/tree/main), and probably [Airoboros-l2-70b-3.1.2.Q4_K_M](https://huggingface.co/TheBloke/Airoboros-L2-70B-3.1.2-GGUF) too. **If you want to save money, use something like a 3090 with a Q_6_K quant of a 13b for the majority of the steps, then copy your data over to a a machine that can run the 70b once you reach the final step. This can potentially lead to roughly 3x cost savings.**
- Install the dependencies shown in [Installation](#installation)
- Put the model(s) into ./logical_models (relative to this notebook). 
- What you do next depends on how you want to proceed:
    - If you can run Airoboros 70b (a low quant should do fine, I use Q4_K_M), then you should run all the cells until the point where it says "Stop here, Restart the Notebook, and Reimport Everything". Once all your steps finish, restart the notebook and import everything again using the third cell of the notebook. Then run the last three cells. This ensures that you use a model smart enough to correctly do multi-turn conversations.
    - If you cannot run Airoboros 70b, and don't want to rent out an A6000 (<$1/hr usually), then change the LARGE_LOGICAL_MODEL constant in the second notebook cell to be the path to the smartest model that you can run, and comment out the cell that loads the model a second time right near the end.

***If you want to run a subset of the total text through the entire pipeline, to evaluate how well it works, uncomment the following line and comment out the one above it:***
![](comment_out.jpg)

Finally, before you scream at me asking why this project of three months requires you to restart the darned notebook: it seems [this memory leak has existed since May](https://github.com/abetlen/llama-cpp-python/issues/223) and you literally can't unload a model from VRAM using llama.cpp Python, AFAIK. ¯\\\_(ツ)_/¯

## Usage
How to get this running at a basic level is covered in [Quickstart](#quickstart). This section describes what you're actually doing while you're running this, as well as how to easily customize the function of this notebook for your own use cases. It describes everything from how to operate the notebook (in greater detail) to how everything's structured, and what folders to watch as you are generating your data. For the most part you can just follow quickstart, but this section may be worth reading if you plan to make this a serious part of your model creation (which I hope you do!).

Here is a flowchart detailing how a typical run of Augmentoolkit may proceed. The source text can be anything with information you can ask questions about.
![](flowchart.jpg)

### Concepts and Operation
Read this subsection for a slightly more detailed version of the more finicky bits of the quickstart, as well as an understanding of the key files in this repo.
Augmentoolkit is centered around a Jupyter Notebook (`processing.ipynb`) so that progress is easier to inspect, and starting and stopping is easier. All the prompts and their GBNF Grammars are stored in `./generation_functions`.

You run Augmentoolkit by running the Jupyter Notebook `processing.ipynb`. **Depending on if you are using a combination of a larger LLM and a smaller one, you will either need to comment out one notebook cell, or restart the notebook at a (clearly indicated) point.**

**If you want to run through the whole thing on a 13b**, I recommend either using as smart a model as you can, along with assistant mode. You can turn on assistant mode by setting this to True:
![](step1.jpg)

Then comment this code cell out (near the very end):
![](step2.jpg)

Then run all cells

**If you want to run using a 2-step process with a small and large LLM**
Then run cells until you see a markdown cell saying "Stop Here" and some other stuff.

Once all the cells before the point below finish executing, restart the notebook...
![](step2.jpg)

...And run the import cell again...
![](step3.jpg)

Once this is done, you can run the last three cells, starting with the one you saw just below the "Stop Here" cell. Apologies for the convoluted process, I don't actually think there's a way to get around the VRAM memory leak yet.

***Important files:*** The core of the project is `processing.ipynb`, which needs `./generation_functions`, at least one .gguf model in `./logical_model`, and one or more plaintext files -- all in the same folder as it -- in order to run. If you are going to change anything, please read [Customization](#customization-arranged-in-order-of-least-to-most-difficult-to-implement) first.
### Understanding what is going on as it runs
This subsection summarizes output folders and code structure.
This notebook makes plenty of folders while it runs. The ones you may want to pay attention to are `./worthy_for_questions`, `./qatuples_raw`, `./qatuples_revised`, `./multiturn_convs_info`, and finally, `./multiturn_convs`. `./multiturn_convs` is the final output directory. Everything else is just the notebook saving the outputs of every single step in case someone wants to train a model specifically for running this pipeline at some point.

Do not move or remove the folders as they're generated.

As for code structure, `processing.ipynb` handles the control flow and file writing, and the functions it imports (from `./generation_functions`) call the LLM with various prompts. `write_output_to_file()` can mostly be ignored; it just saves the full completion of each step for the sake of potential future training of a model specifically for running this pipeline (think jondurbin/cinematika-7b-v0.1). The main output of the function is usually just passed onto the next part of the pipeline. If a file has been written already, any future attempts to write that file will be skipped, allowing for easy resumption of generation after interruption.

Most functions are actually quite simple at heart; they call functions that generate output, and pass that output onto other functions, writing things to files as need be. 90% of the code in `processing` can be understood with that in mind.

### Some features worth being aware of
This subsection describes things that make life easier in Augmentoolkit.
- **Easy resume:** don't have long uninterrupted periods of time to run this? No problem! Augmentoolkit saves outputs as they're written and resumes generation painlessly, so you can start and stop stress free.
- **Two-model generation for the sake of SPEED:** every single task, except the very last one (multi-turn conversation generation) can be accomplished reliably by a good enough 13b. It is highly recommended that you run all steps until the actual multi-turn conversation generation using a 13b, and then switch to a 70b for the last part. This will require a restart of the notebook to deallocate the VRAM, but don't worry, the easy resume means this should work fine (if you haven't moved any files around).
- **Validation, validation, validation:** Learning lessons from the original Augmental, consistency with the source text is an extremely high priority here, and this is ensured with multiple layers of LLM-based validation (and at the end, numerous examples of regex-based validation).
- **GBNF Grammars:** possibly the most under-utilized feature of Llama.cpp sees a lot of love here. Check out any `_grammar.py` file in `./generation_functions` and see for yourself how this notebook ensures consistent output across its many steps!

The steps above describe how to run the notebook with default settings. But your use case likely differs from the default. Here's a step-by-step process about how to customize it!
### Customization (arranged in order of least-to-most difficult to implement):
Read this to learn how to hack Augmentoolkit for your own use cases.
1. ***Change the source texts used to generate training data.*** You can do this in the first code cell of the notebook. **IMPORTANT** the filenames of these should be formatted in a specific way, since the filenames are used as part of the prompts and in at least one regex. You need to have them be like: `[textname], by authorname`. So for example, `Simple Sabotage, by the Office of Strategic Services`. You can also include the publication date after the author name if you want (as in `Principles of Chemistry, by Demitry Mendeleev, published 1897`), but note that this may bias most of the characters to live in the era of the textbook, which may or may not be what you want. **If you have a PDF you want to use as a source text, you can convert it to a .txt using `convert_pdf_to_text.py`.** If you want a good source of plaintext documents, [try Project Gutenberg](https://www.gutenberg.org/); if you want educational PDFs, try [OpenStax](https://openstax.org/subjects).

![](changetext.jpg)

2. ***Change the personalities of the characters generated.*** Currently, when generating characters for the multi-turn conversation step, three randomly-selected traits are appended to the "special instructions" set of the prompt to constrain what kind of character is generated by the model. Depending on what kind of model you want to make, or even just if your preferences vary, then you will probably want to modify this a bit. You can do so in `./generation_functions/special_instructions.py`. A more in-depth description of the trait-axis system that I (over)thought up is available in the comments of that file.

![](specialinstructions.jpg)

3. ***Change the constants.*** There are a few constant values in this notebook, and in `./generation_functions/constant_values.py` (the latter is only really used when testing prompts during development). These constants are tested, but if your use case requires special settings (e.g., you want to make conversations from more permutations of existing questions; or you think the character counts for the "duplicate question/answer" validation functions are too restrictive) then feel free to change the related setting. The most intuitive and least-likely-to-break-anything settings to change are rearrangements_to_take and double_check_counter. Beyond that... you'll need to figure out what the function does before changing it if you expect it to run.

4. ***Assistant Mode*** Technically this could be considered part of 3), but it's different enough that I feel it warrants separate explanation. By default, the notebook is configured to produce RP-style data; "Assistant mode" is something you can toggle in the settings cell immediately below this one, which skips character and scenario generation and answers every question in a chat between a user and a helpful AI assistant (with no personality). In the limited testing I have done with this, **it seems that assistant mode is simple enough to work from start-to-finish with 13b models** such as Flatorcamaid by Ikari. So if your compute or time are very limited, or you are using this for a more professional use case, feel free to turn this on.

5. ***Change the model.*** This is as simple as switching the LOGICAL_MODEL value out for another one, but your mileage may vary significantly. My personal recommendation is to use [FlatOrcamaid](https://huggingface.co/TheBloke/FlatOrcamaid-13B-v0.2-GGUF/tree/main) (helluva name, I know) for the small model, and [Airoboros-l2-70b-3.1.2.Q4_K_M](https://huggingface.co/TheBloke/Airoboros-L2-70B-3.1.2-GGUF) for the large model. You will also have to adjust RoPE scaling for non-Llama 2 models -- e.g., if you're using Mixtral, don't leave `rope_freq_scale=0.33`, which 3xes the context (you do not need 96k context, only 12k).

6. ***Change the examples.*** If you change the examples used in some of the prompts in `./generation_functions` you can completely overhaul what this notebook does (and critically, what kinds of texts it targets), but this requires a lot of prompting skill and possibly huge amounts of time to get it working again. Unless you want to convert this notebook from question-and-answer generation to some completely other task, I'd recommend changing only the conversation generation prompts -- they're a bit less finnicky, and if you just want to change the kind of characters generated (maybe you want a different writing style) that's where you'd find the greatest differences.


## General Notes, Known Limitations, Quirks, Features
### Why is it writing so many files?
This notebook writes the final questions generated, the revisions of those questions, and the final multi-turn conversations, to files. But it also writes the output of every single prompt to a unique file in a folder named for the prompt it's a part of (to a unique file whose filename is a uuid). Why all the writing? Taking inspiration from Jon Durbin's Cinematika, this notebook saves output information so that, in the future, possibly, a model can be finetuned specifically for running as the logical model behind the notebook. Writing each step down ensures that a dataset is made and outputs are not wasted. If a model is ever built, what'll probably be done is a regex and other code will be used to determine which runs (identified by the same uuid across folders) ended successfully, and these will make up the dataset. DPO might also be done on steps that failed vs steps that succeeded.

The folders you want to look out for, by default, are named `qatuples_raw`, `qatuples_revised`, and `multi_turn_convs`.

### Known limitations:
Multi-turn conversations sometimes have impersonation (ie, one character will describe what another character does in their own message). This only happens sometimes from my testing, became much less common when using a mix of Flatorcamaid + Airoboros 70b, and is quite possibly easily fixable by creating a prompt that takes conversations with potential impersonation and rewrites them to have none. I simply have not done this yet because there are a number of such improvements I *could* do.

Multi-turn conversations can have the primary character ask if the secondary character needs anything else in a repetitive way. So for instance, the primary character might end with "Do you need anything else?" twice or thrice in a row. I am unsure whether this is a quirk of the model or the notebook, either way it should be easily fixable enough with a prompt (+ a regex that checks the end of statements, so that the prompt isn't called on things that are fine). Also became much less of a problem after switching to a combo of Flatorcamaid + Airo.

Spelling mistakes -- I had to use RoPE to boost the ctx quite high, and I think this is causing the model to (VERY rarely) misspell things. This happens maybe one in a dozen outputs, maybe less. Models with higher ctx, e.g., Mixtral, probably won't suffer from this problem at all.

Numbers -- I've found the model missing or adding zeroes occasionally when spelling out dates. I am 99% certain this is also a RoPE issue.

Sensitive to text differences -- I've tested this on a few texts, but I will say that depending on the book you are using, and how it's written, what you get with the default Augmentool will vary significantly. This can be unpredictable: for instance, this notebook really struggles with H.G. Wells' "A Short History of the World" but is mostly fine with "Principles of Chemistry" despite them both being quite old, factual texts. If you try this on a text you like and it doesn't work, here's the process to debug it: take some times it failed (the notebook saves prompt outputs at each step so you can find where it went stupid), and manually turn the worst of those into few-shot examples for the step that went bad, except fix it up yourself. This should make the notebook less inclined to commit the same error. Then run it again, and if there's a new problem, fix it the same way.

Multi-turn conversation generation seems to have a fondness for the 19th century as a setting. This can likely be mitigated by including the publication date of the text in the filename.

`judge_paragraph.py` is way too forgiving. In my testing it was fine because at the beginning of books all it had to deal with was metadata, but it does not reject exercises in textbooks, nor does it reject things like markdown tables. This leads to some questionable questions in some cases. This prompt should be more strict.

Sometimes the model puts information about the questions or answers in the bios of characters. To fix this, the few-shot examples will need to be adjusted, or the questions and answers will need to be removed from the context of that prompt.

Occasionally things with grammars will hang for a while and then produce a screwed output. This is extremely rare, like 1 in 200, maybe more. If you find a prompt is taking much longer than it usually does, interrupt it and rerun the cell without restarting the notebook

The secondary character is quite often timid. This can probably be solved by making one of the secondary characters in the multi_turn_convs few-shot example not be timid. In other words, I forsee a shouting match between Hugo and Juan, and it's hilarious.

Other limitations -- I've listed the major ones, and the ones I've found while generating the full demo dataset, but I'm sure there are a handful I'm forgetting.

## Contributing
- This is my first-ever repo accepting (and seeking!) contributions. I really think this can make a difference to the community.
- But due to haphazard development over time, the code in this repo is only minimally cleaned-up. This means that there is no real coding style standard as of yet, even though there should be. Any more-experienced dev who shows up first and wants to enforce a proper coding standard is welcome, so long as it isn't too nitpicky.
- If you make a PR, please try running Augmentoolkit from start to finish on at least 10 paragraphs from a book to make sure that the prompts and pipeline still work. If you lack the compute I can handle this part.
- If you make an issue, please use the appropriate label (feature request or bug report)
- If you want to contact me, reach out on GitHub, on Discord (@Heralax, I usually hang out in TheBloke's server), or by [email](mailto:evanpeterarmstrong@gmail.com) (NOTE: I am decently slow at replying to email).

### Obvious areas for improvement (feel free to open a PR!):
- Convert into a script with command line arguments for the model and texts used
- Multithreading, or use the Llama.cpp server, or something else that allows multiple simultaneous forward passes if there is spare VRAM
- If there is a way to stop the vram memory leak without restarting the notebook which I have missed, implementing that would be a godsend
- Use a faster backend. I stuck with Llama.cpp because it had grammars but something like AWQ might be faster.
- General Cleanup and Bug Fixes
- Prompting format inconsistency fixes (newlines may vary even within the same prompt)
- An experimental version using mixtral instruct, which would get around the RoPE issues. Would need to change every prompt to use its format, then test it.
- Perhaps a version that, in the spirit of lean manufacturing, runs each paragraph through the entire pipeline one at a time (rather than going from one step to the next for all paragraphs) might be good for evaluating how a run is going while there is still time to abort it. May pose a problem if the VRAM memory leak issue is not solved though, as that prohibits the two-model approach.

## Contact
evanpeterarmstrong@gmail.com || @Heralax on Discord