import asyncio
import uuid
from openai import AsyncOpenAI
import re
from tqdm import asyncio as tqdmasyncio
import random
import os
import traceback
import json
import logging



def make_id():
    return str(uuid.uuid4())


# This and the class immediately below it are most relevant to API debugging
class GenerationStep:
    def __init__(self,
                 prompt_path="", # relative to the Inputs directory
                 regex=re.compile(r'.*', re.DOTALL), # take whole completion
                 sampling_params={
                     "temperature": 1,
                     "top_p": 1,
                     "max_tokens": 3000,
                     "stop": [
                            "### Response",
                            "\n\n\n\n\n",
                            "</s>",
                            "# Input:",
                            "[INST]",
                            "### Instruction",
                            "### Information",
                            "## Information",
                            "## Instruction",
                            "Name:",
                        ]
                    },
                 completion_mode=True, # Chat vs completion mode
                 retries=0,
                 engine_wrapper=None,
                 logging_level=logging.INFO,  # Default logging level
                 output_processor=lambda x: x, # to ensure that control flow does not need to have decision code handling the outputs of the LLM, you can pass in a function to handle and modify the outputs (post regex) here. By default it's just the identity function and does nothing.
                 return_input_too=True
                ):
        self.prompt_path = prompt_path
        self.regex = regex
        self.sampling_params = sampling_params
        self.completion_mode = completion_mode
        self.retries = retries
        self.logging_level = logging_level
        self.output_processor = output_processor
        self.return_input_too = return_input_too
        if not engine_wrapper:
            raise Exception("Engine wrapper not passed in!")
        self.engine_wrapper = engine_wrapper
        logging.basicConfig(level=self.logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
    async def generate(self,arguments={}):
        # Current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Dynamic INPUT_DIRECTORY path (feel free to change, DragonFox, depending on what structure you have been working towards)
        # full_prompt_path = os.path.join(current_dir, '..', '..', 'prompts',self.prompt_path)
        # Read file and escape all curly braces
        # with open(full_prompt_path, 'r') as pf:
        #     prompt = pf.read()
            # Code to ensure that interpolation works, but curly braces are still allowed in the input
            # 1. Escape all curly braces
        
        # NOTE since this is a minimal reproducible snippet I've removed the "load prompt from file" logic and just pasted the prompt in as a string
        prompt = '''You are an expert educational AI that will make a determination as to whether the contents of the paragraph(s) provided are suitable for making educational questions based off of them; these questions should be able to test the knowledge in in the book. The book in question is {textname}, and you should keep this in mind when considering what kind of questions should be capable of being developed. If there is sufficiently deep information to make questions about, you will judge it suitable, even if the knowledge being tested does not reflect typical curricula. Essentially: you will determine if provided text is a table of contents, introductory paragraph for a book, etc., or if it actually contains real information that would be worthy to teach and make questions for an examination from. Your task includes first analyzing the text, thinking through whether or not good questions can be made from it. 

End-of-chapter exercises, or questions that appear in the text BUT DO NOT HAVE A PROVIDED ANSWER, are not suitable for question generation, since it should be assumed that the creator of the questions must rely on the information in the text to get their answers.


Following this, at the very end of your response, you will write "Suitable" or "Not suitable". It is imperative that you write one of these two things, as your answer is being automatically processed by a regex, so it must match one of those two strings exactly.


### Instruction:

Text: 
"""
The Project Gutenberg eBook of Through England on a side saddle
    
This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this ebook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: Through England on a side saddle
        In the time of William and Mary


Author: Celia Fiennes

Contributor: Emily W. Griffiths

Release date: November 17, 2023 [eBook #72156]

Language: English

Original publication: London: Simpkin, Marshall & Co.; Hamilton, Adams & Co, 1888

Credits: Steve Mattern, Barry Abrahamsen, and the Online Distributed Proofreading Team at https://www.pgdp.net (This book was produced from images made available by the HathiTrust Digital Library.)


*** START OF THE PROJECT GUTENBERG EBOOK THROUGH ENGLAND ON A SIDE SADDLE ***




                            Through England

                            On a Side Saddle


                    IN THE TIME OF WILLIAM AND MARY


                           BEING THE DIARY OF

                            _CELIA FIENNES._


                             --------------


                        WITH AN INTRODUCTION BY

                        THE HON. MRS GRIFFITHS.




                         ---------------------




                                _LONDON:
                Field & Tuer, The Leadenhall Press, E.C.
            Simpkin, Marshall & Co.; Hamilton, Adams & Co._

                               ----------

          _New York: Scribner & Welford, 743 & 745, Broadway._

                                   --

                                  1888


------------------------------------------------------------------------
"""

### Response:
## Reasoning and thought process:
Step 1. Identify Paragraph Content: This paragraph is from the introductory section of "Through England on a Side Saddle" by Celia Fiennes, including copyright and publication information.
Step 2. Evaluate Educational Relevance: The paragraph contains legal, copyright, and publication information specific to the book's distribution and not its content.
Step 3. Assess Specific Contexts and Formats:
  - Purely Descriptive Lists: The paragraph is a descriptive list of publication details without educational content.
  - Dialogues or Quotes Without Context: Not applicable.
  - Ambiguous or Unclear Texts: The text is clear but lacks educational substance.
Step 4. Assess the Possibility of Formulating Questions: The paragraph's content does not lend itself to educational questions due to its focus on publication rather than substantive content.
Step 5. Determine Suitability for Educational Purposes: The paragraph lacks thematic, historical, scientific, or literary information and is thus unsuitable for creating educational questions.
Step 6. Check for Contextual Completeness: The paragraph, while complete in its context, does not offer educational insights or concepts.
Step 7. Final Judgment: Unsuitable for educational questions.</s>

### Instruction:

Text:
"""
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
"""

### Response:
## Reasoning and thought process:

Step 1. Identify Paragraph Content: The text is a philosophical excerpt from a narrative text, describing Zarathustra's reflections and decisions.
Step 2. Evaluate Educational Relevance: The passage is a philosophical monologue, covering themes like meaning, wisdom, and human condition.
Step 3. Assess Specific Contexts and Formats:
  - Purely Descriptive Lists: Not applicable.
  - Dialogues or Quotes Without Context: The text, though in the form of a monologue, provides sufficient context for understanding.
  - Ambiguous or Unclear Texts: The text is metaphorically deep but not ambiguous; it can be interpreted for educational purposes.
Step 4. Assess the Possibility of Formulating Questions: The passage allows for various educational questions about philosophy, literature, and existential thought.
Step 5. Determine Suitability for Educational Purposes: The text offers substantial material for discussion and analysis, suitable for creating educational questions.
Step 6. Check for Contextual Completeness: The paragraph provides enough context to stand alone for educational discussion.
Step 7. Final Judgment: Suitable for educational questions.


### Instruction:

Text:
"""

1. Apples, 2. Oranges, 3. Bananas, 4. Grapes, 5. Papayas, 6. Pomegranates, 7. Cherries. Sarah wandered through the market. Amidst her thoughts, she overheard a conversation:

"Did you hear about the mayor's decision?" asked John.
"It's quite surprising, isn't it?" replied Emily, her voice tinged with disbelief.
"I know, right? But what can we do?" John sighed.

Sarah continued her shopping, her mind now on the mayor's mysterious decision.
"""

### Response:
## Reasoning and thought process:
Step 1. Identify Paragraph Content: The paragraph appears to be from a narrative, including a list and a snippet of overheard conversation.
Step 2. Evaluate Educational Relevance: The paragraph contains a simple list of fruits and an unrelated, context-less dialogue, offering little educational substance.
Step 3. Assess Specific Contexts and Formats:
  - Purely Descriptive Lists: The list is purely descriptive, providing no educational context or analytical significance. Further, there's no context about what it could be about.
  - Dialogues or Quotes Without Context: The conversation about the mayor's decision lacks context and background, making it difficult to derive any educational content.
  - Ambiguous or Unclear Texts: While not ambiguous, the paragraph's content does not contribute to a clear narrative or concept that has philosophical or educational relevance.
Step 4. Assess the Possibility of Formulating Questions: Due to the lack of contextual or thematic depth, formulating educational questions from this paragraph is challenging.
Step 5. Determine Suitability for Educational Purposes: The paragraph's lack of thematic depth and contextual clarity renders it unsuitable for creating educational questions.
Step 6. Check for Contextual Completeness: The paragraph fails to provide a complete context for the dialogue and the list, hindering any meaningful educational interpretation.
Step 7. Final Judgment: Unsuitable for educational questions.


### Instruction:

Text:
"""
If, then, we represent our earth as a little ball of one inch diameter, the sun would be a big globe nine feet across and 323 yards away, that is about a fifth of a mile, four or five minutes’ walking. The moon would be a small pea two feet and a half from the world.  Between earth and sun there would be the two inner planets, Mercury and Venus, at distances of one hundred and twenty-five and two hundred and fifty  yards from the sun. All round and about these bodies there would be  emptiness until you came to Mars, a hundred and seventy-five feet beyond the earth; Jupiter nearly a mile away, a foot in diameter; Saturn, a little smaller, two miles off; Uranus four miles off and Neptune six miles off. Then nothingness and nothingness except for small particles and drifting scraps of attenuated vapour for thousands of miles.
"""

### Response:
## Reasoning and thought process:
Step 1. Identify Paragraph Content: This describes a scale model of the solar system.
Step 2. Evaluate Educational Relevance: The paragraph contains analogies and information about the solar system's scale and distances between celestial bodies.
Step 3. Assess Specific Contexts and Formats:
  - Purely Descriptive Lists: While the paragraph lists celestial bodies and their distances, it does so to illustrate a scientific concept, providing educational value.
  - Dialogues or Quotes Without Context: Not applicable.
  - Ambiguous or Unclear Texts: The paragraph is clear and offers a concrete concept in astronomy.
Step 4. Assess the Possibility of Formulating Questions: The content enables the creation of educational questions about astronomy, scale, and the solar system's structure.
Step 5. Determine Suitability for Educational Purposes: The paragraph's scientific content makes it suitable for educational questions in astronomy and spatial understanding.
Step 6. Check for Contextual Completeness: The paragraph is contextually complete, allowing for independent understanding of the solar system's scale.
Step 7. Final Judgment: Suitable for educational questions.


### Instruction:

Text:
"""
In the world of science, there are countless mysteries and phenomena that elude easy explanation. For instance, certain forces and energies interact in ways that are not fully understood, shaping the universe in subtle and profound manners. These interactions often occur at levels beyond human perception, leaving much to speculation and theory. Various scientific disciplines attempt to explain these interactions, each offering unique perspectives but often lacking definitive answers. The vastness of these mysteries spans from the minuscule quantum realm to the expansive cosmos, hinting at complexities that challenge our current understanding.
"""

### Response:
## Reasoning and thought process:
Step 1. Identify Paragraph Content: The paragraph discusses the complexity and mystery of scientific phenomena, mentioning interactions of forces and energies in the universe.
Step 2. Evaluate Educational Relevance: The paragraph touches on scientific themes and makes a number of broad claims.
Step 3. Assess Specific Contexts and Formats:
  - Purely Descriptive Lists: Not applicable.
  - Dialogues or Quotes Without Context: Not applicable.
  - Ambiguous or Unclear Texts: The text is ambiguous and lacks clarity. It refers to scientific phenomena and theories without providing specific examples or concrete information, making it difficult to derive clear educational questions.
Step 4. Assess the Possibility of Formulating Questions: Due to its vagueness and lack of specific content, the paragraph does not lend itself well to formulating meaningful educational questions.
Step 5. Determine Suitability for Educational Purposes: The lack of clarity and specificity in the text renders it unsuitable for generating insightful educational questions.
Step 6. Check for Contextual Completeness: The paragraph, while complete in its presentation, fails to provide the necessary detail and clarity to be understood for educational purposes.
Step 7. Final Judgment: Unsuitable for educational questions due to its ambiguous and unclear content.


### Instruction:

Text: 
"""
{text}
"""

Note that even blunt facts can be suitable for questions, and unconventional knowledge is not necessarily unsuitable. Fictional stories that contain strong morals or philosophy can also have good questions made from them. But legal notices and metadata are not suitable. Lists of information without the context needed for the question-maker to understand the text; quotes or dialogues without context or clear depth; or ambiguous content that isn't precise enough to "nail down" a solid question from, are not valid.

### Response:
## Reasoning and thought process (reason intelligently):
'''
            
        prompt_escaped = prompt.replace('{', '{{').replace('}', '}}')
        # 2. Unescape curly braces that are associated with input keys
        for key in arguments.keys():
            prompt_escaped = prompt_escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}") # Somehow this works
        # 3. Format
        prompt_formatted = prompt_escaped.format(**arguments)
        logging.info(f"Formatted prompt for generation: {prompt_formatted}")
        # Submit generation and return response, retrying as needed
        times_tried = 0
        if self.completion_mode:
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper.submit_completion(prompt_formatted, self.sampling_params)
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + filtered_response
                    return ret
                except Exception as e:
                    # logging.error(f"Error in Generation Step: {e}")
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")
        else:
            messages = json.loads(prompt_formatted)
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper.submit_chat(messages, self.sampling_params)
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + [{"role": "assistant", "content": filtered_response}]
                    return ret
                except Exception as e:
                    logging.error(f"Error in Generation Step: {e}")
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")


class EngineWrapper:
    def __init__(self, model, api_key=None, base_url=None):
        # engine_args = AsyncEngineArgs(
        #     model=model,
        #     quantization=quantization,
        #     engine_use_ray=False,
        #     disable_log_requests=True,
        #     max_model_len=12000,
        #     dtype="float16"
        # )
        # self.engine = AsyncAphrodite.from_engine_args(engine_args)

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        # print("\n\nSETTINGS DUMP\n\n")
        # print(self.model)
        # print(prompt)
        # print(sampling_params["temperature"])
        # print(sampling_params["top_p"])
        # print(sampling_params["max_tokens"])
        
        completion = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            stop=sampling_params["stop"],
            max_tokens=sampling_params["max_tokens"],
        )
        completion = completion.choices[0].text
        return prompt + completion
    
    async def submit_chat(
    self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            stop=sampling_params["stop"],
            max_tokens=sampling_params["max_tokens"],
        )
        completion = completion.choices[0].message.content
        return completion

def judge_paragraph_processor(determination): # TODO extract to separate file to avoid muddying the control flow code
    if "unsuitable" in determination.lower():
        return False # control flow has been modified to use the information it has, based on the determination of the output processors
    elif "suitable" in determination.lower():
        return True


async def determine_worthy(
    idx,
    p,
    judged_worthy_for_questions,
    output_dir,
    judge: GenerationStep,
):
    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)
    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append((None, data[7:])) # hacky way of appending only the text name. See the file output of a failed judgement for details (Takes after "failed|")
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
    else:
        judgement = await judge.generate(
            arguments={
                "text": p[0],
                "textname": p[1]
            }
        )
        to_append = (None,p[1])
        if judgement:
            to_append = (p[0],p[1])
            
        judged_worthy_for_questions.append(to_append)

        # Prepare the data to be written to the file
        if judgement:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": to_append[0], "metadata": to_append[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{to_append[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")

async def filter_all_questions(
    paragraphs_processed,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    take_subset=False,
    use_filenames=False,
    rtwl=None,
    completion_mode=True,
    logging_level=None
):
    
    if use_filenames:
        prompt_path = "judge_paragraph_filenames.txt"
    else:
        prompt_path = "judge_paragraph_no_filenames.txt"

    judgement_regex = re.compile(
            r"Reasoning and thought process \(reason intelligently\):(.+)", re.DOTALL | re.IGNORECASE,)
    
    judge = GenerationStep(
        prompt_path=prompt_path,
        regex=judgement_regex,
        sampling_params={
            "max_tokens": 6000,
            # "min_p": 0.4,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
            ],
            "temperature": 0.2,
        },
        completion_mode=completion_mode,
        retries=2,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level, # TODO change to warning
        output_processor=judge_paragraph_processor,
        return_input_too=False
    )
    if not take_subset:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                output_dir,
                judge
            )
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                output_dir,
                judge
            )
            for idx, p in enumerate(paragraphs_processed[:13])
        ]
    limited_tasks = [rtwl(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks):
        await future

paragraphs_processed = [('\ufeffThe Project Gutenberg eBook of Simple Sabotage Field Manual   This ebook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this ebook or online at www.gutenberg.org. If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook. Title: Simple Sabotage Field Manual  Author: United States. Office of Strategic Services Release date: August 4, 2008 [eBook #26184]         Most recently updated: February 28, 2022 Language: English    [Illustration]   Simple Sabotage Field Manual Office of Strategic Services   OSS REPRODUCTION BRANCH SIMPLE SABOTAGE FIELD MANUAL Strategic Services (Provisional) STRATEGIC SERVICES FIELD MANUAL No. 3   Office of Strategic Services Washington, D. C. 17 January 1944  This Simple Sabotage Field Manual Strategic Services (Provisional) is published for the information and guidance of all concerned and will be used as the basic doctrine for Strategic Services training for this subject. The contents of this Manual should be carefully controlled and should not be allowed to come into unauthorized hands.', './raw_txt_input/Simple Sabotage, by the Office of Strategic Services, published 1944'), ('The instructions may be placed in separate pamphlets or leaflets according to categories of operations but should be distributed with care and not broadly. They should be used as a basis of radio broadcasts only for local and special cases and as directed by the theater commander. AR 380-5, pertaining to handling of secret documents, will be complied with in the handling of this Manual. [Illustration] William J. Donovan   CONTENTS  1. INTRODUCTION 2. POSSIBLE EFFECTS 3. MOTIVATING THE SABOTEUR 4. TOOLS, TARGETS, AND TIMING 5. SPECIFIC SUGGESTIONS FOR SIMPLE SABOTAGE   1. INTRODUCTION  The purpose of this paper is to characterize simple sabotage, to outline its possible effects, and to present suggestions for inciting and executing it. Sabotage varies from highly technical _coup de main_ acts that require detailed planning and the use of specially-trained operatives, to innumerable simple acts which the ordinary individual citizen-saboteur can perform. This paper is primarily concerned with the latter type. Simple sabotage does not require specially prepared tools or equipment; it is executed by an ordinary citizen who may or may not act individually and without the necessity for active connection with an organized group; and it is carried out in such a way as to involve a minimum danger of injury, detection, and reprisal.', './raw_txt_input/Simple Sabotage, by the Office of Strategic Services, published 1944'), ('Where destruction is involved, the weapons of the citizen-saboteur are salt, nails, candles, pebbles, thread, or any other materials he might normally be expected to possess as a householder or as a worker in his particular occupation. His arsenal is the kitchen shelf, the trash pile, his own usual kit of tools and supplies. The targets of his sabotage are usually objects to which he has normal and inconspicuous access in everyday life. A second type of simple sabotage requires no destructive tools whatsoever and produces physical damage, if any, by highly indirect means. It is based on universal opportunities to make faulty decisions, to adopt a noncooperative attitude, and to induce others to follow suit. Making a faulty decision may be simply a matter of placing tools in one spot instead of another. A non-cooperative attitude may involve nothing more than creating an unpleasant situation among one’s fellow workers, engaging in bickerings, or displaying surliness and stupidity. This type of activity, sometimes referred to as the “human element,” is frequently responsible for accidents, delays, and general obstruction even under normal conditions. The potential saboteur should discover what types of faulty decisions and the operations are _normally_ found in this kind of work and should then devise his sabotage so as to enlarge that “margin for error.”   2. POSSIBLE EFFECTS  Acts of simple sabotage are occurring throughout Europe.', './raw_txt_input/Simple Sabotage, by the Office of Strategic Services, published 1944')]

semaphore = asyncio.Semaphore(10)

async def run_task_with_limit(task):
    async with semaphore:
        # Run your task here
        return await task

judged_worthy_for_questions = []
output_dir = "./"


async def callable_function(): # simulates jupyter cell so that async works
    await filter_all_questions(paragraphs_processed, 
                         judged_worthy_for_questions,
                         EngineWrapper("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",api_key="b04e78e55992ac6722f71e2301e897fbfe816290944c116778a564e3f0cc4eae",
                         base_url="https://api.together.xyz"),
                         output_dir,
                         take_subset=False,
                         use_filenames=False,
                         rtwl=run_task_with_limit,
                         completion_mode=True,
                         logging_level=logging.INFO
                         )
    print(judged_worthy_for_questions)
    
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(callable_function())
    finally:
        loop.close()
    