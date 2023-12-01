import re
from .judge_paragraph_grammar import judge_paragraph_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

def judge_paragraph(p,logic_llm):
    reached_decision = False
    max_retries = 0
    while (not reached_decision and (max_retries <= 3)):
        decision_prompt = f"""# Input:
You are an expert educational AI that will make a determination as to whether the contents of the paragraph(s) provided are suitable for making educational questions based off of them; these questions should be able to test the knowledge in in the book. The book in question is {p[1]}, and you should keep this in mind when considering what kind of questions should be capable of being developed. If there is sufficiently deep information to make questions about, you will judge it suitable, even if the knowledge being tested does not reflect typical curricula. Essentially: you will determine if provided text is a table of contents, introductory paragraph for a book, etc., or if it actually contains real information that would be worthy to teach and make questions for an examination from. Your task includes first analyzing the text, thinking through whether or not good questions can be made from it. 



Following this, at the very end of your response, you will write "Suitable" or "Not suitable". It is imperative that you write one of these two things, as your answer is being automatically processed by a regex, so it must match one of those two strings exactly.

# Input
## Instruction:
Text: 
\"\"\"
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
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Identify paragraph content: the paragraph seems to be the introductory section of an eBook, including copyright and publication information.
Step 2. Evaluate educational relevance: The paragraph primarily contains legal and copyright information, publication details, and credits. This information is specific to the book's distribution and legal use, not its educational content.
Step 3. Assess the possibility of formulating questions: The information provided does not offer educational substance or content that could be the basis for formulating educational questions. It lacks any thematic, historical, scientific, or literary analysis or data.
Step 4. Determine suitability for educational purposes: Given that the paragraph is focused on copyright, legal usage, and publication details without providing substantive educational content or context, it is not suitable for creating educational questions.
Step 5. Final judgement: Unsuitable.

# Input:
## Instruction:
Text:
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

# Response:
## Reasoning and thought process:

Step 1. Identify paragraph content: The text appears to be an excerpt from a fictional or philosophical work. It describes the character Zarathustra's decision to leave his home and his contemplations on wisdom, life, and his purpose.
Step 2. Evaluate educational relevance: The passage is rich in philosophical and literary content. It provides a foundation for exploring themes such as the search for meaning, the nature of wisdom, and the human condition.
Step 3. Assess the possibility of formulating questions: This text allows for the creation of various educational questions. These can range from analyzing Zarathustra's character, his motivation for going down, and the role of acknowledgement in effort. Questions can also focus on literary analysis, like exploring the symbolism of the sun and the metaphorical significance of Zarathustra's descent.
Step 4. Determine suitability for educational purposes: Given the philosophical depth and literary richness of the text, it is suitable for creating educational questions. It offers substantial material for discussion and analysis in fields such as philosophy, literature, and existential thought.
Step 5. Final judgement: Suitable.

# Input:
## Instruction:
Text: 
\"\"\"
{p[0]}
\"\"\"

Note that even blunt facts can be suitable for questions, and unconventional knowledge is not necessarily unsuitable. Fictional stories that contain strong morals or philosophy can also have good questions made from them. But legal notices and metadata are not suitable.
# Response:
## Reasoning and thought process (reason intelligently):
"""
        # print("DEBUG\n\n" + decision_prompt)
        completion = logic_llm(decision_prompt, max_tokens=4000, grammar=judge_paragraph_grammar, stop=["</s>"], echo=True,temperature=0.2)["choices"][0]["text"]

        # print("DEBUG\n\n")
        # print(completion)
        response_pattern = re.compile(r"Reasoning and thought process \(reason intelligently\):(.+)", re.DOTALL | re.IGNORECASE)
        
        judgement_pattern = re.compile(r"Final Judgement:(.+)", re.DOTALL | re.IGNORECASE)
        try:
            response = response_pattern.search(completion).group(1)
            print(response)
            determination = judgement_pattern.search(response).group(1)
            print("\n\nDETERMINATION:\n------")
            print(determination)
            print("\n---------\n")
            if "unsuitable" in determination.lower():
                reached_decision=True
                return (None,p[1])
            elif "suitable" in determination.lower():
                return (p[0],p[1])
        except:
            pass
        max_retries += 1        
        
        
if __name__ == "__main__":
    test = ('Slowly by degrees as one million of years followed another, this fiery scene would lose its eruptive incandescence.  The vapours in the sky would rain down and become less dense overhead; great slaggy cakes of solidifying rock would appear upon the surface of the molten sea, and sink under it, to be replaced by other floating masses.  The sun and moon growing now each more distant and each smaller, would rush with diminishing swiftness across the heavens. The moon now, because of its smaller size, would be already cooled far below incandescence, and would be alternately obstructing and reflecting the sunlight in a series of eclipses and full moons.\n\nAnd so with a tremendous slowness through the vastness of time, the earth would grow more and more like the earth on which we live, until at last an age would come when, in the cooling air, steam would begin to condense into clouds, and the first rain would fall hissing upon the first rocks below.  For endless millenia the greater part of the earth’s water would still be vaporized in the atmosphere, but there would now be hot streams running over the crystallizing rocks below and pools and lakes into which these streams would be carrying detritus and depositing sediment.',
 'A Short History of the World, by HG Wells, published 1922')
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000)
    judge_paragraph(test,logic_llm)
    
    test2 = ("""

  A

  COLLECTION

  OF

  LATIN MAXIMS & RULES,

  IN LAW AND EQUITY,

  SELECTED FROM THE MOST EMINENT AUTHORS, ON THE
  CIVIL, CANON, FEUDAL, ENGLISH AND SCOTS LAW,

  WITH

  AN ENGLISH TRANSLATION,

  AND

  AN APPENDIX

  OF REFERENCE TO THE AUTHORITIES FROM WHICH
  THE MAXIMS ARE SELECTED.

  BY

  PETER HALKERSTON, L.L.D.
  AUTHOR OF “THE COMPENDIUM OF THE FACULTY COLLECTION
  OF DECISIONS, &C.”


  PRINCIPLES, CAUSES, AND ELEMENTS, BEING UNKNOWN,
  THE SCIENCE WHEREOF THEY ARE, IS ALTOGETHER
  UNKNOWN.
            FORTESCUE b. 8.


  EDINBURGH:

  PRINTED FOR JOHN ANDERSON AND CO. ROYAL EXCHANGE;
  MACREDIE, SKELLY, AND CO. 34, PRINCES STREET; AND
  CHARLES HUNTER, LAW BOOKSELLER, LONDON.

  1823.


  Wm. BAYNE, Printer,
  James’s Court, Edinburgh.

""",'Collection of latin maxims & rules in law and equity')