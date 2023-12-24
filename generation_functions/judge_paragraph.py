import re
from .judge_paragraph_grammar import judge_paragraph_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

def judge_paragraph(p,logic_llm):
    reached_decision = False
    max_retries = 0
    while (not reached_decision and (max_retries <= 3)):
        decision_prompt = f"""You are an expert educational AI that will make a determination as to whether the contents of the paragraph(s) provided are suitable for making educational questions based off of them; these questions should be able to test the knowledge in in the book. The book in question is {p[1]}, and you should keep this in mind when considering what kind of questions should be capable of being developed. If there is sufficiently deep information to make questions about, you will judge it suitable, even if the knowledge being tested does not reflect typical curricula. Essentially: you will determine if provided text is a table of contents, introductory paragraph for a book, etc., or if it actually contains real information that would be worthy to teach and make questions for an examination from. Your task includes first analyzing the text, thinking through whether or not good questions can be made from it. 



Following this, at the very end of your response, you will write "Suitable" or "Not suitable". It is imperative that you write one of these two things, as your answer is being automatically processed by a regex, so it must match one of those two strings exactly.


### Instruction:
Text details: Through England On a Side Saddle, by Celia Fiennes

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
Step 7. Final Judgment: Unsuitable for educational questions.


### Instruction:
Text details: Thus Spake Zaranthustra, by Friedrich Nietzsche

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

### Response:
## Reasoning and thought process:

Step 1. Identify Paragraph Content: The text is a philosophical excerpt from "Thus Spake Zarathustra" by Friedrich Nietzsche, describing Zarathustra's reflections and decisions.
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
Text details: A Bizarre Day in the Life of Sarah Anderson, by Felicia Anderson

Text:
\"\"\"

1. Apples, 2. Oranges, 3. Bananas, 4. Grapes, 5. Papayas, 6. Pomegranates, 7. Cherries. Sarah wandered through the market. Amidst her thoughts, she overheard a conversation:

"Did you hear about the mayor's decision?" asked John.
"It's quite surprising, isn't it?" replied Emily, her voice tinged with disbelief.
"I know, right? But what can we do?" John sighed.

Sarah continued her shopping, her mind now on the mayor's mysterious decision.
\"\"\"

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
Text details: The Scale of the Solar System, by Rob Robertson

Text:
\"\"\"
If, then, we represent our earth as a little ball of one inch diameter, the sun would be a big globe nine feet across and 323 yards away, that is about a fifth of a mile, four or five minutes’ walking. The moon would be a small pea two feet and a half from the world.  Between earth and sun there would be the two inner planets, Mercury and Venus, at distances of one hundred and twenty-five and two hundred and fifty  yards from the sun. All round and about these bodies there would be  emptiness until you came to Mars, a hundred and seventy-five feet beyond the earth; Jupiter nearly a mile away, a foot in diameter; Saturn, a little smaller, two miles off; Uranus four miles off and Neptune six miles off. Then nothingness and nothingness except for small particles and drifting scraps of attenuated vapour for thousands of miles.
\"\"\"

### Response:
## Reasoning and thought process:
Step 1. Identify Paragraph Content: This paragraph is from Rob Robertson's "The Scale of the Solar System," describing a scale model of the solar system.
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
Text details: The Scale of the Solar System, by Rob Robertson

Text:
\"\"\"
In the world of science, there are countless mysteries and phenomena that elude easy explanation. For instance, certain forces and energies interact in ways that are not fully understood, shaping the universe in subtle and profound manners. These interactions often occur at levels beyond human perception, leaving much to speculation and theory. Various scientific disciplines attempt to explain these interactions, each offering unique perspectives but often lacking definitive answers. The vastness of these mysteries spans from the minuscule quantum realm to the expansive cosmos, hinting at complexities that challenge our current understanding.
\"\"\"

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
Text details: {p[1]}

Text: 
\"\"\"
{p[0]}
\"\"\"

Note that even blunt facts can be suitable for questions, and unconventional knowledge is not necessarily unsuitable. Fictional stories that contain strong morals or philosophy can also have good questions made from them. But legal notices and metadata are not suitable. Lists of information without the context needed for the question-maker to understand the text; quotes or dialogues without context or clear depth; or ambiguous content that isn't precise enough to "nail down" a solid question from, are not valid.

### Response:
## Reasoning and thought process (reason intelligently):
"""
        # print("DEBUG\n\n" + decision_prompt)
        completion = logic_llm(decision_prompt, max_tokens=6000, grammar=judge_paragraph_grammar, stop=["</s>","# Input:"], echo=True,temperature=0.2)["choices"][0]["text"]

        # print("DEBUG\n\n")
        # print(completion)
        response_pattern = re.compile(r"Reasoning and thought process \(reason intelligently\):(.+)", re.DOTALL | re.IGNORECASE)
        
        judgement_pattern = re.compile(r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE)
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
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,
                      n_ctx=7000,
                      rope_freq_scale=0.33,
                      n_gpu_layers=100,
                      verbose=True,
                      # n_gqa=8
                      )
    right = 0
    d1 = judge_paragraph(test,logic_llm)
    if d1[0]:
        right += 1
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
    
    d2 = judge_paragraph(test2,logic_llm)
    if not d2[0]:
        right += 1
    
    test3 = ("""A communi observantia non est recedendum; et minime mutandæ sunt quæ
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
     expounder, exposition, or explanation.""",'A Collection of latin maxims & rules in law and equity, by Peter Halkerston')
    
    d3 = judge_paragraph(test3,logic_llm)
    if d3[0]:
        right += 1
        
    print("Num right: ", right)