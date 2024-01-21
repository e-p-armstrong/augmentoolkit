import re
from .check_qatuple_context_grammar import check_qatuple_context_grammar
from .constants import LOGICAL_MODEL
from aphrodite import SamplingParams


def extract_question_answer(response):
    # Define the regex pattern to match the question and answer
    pattern = r"### Question Rewording \(using text details as reference\):\nQuestion: (.+?)\nAnswer: (.+)"

    # Search for the pattern in the response
    match = re.search(pattern, response)

    # Extract and return the question and answer if a match is found
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    else:
        print("Returned none, failed to match")
        return None, None


# A separate prompt for the reword step of checking qatuple context, since the grammar is bugged on the original
async def check_qatuple_context(qatuple, engine_wrapper):
    retries = 0
    while retries <= 4:
        decision_prompt = f"""<s> [INST] You are checking whether a provided question and answer make sense if asked by themselves, with no additional information. You need to check for vague wording that a reader cannot interpret correctly, and questions that lack key context and would not be possibly answerable even if asked of someone with complete, masterful knowledge of the general subject matter of the question.

Evaluate the provided question-answer pair step-by-step. Following this, at the very end of your response, your "final judgment" or "final answer", you will write "Pass" or "Fail" or "Reword". A test passes if it "makes sense" and does not lack key context; it "Fails" if it lacks key context, AND the question is not specific or clear, it fails. If it lacks context but the question is specific, pointed, and grounded, then it needs to be reworded to have the context-needing terms (i.e., vague reference to "the text") removed. If it has no problems, it passes. 

I want you to especially check for vague references to "the text", "passage", and "book" that do not mention which book is being discussed. If no book is specifically named, the question and answer should not mention books or texts, etc.

Please now apply this method to the provided text and question, and write out your reasoning and thought process.


### Instruction:
Text details: Introduction to Study in the Field of Chemistry, by Greg Bukele, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What is the main theme of this book?
Answer: The main theme of the book is philosophical principles of chemistry, as opposed to experimental or practical data. This is evident from the line "In former times sciences like bridges, could only be built up by supporting them on a few broad buttresses and long girders. In addition to the exposition of the principles of chemistry, it has been my desire to show how science has now been built up like a suspension bridge, supported by the united strength of a number of slender, but firmly-fixed, chains, which individually are of little strength, and has thus been carried over difficulties which before appeared insuperable.\" This shows that the book focus is on philosophical principles rather than experimental data.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. This question asks about "this book" without stating which book this is.
#### Text and Author Specificity: The question asks what the main theme of "this book" is, without mentioning which book is being referred to. It does not ask about any "author" or similar without providing context. Because of the vague reference to "this book", this fails.
#### Scope and Precision: The question asks about the "main theme" of the entire book, which is a broad question. Someone could feasibly know what the "main theme" of the book is, but when paired with a vague reference to "this book", this question is both vague and broad, and thus fails this step.

### Answer Context Validation:
#### Special Term Context Check: specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. This answer references "the book" without stating which book it is talking about.
#### Specificity and Clarity: The answer states what the main theme of "the book" is, without mentioning which book is being referred to. It does not reference any "author" or similar without providing context.  Because of the vague reference to "this book", this fails.
#### Answer-Only Context Issues: Confirm that the answer does not introduce new and vague context that the question does not. The answer mentions "the book", but this was already introduced by the question. The question's doing so was flawed, but the answer still does not introduce new, vague context that the question does not. So this passes.

### Critical Evaluation and Final Judgment:
#### Evaluation: Some checks related to the question or answer failed. So this question and answer should be reworded if they can be, or fail otherwise. "What is the main theme of this book" asks a specific thing about a specific object (the book) so the question is precise. The question (and the answer) only lack context in mentioning *which* book they refer to. Therefore they will both be reworded.
#### Final judgement: Reword.

### Question Rewording (using text details as reference):
Question: What is the main theme of Principles of Chemistry, by Demitry Mendeleev?
Answer: The main theme of Principles of Chemistry is philosophical principles of chemistry, as opposed to experimental or practical data. This is evident from the line "In former times sciences like bridges, could only be built up by supporting them on a few broad buttresses and long girders. In addition to the exposition of the principles of chemistry, it has been my desire to show how science has now been built up like a suspension bridge, supported by the united strength of a number of slender, but firmly-fixed, chains, which individually are of little strength, and has thus been carried over difficulties which before appeared insuperable.\" This shows that focus of Principles of Chemistry is on philosophical principles rather than experimental data.</s> [INST]


### Instruction:
Text details: Principles of Chemistry, by Demitry Mendeleev, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What does Mendeleev consider important about solutions?
Answer: He considers them an unsolved subject that he cannot ignore in his book, despite the lack of proof for his own theory on their nature.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: This question does not use terms like "book", "text", "passage", or "excerpt" without context, as it directly asks about Mendeleev's view on a specific topic.
#### Text and Author Specificity: The question is specific and clear about what it's asking regarding Mendeleev's views on solutions.
#### Scope and Precision: The question is precise and focused, asking about Mendeleev's considerations on a specific scientific topic.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer references "his book" but fails to specify which book it is referring to.
#### Specificity and Clarity: While the answer is clear on Mendeleev's view, it lacks specificity by not naming the book 'Principles of Chemistry'.
#### Answer-Only Context Issues: The answer introduces a vague reference to "his book", which is not specified in the question, thus lacking context.

### Critical Evaluation and Final Judgment:
#### Evaluation: Some checks related to the question or answer failed. So this question and answer should be reworded if they can be, or fail otherwise. The question is precise and relevant, but the answer introduces a lack of context by not specifying the book. This requires a rewording of the answer to include the specific reference. The question passed its checks and will remain untouched.
#### Final judgment: Reword.

### Question Rewording (using text details as reference):
Question: What does Mendeleev consider important about solutions?
Answer: Mendeleev considers solutions an unsolved subject that he cannot ignore in his book Principles of Chemistry, despite the lack of proof for his own theory on their nature.</s> [INST]


### Instruction:
Text details: Principles of Chemistry, by Demitry Mendeleev, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What is the main theme of this passage?
Answer: The main theme of this passage is the principle that learning scientists should study the latest literature and discoveries of their field.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. This question asks about "this passage" without stating which passage this is (or what book it belongs to).
#### Text and Author Specificity: The question fails to mention which passage or book is being referred to, leading to a lack of context.
#### Scope and Precision: The question is specific in asking about the "main theme" but lacks the necessary context to be answerable, as it does not specify the passage it refers to.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer refers to "this passage" but does not specify which passage it is discussing.
#### Specificity and Clarity: The answer is clear about the theme it is discussing but fails to clarify which specific passage from 'Principles of Chemistry' it is referring to.
#### Answer-Only Context Issues: The answer does not introduce new vague context that the question does not. However, both the question and answer lack necessary context.

### Critical Evaluation and Final Judgment:
#### Evaluation: Some checks related to the question or answer failed. So this question and answer should be reworded if they can be, or fail otherwise. Both the question and answer lack specific context about the "passage", making it impossible to determine which passage from 'Principles of Chemistry' they are referring to. The question is precise in asking for a main theme but fails due to lack of context. Since context cannot be determined, even knowing what book the question is asking about, context cannot be added with rewording. 
#### Final judgment: Fail.</s> [INST]


### Instruction:
Text details: Simple Sabotage, by the Office of Strategic Services, Published 1944
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: How can you avoid blame for an act of sabotage, according to the text?
Answer: You can do them in public places where anyone would have been capable of carrying out the act.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. This question mentions "the text" without specifying which text it is referring to.
#### Text and Author Specificity: The question fails to mention which passage or book is being referred to, leading to a lack of context.
#### Scope and Precision: The question is specific in asking how to avoid blame for an act of sabotage.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer does not use any of these terms.
#### Specificity and Clarity: The answer is clear in its message and content. It does not introduce any vague context.
#### Answer-Only Context Issues: The answer is straightforward, makes no reference to outside material, and only requires knowledge to provide -- therefore, it is completely valid.

### Critical Evaluation and Final Judgment:
#### Evaluation: Some checks related to the question or answer failed. So this question and answer should be reworded if they can be, or fail otherwise. Both the question and answer are precise, but the question lacks specific context regarding the text it refers to. This necessitates a rewording to include the specific text. The answer passed its checks and will remain untouched.
#### Final judgment: Reword.

### Question Rewording (using text details as reference):
Question: How can you avoid blame for an act of sabotage, according to 'Simple Sabotage' by the Office of Strategic Services?
Answer: You can do them in public places where anyone would have been capable of carrying out the act.</s> [INST]


### Instruction:
Text details: Principles of Chemistry, by Demitry Mendeleev, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What was the basis of Mendeleev's work in his book?
Answer: The periodic law.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. This question uses the term "his book" without specifying which book it is referring to.
#### Text and Author Specificity: The question fails to provide the necessary context as it does not mention the specific book 'Principles of Chemistry' by Mendeleev.
#### Scope and Precision: The question is precise in asking about the basis of Mendeleev's work.

### Answer Context Validation:
#### Special Term Context Check: The answer does not use vague terms like "book", "text", "passage", or "excerpt" without context.
#### Specificity and Clarity: The answer clearly states 'The periodic law' as the basis of Mendeleev's work. It provides a straightforward answer without introducing any vague context.
#### Answer-Only Context Issues: The answer is valid as it stands since it requires only knowledge to answer and doesn't reference unspecified external material.

### Critical Evaluation and Final Judgment:
#### Evaluation: The question is precise but lacks specific context regarding the text it refers to. The answer is valid as it is clear and doesn't introduce any vague references. Therefore, the question requires rewording for context, while the answer remains the same.
#### Final judgment: Reword.

### Question Rewording (using text details as reference):
Question: What was the basis of Mendeleev's work in 'Principles of Chemistry'?
Answer: The periodic law.</s> [INST]


### Instruction:
Text details: Principles of Chemistry, by Demitry Mendeleev, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What does Demitry Mendeleev say about inquiry in 'Principles of Chemistry'?
Answer: Inquiry should be encouraged, and dissatisfied with speculative reasonings alone. It should subject every idea to experiment.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The question does not misuse terms like "book", "text", "passage", or "excerpt" without proper context.
#### Text and Author Specificity: The question specifically mentions 'Principles of Chemistry' by Demitry Mendeleev, providing clear context.
#### Scope and Precision: The question is precise and specific enough for someone familiar with the text to answer it.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer does not use vague terms like "book", "text", "passage", or "excerpt" without context.
#### Specificity and Clarity: The answer is clear and directly addresses the question. It provides specific information without needing additional context.
#### Answer-Only Context Issues: The answer is straightforward and does not introduce any vague or unspecified external material.

### Critical Evaluation and Final Judgment:
#### Evaluation: The question and answer both pass all checks for context, specificity, precision, and clarity.
#### Final judgment: Pass.</s> [INST]


### Instruction:
Text details: Principles of Chemistry, by Demitry Mendeleev, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: How does science advance, according to Demitry Mendeleev's text?
Answer: Science advances through discovering new truths and practical results.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The question mentions "Mendeleev's text" but does not specify which text it is referring to.
#### Text and Author Specificity: The question fails to provide the necessary context as it references "the text" but provides no indication of which text this is.
#### Scope and Precision: The question is precise in asking how science advances according to Mendeleev.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer does not use vague terms like "book", "text", "passage", or "excerpt" without context.
#### Specificity and Clarity: The answer is clear and provides a direct response but is based on the vague reference in the question.
#### Answer-Only Context Issues: The answer does not introduce any new vague context and is valid as it stands.

### Critical Evaluation and Final Judgment:
#### Evaluation: The question is precise but lacks specific context regarding the text it refers to. The answer is valid as it is. Therefore, the question requires rewording for context, while the answer remains the same.
#### Final judgment: Reword.

### Question Rewording (using text details as reference):
Question: What was the basis of Mendeleev's work in 'Principles of Chemistry'?
Answer: The periodic law.


### Instruction:
Text details: Simple Sabotage, By the Office of Strategic Services, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What are some ways information can be spread, according to the Office of Strategic Services?
Answer: Various media may be used to disseminate suggestions and information regarding simple sabotage. Among these are radio broadcasts or leaflets, which may be directed towards specific areas or general in scope. Agents may also be trained in the art of simple sabotage.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The question does not misuse terms like "book", "text", "passage", or "excerpt" without proper context.
#### Text and Author Specificity: The question specifically mentions the Office of Strategic Services, providing clear context.
#### Scope and Precision: The question is precise and specific enough for someone familiar with the works of the Office of Strategic Services to answer.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer does not use vague terms without context.
#### Specificity and Clarity: The answer is clear and directly addresses the question. It provides specific information without needing additional context.
#### Answer-Only Context Issues: The answer is straightforward and does not introduce any vague or unspecified external material.

### Critical Evaluation and Final Judgment:
#### Evaluation: The question and answer both pass all checks for context, specificity, precision, and clarity.
#### Final judgment: Pass.</s> [INST]


### Instruction:
Text details: Simple Sabotage, By the Office of Strategic Services, Published 1867
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: How does the type of saboteur affect their role in destruction?
Answer: If they are a technician, they can devise methods of simple sabotage appropriate to their facilities. If not technically trained, they need suggestions for what to destroy and how to accomplish it.

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The question does not misuse any specific terms without proper context.
#### Text and Author Specificity: The question does not require a reference to a specific text or author as it is asking about general knowledge.
#### Scope and Precision: The question is precise in asking about the impact of a saboteur's type on their role.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer does not use vague terms without context.
#### Specificity and Clarity: The answer is clear, providing a direct response to the question based on general knowledge.
#### Answer-Only Context Issues: The answer does not introduce any vague or unspecified external material and is valid as it stands.

### Critical Evaluation and Final Judgment:
#### Evaluation: Both the question and answer are precise and do not require additional context for understanding.
#### Final judgment: Pass.</s> [INST]


### Instruction:
Text details: Introduction to Philosophy, by George Stuart Fullerton
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: What is the meaning of this passage?
Answer: This passage means that things which think, form plans, and act on those plans, are beyond simple machines. This is evidenced by the line "Creatures that think, form plans, and _act_, are not what we call automata."

[/INST]### Response:
## Reasoning and thought process:
### Question Context Validation
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The question asks about "this passage" without specifying which passage it is referring to or what book it belongs to.
#### Text and Author Specificity: The question fails to mention which passage or book is being referred to, leading to a lack of context.
#### Scope and Precision: The question asks about the meaning of a passage, which is a precise inquiry. However, when paired with a vague reference to "this passage", the question becomes too vague and broad, thus failing this step.

### Answer Context Validation:
#### Special Term Context Check: Specifically check for use of the terms "book", "text", "passage", and "excerpt" without context about which specific thing is being discussed. The answer references "this passage" without stating which passage it is talking about.
#### Specificity and Clarity: The answer states what the passage means but fails to clarify which specific passage from 'Introduction to Philosophy' by George Stuart Fullerton it is referring to.
#### Answer-Only Context Issues: The answer does not introduce new vague context that the question does not. However, both the question and answer lack necessary context.

### Critical Evaluation and Final Judgment:
#### Evaluation: Both the question and answer lack specific context, making it impossible to determine which passage from 'Introduction to Philosophy' they are referring to. The question is precise in asking for a meaning but fails due to lack of context.
#### Final judgment: Fail.</s> [INST]


### Instruction:
Text details: {qatuple[3]}
Note that while you have access to this information, for the sake of rewording questions, you should evaluate the question as if you could not see this.

Question: {qatuple[0]}
Answer: {qatuple[1]}

[/INST]### Response:
## Reasoning and thought process (be thorough):
"""
        # print("DEBUG\n\n" + prompt=decision_prompt)
        sampling_params = SamplingParams(max_tokens=10000,stop=["</s>", "# Input:", "[INST]","### Instruction"],temperature=0.2)
        try:
            completion = await engine_wrapper.submit(
                    decision_prompt,
                    sampling_params
                )

            # print("DEBUG\n\n")
            # # print(completion)

            response_pattern = re.compile(
                r"Reasoning and thought process \(be thorough\):(.+)",
                re.DOTALL | re.IGNORECASE,
            )
            response = response_pattern.search(completion).group(1).strip()
            decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
            # print(response)
            determination = decision_pattern.search(response).group(1).strip()
            #print("\n\nDETERMINATION:\n------")
            #print(determination)
            #print("\n---------\n")
            if "pass" in determination.lower():
                print("Leaving be...")
                return (True, response), completion
            elif "reword" in determination.lower():
                print("Rewording...")
                q, a = extract_question_answer(response)
                print((q, a, qatuple[2], qatuple[3]))
                return (q, a, qatuple[2], qatuple[3]), completion
            elif "fail" in determination.lower():
                print("Setting to None...")
                return (False, response), completion
            else:
                print("Did not contain relevant or irrelevant! Retrying")
                retries += 1
        except Exception as e:
            print("Exception!", e)
            if retries <= 4:
                retries += 1
            else:
                return (None, None), None
    return (None, None), None


# There is no bug about this ignoring certain judgments and retrying; that's just the dissenting reasoning from the print statement

if __name__ == "__main__":  # test
    logic_llm = Llama(
        model_path=LOGICAL_MODEL,
        n_gqa=8,
        offload_kqv=True,
        n_ctx=8000,
        n_gpu_layers=100,
    )  # load the logical LLM and offload everything
    # NOTE: change these examples to have actual body text if you end up incorporating that into this step
    q_test = [
        (
            "What is the central philosophy presented in this book?",
            "The central philosophy is Stoicism, which advocates for living in harmony with nature and understanding that human happiness depends not on external events but on our own internal attitude and actions.",
            "fucking gauls",
            "Meditations, by Marcus Aurelius, Published 180 AD",
        ),
        (
            "What does the author argue in this part of the book?",
            "Plato argues for the philosopher-king as the ideal ruler, who possesses both wisdom and moral virtue.",
            "fucking sophists",
            "The Republic, by Plato",
        ),
        (
            "How does Darwin explain natural selection?",
            "Darwin explains natural selection as a process where organisms better adapted to their environment tend to survive and produce more offspring. This theory suggests that traits beneficial for survival are more likely to be passed on to subsequent generations.",
            "fucking creationists",
            "The Origin of Species, by Charles Darwin",
        ),
    ]

    print("Begin variety test")
    # Try to detect bad question
    d = check_qatuple_context(q_test[0], logic_llm)
    if d[0] == "New QA Tuple":  # if not relevant
        print("Made right choice for rewordable question")
    else:
        print("Made wrong choice for rewordable question")
    d2 = check_qatuple_context(q_test[1], logic_llm)
    if not d2[1]:
        print("Made right choice for bad question")
    else:
        print("Made wrong choice for bad question")
    d3 = check_qatuple_context(q_test[2], logic_llm)
    if d3[2]:
        print("Made right choice for good question")
    else:
        print("Made wrong choice for good question")

    print("Begin Mendeleev test")
    # NOTE I should actually do a mendeleev test, to see if including examples from that text has screwed it
    text2 = """A substance or material is that which occupies space and has
      weight; that is, which presents a mass attracted by the earth and
      by other masses of material, and of which the _objects_ of nature
      are composed, and by means of which the motions and _phenomena_
      of nature are accomplished. It is easy to discover by examining
      and investigating, by various methods, the objects met with
      in nature and in the arts, that some of them are homogeneous,
      whilst others are composed of a mixture of several homogeneous
      substances. This is most clearly apparent in solid substances.
      The metals used in the arts (for example, gold, iron, copper)
      must be homogeneous, otherwise they are brittle and unfit for
      many purposes. Homogeneous matter exhibits similar properties in
      all its parts. By breaking up a homogeneous substance we obtain
      parts which, although different in form, resemble each other in
      their properties. Glass, pure sugar, marble, &c., are examples of
      homogeneous substances. Examples of non-homogeneous substances
      are, however, much more frequent in nature and the arts. Thus
      the majority of the rocks are not homogeneous. In porphyries
      bright pieces of a mineral called 'orthoclase' are often seen
      interspersed amongst the dark mass of the rock. In ordinary red
      granite it is easy to distinguish large pieces of orthoclase mixed
      with dark semi-transparent quartz and flexible laminæ of mica.
      Similarly, plants and animals are non-homogeneous. Thus, leaves
      are composed of a skin, fibre, pulp, sap, and a green colouring
      matter. As an example of those non-homogeneous substances which
      are produced artificially, gunpowder may be cited, which is
      prepared by mixing together known proportions of sulphur, nitre,
      and charcoal. Many liquids, also, are not homogeneous, as may be
      observed by the aid of the microscope, when drops of blood are
      seen to consist of a colourless liquid in which red corpuscles,
      invisible to the naked eye owing to their small size, are floating
      about. It is these corpuscles which give blood its peculiar
      colour. Milk is also a transparent liquid, in which microscopical
      drops of fat are floating, which rise to the top when milk is
      left at rest, forming cream. It is possible to extract from every
      non-homogeneous substance those homogeneous substances of which
      it is made up. Thus orthoclase may he separated from porphyry by
      breaking it off. So also gold is extracted from auriferous sand by
      washing away the mixture of clay and sand. Chemistry deals only
      with the homogeneous substances met with in nature, or extracted
      from natural or artificial non-homogeneous substances. The various
      mixtures found in nature form the subjects of other natural
      sciences--as geognosy, botany, zoology, anatomy, &c."""
    q_test_2 = [  # note that the full text isn't included in each of the tuples here, I need to  change that
        (
            "Why is it important to distinguish between homogeneous and non-homogeneous substances?",
            "Homogeneous substances consist of parts that resemble each other in their properties, while non-homogeneous substances are made up of several homogeneous substances mixed together. Chemistry deals with the homogeneous substances met with in nature or extracted from natural or artificial non-homogeneous substances, so it is important to distinguish between them because it determines which parts of a given substance can be used for chemical analysis and study.",
        ),
        (
            "What is an example of an artistic mixture that would be non-homogeneous?",
            "An example of a non-homogeneous artistic mixture could be gunpowder, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal.",
        ),
        (
            "How might the concept of homogeneity apply to education or learning?",
            "In education or learning, students can think about their own knowledge as a homogeneous substance, made up of similar concepts that resemble each other in terms of understanding. They may need to separate out these ideas from non-homogeneous ones (e.g., misconceptions) in order to fully grasp the concept and build upon it.",
        ),
        (
            "If we were told to find homogeneous substances in nature, how would we go about doing this?",
            "To find homogeneous substances in nature, one could examine and investigate various objects met with in nature and in the arts. Some of these objects might be homogeneous, whilst others are composed of a mixture of several homogeneous substances. By breaking up a homogeneous substance, we would obtain parts which, although different in form, resemble each other in their properties. This suggests that we could identify homogeneous substances by looking for these characteristics. Additionally, some examples mentioned in the text include gold, iron, copper, glass, pure sugar, marble, and ordinary red granite. However, not all non-homogeneous substances are immediately apparent; it requires investigating and understanding how they are made up of different components (such as orthoclase being separated from porphyry). Therefore, a combination of observing physical properties, breaking down materials, and understanding their composition would allow us to identify homogeneous substances in nature.",
        ),
    ]
