import re
from .question_grammar import question_grammar
from .strip_steps import strip_steps


# Question regeneration (triggered after a relevance-check fails on the question).
def regenerate_question_DEPRECATED(qatuple, dissenting_reasoning, logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""You are an expert educational AI. You are focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels). Someone has written a question that was supposed to be based on the provided paragraphs of text, but actually requires significant knowledge outside these paragraphs to answer. Given these paragraphs, the flawed question based on the paragraphs, and the explanation of why the answer is flawed, you will write out a new question that only requires information from the paragraphs to solve. However, you will not explicitly refer to the paragraphs in the question; the student will not have access to the text when answering it.
        
So, in short, your task is to rewrite a new question that is actually answerable if one knows the concepts from the paragraphs.

Text: \"\"\"{qatuple[2]}\"\"\"

Question (based on text): \"\"\"{qatuple[0]}\"\"\"

Reasoning as to why the question is irrelevant: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

Do not explicitly mention the paragraphs in your question itself — just ask about the concepts, and only those concepts which appear in the text.

An example about how not to ask questions: if the text states fact X, but does not explain how X was established, do not ask a question "How do we know X". But instead you might consider asking how X relates to other facts in the paragraph, or how these facts together lead to a progression of ideas, "Explain how X, Y, and Z are related" for instance. Use Bloom's taxonomy, and focus on the cognitive levels of understanding, application, analysis, and synthesis of ideas.

As a final requirement, you must write your question and its answer like this:
num) question contents, drawing only on info in the text.
Answer: question answer, using only info in the paragraphs.


# New question (1 paragraph at most; do not explicitly refer to the provided text, just test the concepts; FIX THE PROBLEMS IDENTIFIED IN THE CRITIQUE BY REWRITING THE QUESTION):
"""
        completion = logic_llm(
            decision_prompt,
            max_tokens=500,
            stop=["</s>", "2)", "2.)"],
            echo=True,
            grammar=question_grammar,
        )["choices"][0]["text"]

        # print("DEBUG\n\n")
        # print(completion)
        completion_pattern = re.compile(
            r"New question \(1 paragraph at most; do not explicitly refer to the provided text, just test the concepts\):\n(.+)",
            re.DOTALL,
        )
        correction = completion_pattern.search(completion).group(1)

        pattern = re.compile(
            r"(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)",
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        matches = pattern.findall(correction)
        if len(matches) == 0:
            retries += 1
            continue
        return correction.strip()
