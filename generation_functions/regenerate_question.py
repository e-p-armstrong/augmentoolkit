import re
from generation_functions.question_grammar import question_grammar
from .strip_steps import strip_steps

# Question regeneration (triggered after a relevance-check fails on the question).
def regenerate_question(qatuple, dissenting_reasoning,plan,logic_llm):
    retries = 0
    while retries < 5:
        decision_prompt = f"""You are an expert educational AI. You are provided with a flawed question that requires significant knowledge outside these paragraphs to answer. You will make a different question that is solveable if one knows the information in the paragraphs (but since students will not have the paragraphs at hand when given the question, do not refer to the text). Given these paragraphs, the flawed question based on the paragraphs, and the explanation of why the answer is flawed, and a plan for the new question, you will write out a new question that only requires information from the paragraphs to solve. 

Do NOT just rephrase the old question.

Text: \"\"\"{qatuple[2]}\"\"\"

FLAWED Question (based on text) (you will be writing something ENTIRELY DIFFERENT that FIXES THIS ONE'S PROBLEMS): \"\"\"{qatuple[0]}\"\"\"

Reasoning as to why the question is irrelevant: \"\"\"{strip_steps(dissenting_reasoning)}\"\"\"

New question plan: \"\"\"{plan}\"\"\"

Note: Do not explicitly mention the paragraphs in your question itself — just ask about the concepts, and only those concepts which appear in the text. The students will not have access to the paragraphs when asked the question, so you cannot refer to the paragraphs.

Keep in mind: An example about how NOT to ask questions: if the text states fact X, but does not explain how X was established, do not ask a question "How do we know X". But instead you might consider asking how X relates to other facts in the paragraph, or how these facts together lead to a progression of ideas, "Explain how X, Y, and Z are related" for instance.

Final note: you are allowed and encouraged to dramatically change/revamp/rewrite the question's content. But ONLY ASK ONE QUESTION.

# New question:
"""
        completion = logic_llm(decision_prompt, max_tokens=4000, stop=["</s>","2.)"], echo=True, grammar=question_grammar,temperature=0.2)["choices"][0]["text"]

        # print("DEBUG\n\n")
        # print(completion)
        completion_pattern = re.compile(r"New question:\n(.+)", re.DOTALL)
        correction = completion_pattern.search(completion).group(1)
        return correction.strip()