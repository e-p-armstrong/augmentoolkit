from llama_cpp import LlamaGrammar

ensure_answer_consistent_grammar = LlamaGrammar.from_string(
    r"""                     
       
root ::= understand-question-step compare-question-step understand-answer-step compare-step final-step "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"

understand-question-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided question:" [^\n]+ "\n"

compare-question-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's question: " [^\n]+ "\n"

understand-answer-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided answer:" [^\n]+ "\n"

# compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the " ("first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth") " Part of the Answer with the Text: check if the text " [^\n]+ "\n"

compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's answer:" [^\n]+ "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " ("Inconsistent" | "Consistent") "\n"
"""
)
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
