from llama_cpp import LlamaGrammar



make_regenerate_answer_plan_grammar = LlamaGrammar.from_string(r"""
root ::= analyze-step understand-step identify-step plan-step

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Text:" [^\n]+ "\n"

understand-step ::= "Step " [0-9]?[0-9] ". " "Understand the Question:" [^\n]+ "\n"

identify-step ::= "Step " [0-9]?[0-9] ". " "Identify the Incorrect Part of the Answer:" [^\n]+ "\n"

plan-step ::= "Step " [0-9]?[0-9] ". " "Plan a Corrected Answer:" [^\n]+ "\n"
""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)

