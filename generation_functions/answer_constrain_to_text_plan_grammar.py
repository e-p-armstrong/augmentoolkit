from llama_cpp import LlamaGrammar

answer_constrain_to_text_plan_grammar = LlamaGrammar.from_string(
    r"""
       
root ::= analyze-step understand-step identify-step plan-revised-step "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Final" | "Plan" | "Identify" ) [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Text:" [^\n]+ "\n"

understand-step ::= "Step " [0-9]?[0-9] ". " "Understand the Question:" [^\n]+ "\n"

identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Flawed Part of the Answer:" [^\n]+ "\n"

plan-revised-step ::= "Step " [0-9]?[0-9] ". " "Plan Revised Answer:" [^\n]+ "\n"
"""
)
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
