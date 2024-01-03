from llama_cpp import LlamaGrammar

proofread_output_grammar = LlamaGrammar.from_string(
    r"""                     
       
root ::= analyze-step step+ "\n\nBegin Edit: " [^\n]+

step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Notice" | "Note" | "There is" | "Error" | "I found" | "End" | "There are" ) [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze" [^\n]+ "\n"
"""
)
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
