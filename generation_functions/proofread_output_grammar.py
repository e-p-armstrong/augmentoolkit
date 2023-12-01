from llama_cpp import LlamaGrammar

# TODO
proofread_output_grammar = LlamaGrammar.from_string(r"""                     
    
# TODO                                               
# I COULD break this down further, by having the first step be a special "Analyze" step, the second step being a special "Understand" step, and the third+ being "Compare" steps" that each must end with "relevant" or "irrelevant" followed by a final judgement step... but currently it's working, and the model can't be that stupid right? Well I might do it later. I'll leave this comment here as a reminder.
       
root ::= analyze-step step+ "\n\nBegin Edit: " [^\n]+

step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Notice" | "Note" | "There is" | "Error" | "I found" | "End" | "There are" ) [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze" [^\n]+ "\n"
""")
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)