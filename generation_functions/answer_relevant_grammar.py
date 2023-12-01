from llama_cpp import LlamaGrammar

# TODO improve? I just copy-pasted it from something else

answer_relevant_grammar = LlamaGrammar.from_string(r"""                     
    
# TODO                                               
# I COULD break this down further, by having the first step be a special "Analyze" step, the second step being a special "Understand" step, and the third+ being "Compare" steps" that each must end with "relevant" or "irrelevant" followed by a final judgement step... but currently it's working, and the model can't be that stupid right? Well I might do it later. I'll leave this comment here as a reminder.
       
root ::= analyze-step understand-step summarize-relevant-step compare-step final-step "\n"

step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" ) [^\n]+ "\n"

compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the First Part of the Answer with the Text:" [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze" [^\n]+ "\n"

understand-step ::= "Step " [0-9]?[0-9] ". " "Understand" [^\n]+ "\n"

summarize-relevant-step ::= "Step " [0-9]?[0-9] ". " "Summarize Relevant Parts of the Text:" [^\n]+ "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " [^\n]+
""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)

