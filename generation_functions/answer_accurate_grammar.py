from llama_cpp import LlamaGrammar

# NOTE might struggle with very complex answers that have more than nine parts to them. This can be amended by adding more options to the "compare-step" rule, or making a more general pattern, if your use-case requires it.
# TODO                                               
# I COULD break this down further, by having the first step be a special "Analyze" step, the second step being a special "Understand" step, and the third+ being "Compare" steps" that each must end with "relevant" or "irrelevant" followed by a final judgement step... but currently it's working, and the model can't be that stupid right? Well I might do it later. I'll leave this comment here as a reminder.
# TODO

answer_accurate_grammar = LlamaGrammar.from_string(r"""                     
    
       
root ::= analyze-step understand-step compare-step final-step "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"


compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the First Part of the Answer with the Text: check if the text " [^\n]+ ("accurate"|"inaccurate"|"Accurate"|"Inaccurate") [^\n]* "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze" [^\n]+ "\n"

understand-step ::= "Step " [0-9]?[0-9] ". " "Understand" [^\n]+ "\n"

# understand-step ::= "Step " [0-9]?[0-9] ". " "Understand" [^\n]+ "\n"
""")
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)