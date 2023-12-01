from llama_cpp import LlamaGrammar

# TODO
ensure_answer_consistent_grammar = LlamaGrammar.from_string(r"""                     
    
# TODO                                               
# I COULD break this down further, by having the first step be a special "Analyze" step, the second step being a special "Understand" step, and the third+ being "Compare" steps" that each must end with "relevant" or "irrelevant" followed by a final judgement step... but currently it's working, and the model can't be that stupid right? Well I might do it later. I'll leave this comment here as a reminder.
       
root ::= understand-question-step compare-question-step understand-answer-step compare-step final-step "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"

# NOTE might struggle with very complex answers that have more than nine parts to them. This can be amended by adding more options to the "compare-step" rule, or making a more general pattern, if your use-case requires it.

understand-question-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided question:" [^\n]+ "\n"

compare-question-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's question: " [^\n]+ "\n"

understand-answer-step ::= "Step " [0-9]?[0-9] ". " "Understand the provided answer:" [^\n]+ "\n"

# compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the " ("first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth") " Part of the Answer with the Text: check if the text " [^\n]+ "\n"

compare-step ::= "Step " [0-9]?[0-9] ". " "Compare the conversation's answer:" [^\n]+ "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " ("Inconsistent" | "Consistent") "\n"
""")
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)