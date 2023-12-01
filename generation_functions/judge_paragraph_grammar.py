from llama_cpp import LlamaGrammar

# TODO
judge_paragraph_grammar = LlamaGrammar.from_string(r"""                     
    
# TODO                                               
# I COULD break this down further, by having the first step be a special "Analyze" step, the second step being a special "Understand" step, and the third+ being "Compare" steps" that each must end with "relevant" or "irrelevant" followed by a final judgement step... but currently it's working, and the model can't be that stupid right? Well I might do it later. I'll leave this comment here as a reminder.
       
root ::= identify-content-step evaluate-relevance-step assess-possibility-step determine-suitability-step final-step "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"

# NOTE might struggle with very complex answers that have more than nine parts to them. This can be amended by adding more options to the "compare-step" rule, or making a more general pattern, if your use-case requires it.

identify-content-step ::= "Step " [0-9]?[0-9] ". " "Identify paragraph content: " [^\n]+ "\n"

assess-possibility-step ::= "Step " [0-9]?[0-9] ". " "Assess the possibility of formulating questions: " [^\n]+ "\n"

evaluate-relevance-step ::= "Step " [0-9]?[0-9] ". " "Evaluate educational relevance: " [^\n]+ "\n"

determine-suitability-step ::= "Step " [0-9]?[0-9] ". " "Determine suitability for educational purposes: " [^\n]+ "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: " ("Unsuitable" | "Suitable" | "suitable" | "unsuitable") "\n"
""")
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)