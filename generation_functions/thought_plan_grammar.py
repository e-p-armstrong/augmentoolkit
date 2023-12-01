from llama_cpp import LlamaGrammar


# A GGBNF grammar that forces the model to output text in a particular format
thought_plan_grammar = LlamaGrammar.from_string(r"""
                                            
# Root rule defining the overall structure
root ::= step+ "\n"

# Step rule with some text (any characters except newline) followed by a period
step ::= "Step " [0-9]?[0-9] ". " ("Realize" | "Recognize" | "Conclude" | "Recall" | "Remember" | "Formulate" | "Decompose" | "Break down" | "Break" | "Therefore, the answer is" | "The answer is" | "Realise" | "Calculate" | "Understand" | "Note" | "The plan will") [^\n]+ "\n"

# Potential way forward: change these reasoning steps to use 
# step ::= "Step " [0-9]?[0-9] ". " ("Realize" | "Recall" | "Remember" | "Formulate" | "Decompose" | "Break down" | "Break" | "Therefore, the answer is" | "The answer is" | "Realise" | "Calculate" | "Understand" | "Note" | "The plan will") [^\n]+ "\n"

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)