from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)


# We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
check_qatuple_context_reword_grammar = LlamaGrammar.from_string(r"""
                        
# Root rule specifying the overall structure of the reasoning and thought process
root ::= "Question: " [^\n]+ "\n" "Answer: " [^\n]+ "\n"

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)


# from llama_cpp import LlamaGrammar

# ### A grammar that forces the model to generate correct character cards (with traits, names, everything)

# question_relevant_grammar = LlamaGrammar.from_string(r"""
                                            
# root ::= reasoning from-the-text judgement

# reasoning ::= "First, I will check whether the question is answerable using the information in the paragraphs. The question asks " [^\n]+ "."
# from-the-text ::= "\nThe paragraphs, for their part, only mention the following information: " [^\n]+ "."
# judgement ::= "\nAll this considered, the question is, compared to the provided text," (relevant|irrelevant) "."
# relevant ::= " relevant" | " Relevant"
# irrelevant ::= " irrelevant" | " Irrelevant"

# """)

# # question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# # root ::= answer
# # answer ::= "Test"
# # """)