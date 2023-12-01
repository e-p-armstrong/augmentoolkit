from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)
 
# TODO make more reliable
regenerate_answer_constrain_to_text_grammar = LlamaGrammar.from_string(r"""
                                            
root ::= reasoning

reasoning ::= [^\n]+ "\"\"\""
""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)