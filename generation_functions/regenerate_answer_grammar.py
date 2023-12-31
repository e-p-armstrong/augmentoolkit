from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)
 
regenerate_answer_grammar = LlamaGrammar.from_string(r"""
                                            
root ::= reasoning

reasoning ::= [^\n]+ "."
""")
