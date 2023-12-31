from llama_cpp import LlamaGrammar


regenerate_answer_constrain_to_text_grammar = LlamaGrammar.from_string(r"""
                                            
root ::= reasoning

reasoning ::= [^\n]+ "\"\"\""
""")
