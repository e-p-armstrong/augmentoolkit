from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

scenario_grammar = LlamaGrammar.from_string(r"""

root ::= reasoning-start

reasoning-start ::= [^\n\t]+ "."

# no-questions-after-here ::= "\nI will not ask any questions about the following information: " [^\n]+ "."

# TODO blacklist the word Bryon? To stop example leaks?
""")