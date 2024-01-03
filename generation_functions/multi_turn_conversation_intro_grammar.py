from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

multi_turn_conversation_grammar = LlamaGrammar.from_string(
    r"""

# The root rule defines the structure of the dialogue
root ::= [^\n]+ ":" [^\n]+ #"\n" statement+# statement anything+ # Idea: get it started off right, then  let it end how it wants

"""
)

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
