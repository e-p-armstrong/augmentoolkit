from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

multi_turn_conversation_grammar = LlamaGrammar.from_string(r"""

# The root rule defines the structure of the dialogue
root ::= [^\t]+ #"\n" statement+# statement anything+ # Idea: get it started off right, then  let it end how it wants

# Statement by Character Name 1
statement ::= [^\n]+ ":" [^\n]+ "\n"

anything ::= [^\t]+ # I don't think GBNF has a wildcard for ANY character, so I just ban tabs because this text shouldn't be indented, and say "go for it"

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)