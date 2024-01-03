from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

single_turn_conversation_grammar = LlamaGrammar.from_string(
    r"""

# The root rule defines the structure of the dialogue
root ::= statement "\n\n" response "\n"

# Statement by Character Name 1
statement ::= [^\n]+ ":" [^\n]+

# Response by Character Name 2
response ::= [^\n]+ ":" [^\n]+

# Definition of a character name
character-name ::= word ("-" word)*
word ::= [A-Za-z]+
# Limiting to a maximum of six words
character-name ::= word | word word | word word word | word word word word | word word word word word | word word word word word word

# Definition of a dialogue line
dialogue-line ::= [^\n]+


"""
)

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
