from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

character_card_grammar = LlamaGrammar.from_string(r"""
       
# Testing making traits come BEFORE dialogue examples, unlike AliChat, so that way it kind of "flows" into dialogue; and also the details are closer to the start and thus more easily remembered.                                     
#root ::= "Name: " name "\n" "Traits: " traits "\n\nDialogue Examples:" dialogue-examples

root ::= name "\n" "Traits: " traits "\n\nDialogue Examples:" dialogue-examples

# Spaces are forbidden in names because during Principles of Chemistry, the script wouldn't stop making the character have the last name Mendeleev!!!
name ::= [^\n ]+

traits ::=  trait trait trait trait trait trait trait trait trait trait trait trait trait? trait? trait? trait? trait? trait? trait? trait? # 14 comma-separated traits

trait ::= [A-Z][a-z ']+ ", " # todo, it wants hyphens, I can tell because I see it using double spaces for things like Mid twenties. But idk how to add hyphens.

dialogue-examples ::= history personality

history ::= "\nStranger: \"What's your backstory?\"\n" name ": \"" [^\n]+
personality ::= "\nStranger: \"What's your personality?\"\n" name ": \"" [^\n]+

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)