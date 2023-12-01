from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

character_card_grammar = LlamaGrammar.from_string(r"""
       
# Testing making traits come BEFORE dialogue examples, unlike AliChat, so that way it kind of "flows" into dialogue; and also the details are closer to the start and thus more easily remembered.                                     
root ::= "Name: " name "\n" "Traits: " traits "\nDialogue Examples:" dialogue-examples

name ::= [^\n]+

traits ::=  trait trait trait trait trait trait trait trait trait trait trait trait trait? trait? trait? trait? trait? trait? trait? trait? [A-Z][a-z]+ # 14 comma-separated traits

trait ::= [A-Z][a-z ]+ ", "

dialogue-examples ::= history personality

history ::= "\nStranger: \"What's your backstory?\"\n" name ": \"" [^\n]+ ".\"" 
personality ::= "\nStranger: \"What's your personality?\"\n" name ": \"" [^\n]+ ".\""

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)