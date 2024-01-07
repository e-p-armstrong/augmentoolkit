from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

character_card_plan_grammar = LlamaGrammar.from_string(
    r"""
       
# Testing making traits come BEFORE dialogue examples, unlike AliChat, so that way it kind of "flows" into dialogue; and also the details are closer to the start and thus more easily remembered.       





root ::= [^\n]+


# root ::= consider-step theme-step consistency-step "\n"

# consider-step ::= "Step 1. " "Consider the provided" [^\n]+ "\n"

# theme-step ::= "Step 2. " "Given the question, answer, and overall text, a theme for " [^\n]+ "\n"

# consistency-step ::= "Step 3. " "For this (fictional) character's theme to be what it is, and for them to understand what they do, they would need to live " [^\n]+ # leaving "they must live" relatively open-ended (not "in" or "during") so that this can adapt to even fictional worlds.

# freeflow-reasoning ::= "Step 4. " "Therefore, a promising character for this question is: " [^\n]+ "\n"



                              
# root ::= consider-step theme-step step+ "\n"

# step ::= "Step " [0-9]?[0-9] ". " ( "A Physical Trait" | "One potential detail" | "Another potential detail" | "A potential detail" | "Therefore" | "Note" ) [^\n]+ "\n"

# consider-step ::= "Step " [0-9]?[0-9] ". " "Consider" [^\n]+ "\n"

# theme-step ::= "Step " [0-9]?[0-9] ". " "A theme " [^\n]+ "\n"

"""
)

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
