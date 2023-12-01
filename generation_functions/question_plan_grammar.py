from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

question_plan_grammar = LlamaGrammar.from_string(r"""

# root ::= reasoning-start
# At least 3 steps
root ::= identify-step generate-step brainstorm-step relationships-step if-then-step

# no-questions-after-here ::= "\nI will not ask any questions about the following information: " [^\n]+ "."

identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Key Topics:" [^\n]+ "\n"

# generate-step ::= "Step " [0-9]?[0-9] ". " "Determine Information-Rich Areas:" [^\n]+ "\n"

brainstorm-step ::= "Step " [0-9]?[0-9] ". " "Brainstorm and Develop Questions Testing Recall:" [^\n]+ "\n"

relationships-step ::= "Step " [0-9]?[0-9] ". " "Devise Questions" [^\n]+ "\n"

if-then-step ::= "Step " [0-9]?[0-9] ". " "Create Questions Investigating" [^\n]+ "\n"

""")


# Realize
# Devise
# Recall
# Note that
# The text states that
# The text says
# The text
# Formulate
# Brainstorm
# See if
# See whether
# Determine
# No such questions
# There are
# I have brainstormed
# End of reasoning