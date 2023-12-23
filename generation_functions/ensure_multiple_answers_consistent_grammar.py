from llama_cpp import LlamaGrammar

# TODO
ensure_multiple_answers_consistent_grammar = LlamaGrammar.from_string(r"""                     
    
# Root rule to define the overall structure
root ::= sequential-matching-section accuracy-check-section conclusion-section

# Section for sequential matching
sequential-matching-section ::= "## Sequential Matching of Questions in the Conversation:\n### Sequence and Phrasing of Questions:\n" matching-statement+

# Section for accuracy check
accuracy-check-section ::= "## Accuracy Check for Answers in the Conversation:\n### Matching Answers with Provided Content:\n" accuracy-statement+

# Conclusion section
conclusion-section ::= "## Conclusion:\n" conclusion-statement+

# Definitions of different components
number ::= [1-9]
matching-statement ::= number ". " [^\n]+ "\n"
accuracy-statement ::= number ". " [^\n]+ "\n"
conclusion-statement ::= "  - " [^\n]+ "\n"
final-judgement ::= "  - Final Judgment:" [^\n]+
""")


# the-tone ::= "  - The tone" [^\n]+ "\n"
# the-conversation ::= "  - The conversation" [^\n]+ "\n"
# the-dialogue ::= "  - " [^\n]+ "\n"

# reflects ::= "  - " [^\n]+ "\n"
# logical-flow ::= "  - " [^\n]+ "\n"
# consistency-check ::= "  - " [^\n]+ "\n"
# final-judgment ::= "  - " [^\n]+ "\n"