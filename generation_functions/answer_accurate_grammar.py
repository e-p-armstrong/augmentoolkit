from llama_cpp import LlamaGrammar

# NOTE might struggle with very complex answers that have more than nine parts to them. This can be amended by adding more options to the "compare-step" rule, or making a more general pattern, if your use-case requires it.

answer_accurate_grammar = LlamaGrammar.from_string(r"""                     
    
       
# Root rule specifying the overall structure of the analysis
root ::= text-analysis answer-breakdown accuracy-check final-judgment

# Text analysis
text-analysis ::= "### Text Analysis:" "\n" identify-key-info categorize-info-type

identify-key-info ::= "#### Identify Key Information: " text-info-detail "\n"
categorize-info-type ::= "#### Categorize Information Type: " info-type-detail "\n\n"

# Answer breakdown
answer-breakdown ::= "### Answer Breakdown:" "\n" dissect-answer identify-answer-type

dissect-answer ::= "#### Dissect the Answer: " answer-detail "\n"
identify-answer-type ::= "#### Identify Answer Type: " answer-type-detail "\n\n"

# Accuracy check
accuracy-check ::= "### Accuracy Check:" "\n" direct-comparison inference-and-contextual-alignment

direct-comparison ::= "#### Direct Comparison for Factual Accuracy:\n" comparison-points
comparison-points ::= bullet-point+
bullet-point ::= "  - " comparison-point-detail "\n"
inference-and-contextual-alignment ::= "#### Inference and Contextual Alignment: " contextual-alignment-detail "\n\n"

# Final judgment
final-judgment ::= "### Final Judgment:" "\n" comprehensive-assessment overall-accuracy-determination

comprehensive-assessment ::= "#### Comprehensive Assessment: " assessment-detail "\n"
overall-accuracy-determination ::= "#### Overall Accuracy Determination: " accuracy-detail "\n"

# Terminal symbols
text-info-detail ::= [^\n]+
info-type-detail ::= [^\n]+
answer-detail ::= [^\n]+
answer-type-detail ::= [^\n]+
comparison-point-detail ::= [^\n]+
contextual-alignment-detail ::= [^\n]+
assessment-detail ::= [^\n]+
accuracy-detail ::= [^\n]+

# understand-step ::= "Step " [0-9]?[0-9] ". " "Understand" [^\n]+ "\n"
""")
# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)