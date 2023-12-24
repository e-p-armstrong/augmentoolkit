from llama_cpp import LlamaGrammar

judge_paragraph_grammar = LlamaGrammar.from_string(r"""                     
       
root ::= identify-content-step evaluate-relevance-step assess-contexts-and-formats-step assess-possibility-step determine-suitability-step check-contextual-completeness-step final-step "\n"

identify-content-step ::= "Step " [0-9]?[0-9] ". " "Identify Paragraph Content: " [^\n]+ "\n"

evaluate-relevance-step ::= "Step " [0-9]?[0-9] ". " "Evaluate Educational Relevance: " [^\n]+ "\n"

assess-contexts-and-formats-step ::= "Step " [0-9]?[0-9] ". " "Assess Specific Contexts and Formats:" "\n" context-format-bullets

assess-possibility-step ::= "Step " [0-9]?[0-9] ". " "Assess the Possibility of Formulating Questions: " [^\n]+ "\n"

determine-suitability-step ::= "Step " [0-9]?[0-9] ". " "Determine Suitability for Educational Purposes: " [^\n]+ "\n"

check-contextual-completeness-step ::= "Step " [0-9]?[0-9] ". " "Check for Contextual Completeness: " [^\n]+ "\n"

final-step ::= "Step " [0-9]?[0-9] ". " "Final Judgment: " ("Unsuitable" | "Suitable" | "suitable" | "unsuitable") "\n"

context-format-bullets ::= bullet-item+
bullet-item ::= "  - " bullet-item-detail "\n"
bullet-item-detail ::= [^\n]+
""")