from llama_cpp import LlamaGrammar


make_regenerate_question_plan_grammar = LlamaGrammar.from_string(
    r"""
root ::= analyze-step identify-step generate-step refine-step ensure-step end-of-reasoning

step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Identify" | "Generate" | "Refine" | "Ensure" ) [^\n]+ "\n"

analyze-step ::= "Step " [0-9]?[0-9] ". " "Analyze the Reason for the Flaw:" [^\n]+ "\n"

identify-step ::= "Step " [0-9]?[0-9] ". " "Identify Key Concepts in Paragraphs:" [^\n]+ "\n"

generate-step ::= "Step " [0-9]?[0-9] ". " "Generate a New Question Idea:" [^\n]+ "\n"

refine-step ::= "Step " [0-9]?[0-9] ". " "Refine the Question:" [^\n]+ "\n"

ensure-step ::= "Step " [0-9]?[0-9] ". " "Ensure Alignment with Text:" [^\n]+ "\n"

end-of-reasoning ::= "Step " [0-9]?[0-9] ". " "End" [^\n]+ "\n"
"""
)
