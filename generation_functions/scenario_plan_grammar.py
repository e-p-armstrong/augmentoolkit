from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)

# TODO ban the word Stranger here, or use a randomized name in the character card. OR get an LLM to generate a name for the character card.
scenario_plan_grammar = LlamaGrammar.from_string(
    r"""
                                                 
root ::= consider-question-step consider-character-step constrain-step setting-step create-step second-message-step "\n"

consider-question-step ::= "Step " [0-9]?[0-9] ". " "Focus on the question and answer: The question" [^\n]+ "\n"

consider-character-step ::= "Step " [0-9]?[0-9] ". " "Character Consideration: The primary character is" [^\n]+ "\n"

constrain-step ::= "Step " [0-9]?[0-9] ". " "Constrain the Scenario: The interaction is limited to a single question from the secondary character and a single, focused reply from" [^\n]+ "\n"

setting-step ::= "Step " [0-9]?[0-9] ". " "Setting: Given the subject of the question, and the character card, the setting will be" [^\n]+ "\n"

create-step ::= "Step " [0-9]?[0-9] ". " "Interaction: Given these constraints, the first message (delivered by the secondary character) might" [^\n]+ "\n"

second-message-step ::= "Step " [0-9]?[0-9] ". In the second message," [^\n]+ "\n"

"""
)
