# from llama_cpp import LlamaGrammar

from lark import Lark

judge_paragraph_ebnf = r"""

start: identifycontentstep evaluaterelevancestep assesscontentsandformatsstep assessprobabilitystep determinesuitabilitystep checkcontextualcompletenessstep finalstep "\n"

identifycontentstep: "step " stepnumber ". " "Identify Paragraph Content: " /[^\n]+/ "\n"

evaluaterelevancestep: "step " stepnumber ". " "Evaluate Educational Relevance: " /[^\n]+/ "\n"

assesscontentsandformatsstep: "step " stepnumber ". " "Assess Specific Contexts and Formats:" "\n" contextformatbullets

assessprobabilitystep: "step " stepnumber ". " "Assess the Possibility of Formulating Questions: " /[^\n]+/ "\n"

determinesuitabilitystep: "step " stepnumber ". " "Determine Suitability for Educational Purposes: " /[^\n]+/ "\n"

checkcontextualcompletenessstep: "step " stepnumber ". " "Check for Contextual Completeness: " /[^\n]+/ "\n"

finalstep: "step " stepnumber ". " "Final Judgment: " /(Unsuitable) | (Suitable) | (suitable) | (unsuitable)/ "\n"

contextformatbullets: bulletitem+
bulletitem: "   " bulletitemdetail "\n"
bulletitemdetail: /[^\n]+/
stepnumber: /[0-9]?[0-9]/
"""

# judge_paragraph_grammar = LlamaGrammar.from_string(
#     r"""                     
       
# root ::= identifycontentstep evaluaterelevancestep assesscontentsandformatsstep assessprobabilitystep determinesuitabilitystep checkcontextualcompletenessstep finalstep "\n"

# identifycontentstep ::= "step " [09]?[09] ". " "Identify Paragraph Content: " [^\n]+ "\n"

# evaluaterelevancestep ::= "step " [09]?[09] ". " "Evaluate Educational Relevance: " [^\n]+ "\n"

# assesscontentsandformatsstep ::= "step " [09]?[09] ". " "Assess Specific Contexts and Formats:" "\n" contextformatbullets

# assessprobabilitystep ::= "step " [09]?[09] ". " "Assess the Possibility of Formulating Questions: " [^\n]+ "\n"

# determinesuitabilitystep ::= "step " [09]?[09] ". " "Determine Suitability for Educational Purposes: " [^\n]+ "\n"

# checkcontextualcompletenessstep ::= "step " [09]?[09] ". " "Check for Contextual Completeness: " [^\n]+ "\n"

# finalstep ::= "step " [09]?[09] ". " "Final Judgment: " ("Unsuitable" | "Suitable" | "suitable" | "unsuitable") "\n"

# contextformatbullets ::= bulletitem+
# bulletitem ::= "   " bulletitemdetail "\n"
# bulletitemdetail ::= [^\n]+
# """
# )
