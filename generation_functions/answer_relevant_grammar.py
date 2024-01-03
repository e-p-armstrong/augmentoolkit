from llama_cpp import LlamaGrammar

answer_relevant_grammar = LlamaGrammar.from_string(
    r"""                     
       
# Root rule specifying the overall structure of the analysis
root ::= deep-analysis "\n" comprehensive-understanding "\n" targeted-comparison "\n" identification-of-extraneous-info "\n" final-judgment

# Deep analysis of the text
deep-analysis ::= "### Deep Analysis of the Text:" "\n" content-scope-and-detail type-of-information

content-scope-and-detail ::= "#### Content Scope and Detail: " text-detail "\n"
type-of-information ::= "#### Type of Information: " info-type "\n"

# Comprehensive understanding of the answer
comprehensive-understanding ::= "### Comprehensive Understanding of the Answer:" "\n" key-components-identification depth-of-explanation

key-components-identification ::= "#### Key Components Identification: " components-detail "\n"
depth-of-explanation ::= "#### Depth of Explanation: " explanation-detail "\n"

# Targeted comparison of answer with text
targeted-comparison ::= "### Targeted Comparison of Answer with Text:" "\n" content-alignment depth-alignment

content-alignment ::= "#### Content Alignment: " alignment-detail "\n"
depth-alignment ::= "#### Depth Alignment: " depth-detail "\n"

# Identification of extraneous information
identification-of-extraneous-info ::= "### Identification of Extraneous Information:" "\n" spotting-additional-details assessing-impact

spotting-additional-details ::= "#### Spotting Additional Details: " additional-details "\n"
assessing-impact ::= "#### Assessing Impact of Additional Information: " impact-assessment "\n"

# Final judgment on answer relevance
final-judgment ::= "### Final Judgment on Answer Relevance:" "\n" relevance-assessment explanation-of-judgment

relevance-assessment ::= "#### Relevance Assessment: " relevance-detail "\n"
explanation-of-judgment ::= "#### Explanation of Judgment: " judgment-detail "\n"

# Terminal symbols
text-detail ::= [^\n]+
info-type ::= [^\n]+
components-detail ::= [^\n]+
explanation-detail ::= [^\n]+
alignment-detail ::= [^\n]+
depth-detail ::= [^\n]+
additional-details ::= [^\n]+
impact-assessment ::= [^\n]+
relevance-detail ::= [^\n]+
judgment-detail ::= [^\n]+

"""
)

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)
