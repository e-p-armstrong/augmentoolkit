from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)


# We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
identify_duplicates_grammar = LlamaGrammar.from_string(r"""
       
# Root rule defining the overall structure of the response
root ::= normalization-block core-components-block comparative-analysis-block criteria-block conclusion-block

# Normalization of Questions
normalization-block ::= "## Normalization of Questions:\n" normalized-question+
normalized-question ::= "- \"" content "\"\n  - Normalized: " content "\n"

# Identification of Core Components
core-components-block ::= "## Identification of Core Components:\n### Subject Matter:\n" subject-matter+ "### Information Sought:\n" information-sought+
subject-matter ::= "- Question " digit+ ": " content "\n"
information-sought ::= "- Question " digit+ ": " content "\n"

# Comparative Analysis Across Questions
comparative-analysis-block ::= "## Comparative Analysis Across Questions:\n### Direct Comparison:\n" bullet-item+ "### Overlap in Core Components:\n" bullet-item+ "\n"

# Criteria for Duplication
criteria-block ::= "## Criteria for Duplication:\n### Exact Information Match:\n" content "### Negation of Minor Differences:\n" content "\n"

# Conclusion and Labeling
conclusion-block ::= "## Conclusion and Labeling:\n" content "\n\n## Unique Questions: " unique-questions "\n"
unique-questions ::= "[" digit+ (", " digit+)* "]"

# Basic components
content ::= char+ # A sequence of characters representing content

digit ::= [0-9] # Digits
char ::= [^\n] # Any character except newline

bullet-item ::= "- " content "\n"
""")