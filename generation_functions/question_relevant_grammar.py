from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)


# We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
question_relevant_grammar = LlamaGrammar.from_string(r"""
       
   
   # TODO contrain to a set number of steps to limit the potential to go completely off the rails
                        
# Root rule specifying the overall structure of the reasoning and thought process
root ::= in-depth-analysis "\n" detailed-understanding "\n" targeted-comparison "\n" critical-evaluation

# In-depth analysis of the text
in-depth-analysis ::= "### In-Depth Analysis of the Text:" "\n" content-and-depth type-of-information

content-and-depth ::= "#### Content and Depth: " text-description "\n"
type-of-information ::= "#### Type of Information: " information-description "\n"

# Detailed understanding of the question
detailed-understanding ::= "### Detailed Understanding of the Question:" "\n" core-requirement depth-of-detail

core-requirement ::= "#### Core Requirement: " requirement-description "\n"
depth-of-detail ::= "#### Depth of Detail: " detail-description "\n"

# Targeted comparison of the question with the text
targeted-comparison ::= "### Targeted Comparison of the Question with the Text:" "\n" content-match depth-match

content-match ::= "#### Content Match: " match-description "\n"
depth-match ::= "#### Depth Match: " depth-match-description "\n"

# Critical evaluation and final judgment
critical-evaluation ::= "### Critical Evaluation and Final Judgment:" "\n" judgment

judgment ::= [^\n]+

# Terminal symbols
text-description ::= [^\n]+
information-description ::= [^\n]+
requirement-description ::= [^\n]+
detail-description ::= [^\n]+
match-description ::= [^\n]+
depth-match-description ::= [^\n]+
relevance ::= "Relevant." | "Irrelevant."


                                            
# root ::= reasoning from-the-text judgement

# reasoning ::= "First, I will check whether the question is answerable using the information in the paragraphs. The question asks " [^\n]+ "."
# from-the-text ::= "\nNow, regardless of what my initial thoughts are, I will try to find some passages from the text that directly answer this question, being mindful that \"How\" is different than \"What\". The text has the following information: " [^\n]+ "."
# judgement ::= "\nAll this considered, the answer is, compared to the provided text," (relevant|irrelevant) "."
# relevant ::= "relevant" | "Relevant"
# irrelevant ::= "irrelevant" | "Irrelevant"

# final_step ::= "Step " [0-9]?[0-9] ". " "Final Judgement: "

""")

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)


# from llama_cpp import LlamaGrammar

# ### A grammar that forces the model to generate correct character cards (with traits, names, everything)

# question_relevant_grammar = LlamaGrammar.from_string(r"""
                                            
# root ::= reasoning from-the-text judgement

# reasoning ::= "First, I will check whether the question is answerable using the information in the paragraphs. The question asks " [^\n]+ "."
# from-the-text ::= "\nThe paragraphs, for their part, only mention the following information: " [^\n]+ "."
# judgement ::= "\nAll this considered, the question is, compared to the provided text," (relevant|irrelevant) "."
# relevant ::= " relevant" | " Relevant"
# irrelevant ::= " irrelevant" | " Irrelevant"

# """)

# # question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# # root ::= answer
# # answer ::= "Test"
# # """)