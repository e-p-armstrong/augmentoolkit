from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)


# We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
question_relevant_grammar = LlamaGrammar.from_string(
    r"""
       
                        
root ::= step step step+ "\n"

reasoning-start ::= [^\n\t]+ "."

step ::= "Step " [0-9]?[0-9] ". " ( "Analyze" | "Understand" | "Compare" | "Skip" | "Final" ) [^\n]+ "\n"
                        
# final-step ::= "Final Judgement: [^\n]+ (relevant|irrelevant) " "."               
                                            
# root ::= reasoning from-the-text judgement

# reasoning ::= "First, I will check whether the question is answerable using the information in the paragraphs. The question asks " [^\n]+ "."
# from-the-text ::= "\nNow, regardless of what my initial thoughts are, I will try to find some passages from the text that directly answer this question, being mindful that \"How\" is different than \"What\". The text has the following information: " [^\n]+ "."
# judgement ::= "\nAll this considered, the answer is, compared to the provided text," (relevant|irrelevant) "."
relevant ::= " relevant" | " Relevant"
irrelevant ::= " irrelevant" | " Irrelevant"

"""
)

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
