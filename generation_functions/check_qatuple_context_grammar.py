from llama_cpp import LlamaGrammar

### A grammar that forces the model to generate correct character cards (with traits, names, everything)


# We don't actually need the final judgement step, the step-by-step combined with the fact that it makes a judgement at each step ensures accuracy
check_qatuple_context_grammar = LlamaGrammar.from_string(r"""
                        
# Root rule specifying the overall structure of the reasoning and thought process
root ::= question-validation "\n" answer-validation "\n" critical-evaluation "\n" revised-qatuple

# Question context validation
question-validation ::= "### Question Context Validation" "\n" special-term-check-question text-and-author-specificity scope-and-precision

special-term-check-question ::= "#### Special Term Context Check: Specifically check for use of the terms \"book\", \"text\", \"passage\", and \"excerpt\" without context about which specific thing is being discussed. " question-term-detail "\n"
text-and-author-specificity ::= "#### Text and Author Specificity: " question-text-author-detail "\n"
scope-and-precision ::= "#### Scope and Precision: " question-scope-detail "\n"

# Answer context validation
answer-validation ::= "### Answer Context Validation:" "\n" special-term-check-answer specificity-and-clarity answer-only-context-issues

special-term-check-answer ::= "#### Special Term Context Check: Specifically check for use of the terms \"book\", \"text\", \"passage\", and \"excerpt\" without context about which specific thing is being discussed. " answer-term-detail "\n"
specificity-and-clarity ::= "#### Specificity and Clarity: " answer-specificity-detail "\n"
answer-only-context-issues ::= "#### Answer-Only Context Issues: " answer-context-issue-detail "\n"

# Critical evaluation and final judgment
critical-evaluation ::= "### Critical Evaluation and Final Judgment:" "\n" evaluation final-judgment

evaluation ::= "#### Evaluation: " evaluation-detail "\n"
final-judgment ::= "#### Final judgment: " judgment-detail "\n"

# Optional revised Q&A tuple
revised-qatuple ::= "### Question Rewording (using text details as reference):" "\n" revised-question-answer

# Terminal symbols
question-term-detail ::= [^\n]+
question-text-author-detail ::= [^\n]+
question-scope-detail ::= [^\n]+
answer-term-detail ::= [^\n]+
answer-specificity-detail ::= [^\n]+
answer-context-issue-detail ::= [^\n]+
evaluation-detail ::= [^\n]+
judgment-detail ::= ("Pass."|"Fail."|"Reword.")
revised-question-answer ::= "Question: " [^\n]+ "\n" "Answer: " [^\n]+ "\n"

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