from llama_cpp import LlamaGrammar

question_grammar = LlamaGrammar.from_string(r"""
root ::= (question-one answer "\n")

# Define the question structure with a number followed by content and ending punctuation
# question ::= number ".) " [^\n]+ [?.!] "\n" # maybe blacklist ?!. along with newlines

# Define the answer structure
answer ::= "Answer: " [^\n]+ "\n"

# Define a number (in this case, limiting to any three-digit number for simplicity)
number ::= [1-9] [0-9]? [0-9]?

# Define content as a sequence of characters excluding the word "paragraph" and using not_paragraph to build up the content
# content ::= (not-paragraph "paragraph")* #not_paragraph


question-one ::= "1.) " [^\n]+ [?.!] "\n" # maybe blacklist ?!. along with newlines
# ws ::= [ \t\n]*
# Define not_paragraph as any sequence of characters that does not contain "paragraph" 
# and is terminated by a space, punctuation or newline to avoid partial matching of the word.
# not-paragraph ::= ([^\n\ \.\?!]*(["\.\?! ]+[^p\n\ \.\?!]*)* 
#     ( "p" ([^\an\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "pa" ([^\br\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "par" ([^\ag\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "para" ([^\bg\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "parag" ([^\rr\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "paragr" ([^\aa\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "paragra" ([^\pp\n\ \.\?!]+ ["\.\?! ]+)* 
#     | "paragraph" [^\np\n\ \.\?!]+))* 
#     [^\n\ \.\?!paragraph]+ 
    
    

# Optional whitespace: space, tab, or newlines zero or more times
""")









# questions_grammar = LlamaGrammar.from_string(r"""
                                            
# root ::= (question answer)

# # Define the question structure with a number followed by content and ending punctuation
# question ::= number ".) " content [?.!] "\n"

# # # Define the answer structure
# answer ::= "Answer: " content "\n"

# # # Define a number (in this case, limiting to any three-digit number for simplicity)
# number ::= [1-9] [0-9]? [0-9]?

# # # Define content as a sequence of characters that can include punctuation
# # # Here we use a negated set for anything that's not a newline to allow for punctuation within the content
# content ::= [^\n]+

# # # Optional whitespace: space, tab, or newlines zero or more times
# ws ::= [ \t\n]*

# """)

# question_grammar = LlamaGrammar.from_string(r"""# GBNF Grammar for Q&A Format with Flexible Punctuation

# root ::= answer
# answer ::= "Test"
# """)