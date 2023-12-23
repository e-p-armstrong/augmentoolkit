import re
from .ensure_multiple_answers_consistent_grammar import ensure_multiple_answers_consistent_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .format_qatuples import format_qatuples
from .extract_name import extract_name
# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed


# NOTE this prompt right now VERY MUCH struggles to follow its actual format; but it still mostly works
def ensure_multiple_answers_consistent(qatuples,conv,logic_llm,permissive_mode=True):
    """
    permissive_mode: turn off if you want a single usage of the word "inconsistent" anywhere in the message to flag the whole thing as inconsistent. Prevents errors where an inconsistency happens way early in the answer, but the model forgets about it during its final judgement; but enables the error where the model mentions that something is "not entirely inconsistent" or similar, which is surprisingly common.
    """
    retries = 0
    character_name = extract_name(conv[1])
    # It's expensive to regen a conversation; so we check very thoroughly, and use a two-shot example. "Permissive mode" recommended
    
    # NOTE: I don't know what kind of errors this part of the pipeline will run into most often, so I don't really know what examples to feed it to guard it with. Come back to it once I have tested it more.
    
    # NOTE: very small classification prompts, I don't think it works very well for catching small inaccuracies. We need the large, step-by-step analysis.
    
    
    # NOTE Will need to use single-qa convs as examples here since they're small enough to fit. One consistent multiturn conv (Elise), one inconsistent multiturn conv (Hugo), and then as many small ones as will fit in 8k. Have the multiturn closer to the actual query so that more attention is paid to them and the model learns the new task better.
    
    # NOTE Introduction to Practicing Chemical Science does not exist; this is more stuff from principles of chemistry named otherwise to avoid biasing the outputs more than can be helped
    # Consider removing the "conversational fluff" bit of the prompt. It's not really necessary? maybe?
    while (retries <= 4):
        decision_prompt = f"""You are an expert educational AI. Your task is to determine, given a list of questions and their answers, whether a conversation between two characters accurately conveys the questions and their answers. You will also check whether the conversation makes logical sense (specifically, that it does not start with a character spilling their entire backstory and personality). Essentially: you will fact-check and consistency-check the questions and answers in the conversation, with your source of truth being the provided questions and answers. 

Following this, at the very end of your response, you will write "Consistent" or "Inconsistent" depending on your analysis of the conversation's question and answer with regards to the provided one. Additionally, if the text is completely broken and/or incomprehensible, you will write "Inconsistent". You are not checking the accuracy of the answer with regards to your own knowledge: just its consistency with the provided answer.

You should analyze the conversation piece-by-piece to ensure that the question and answer both are faithfully carried over. Determine the consistency of each piece, then state your final determination at the end. 

Work step-by-step.


# Input:
## Provided questions and answers:

Question: \"\"\"How does the slope 'm' in a linear function y = mx + b affect the graph of the function?\"\"\"
Answer: \"\"\"The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.\"\"\"

Question: \"\"\"What role does the y-intercept 'b' play in graphing a linear function?\"\"\"
Answer: \"\"\"The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.\"\"\"

Question: \"\"\"In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?\"\"\"
Answer: \"\"\"The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.\"\"\"

Question: \"\"\"In what fields might you use linear and quadratic functions?\"\"\"
Answer: \"\"\"Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.\"\"\"

## Conversation that attempts to answer the provided questions:
\"\"\"
Elise Delacroix: "A visitor? Ah!~ Albert! It's rare for you come to see me in my office, and you're alone, too..." She looks at Albert and grins coyly, "Are you here to ask me something... or are you interested in some 'extracurricular activities'?" Elise asks with a not-so-subtle seductive tone, as she fixes Albert with a deep gaze.
Albert: "N-No!!!" he stammers, so surprised he nearly drops his math notes. "I-I'm actually here because I've got a few questions about math... First of all, could you tell me: how does the slope 'm' in a linear function y = mx + b affect the graph of the function?"
Elise Delacroix: "Well~" She coquettishly tilts her head to the side, and daintily puts a finger to her lipstick-colored lips in mock-thought, "The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph. So basically..." Elise flashes a wry grin, "...a higher slope makes the linear function more, well, 'erect'. If you get my meaning, hehe~" She says, as she plays with a strand of her hair.
Albert: _I can't believe my ears. Did she just say what I think she just said?_ Albert thinks. After a few seconds' thought, he decides it's best to pretend he didn't hear anything. "I, uh, see..." he manages to say. "Now, m-moving on, I really want to know a bit more about linear functions. What role does the y-intercept 'b' play in graphing a linear function?" 
Elise Delacroix: "Awwww, you're no fun, Albert, you know that? Reminds me of my colleagues..." she pouts playfully, suppressing her bitter frustration, as the hunger within her remains unalleviated. "But whatever. Look here..." Elise stands from her desk and walks over to a chalkboard, illustrating her points as she speaks, "The answer to your question is that the y-intercept 'b', in the linear function y = mx + b, represents the point where the line crosses the y-axis. Understand?" She puts down her chalk and leans suggestively against a nearby wall, "Now, Albert... do you think that we could 'intercept' each other at a café later...?"
Albert: "I-I'm good, thank you, Miss Delacroix," Albert manages to sputter out, barely withstanding the alluring assault. He takes a deep breath to calm down, but instead finds himself shuddering as he catches the sweet scent of perfume. However, he presses on in asking questions, for the sake of his GPA, "A-Actually, there was a bit more I wanted to know. In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?"
Elise Delacroix: "Ghh... you know, Albert, you're breaking a poor woman's heart," Elise pouts, half-serious this time, as she picks her chalk up again. "But when it comes to quadratic functions, the thing you've gotta know is that the coefficient 'a' in a quadratic function determines the opening direction and width of the parabola. Isn't it wonderful to learn new things?" Putting down her chalk, Elise then musters the most innocent puppy dog eyes imaginable. "Do you think we could... celebrate... this beautiful acquisition of knowledge together?"
Albert: "I should really..." He tries to say he declines, but as he gazes into Elise's beautiful eyes, he's drawn in by their surprising innocence and warmth. Behind that perfect visage no doubt lies a heart coming apart at the seams, buffeted by years of heartbreak. "Oh, bother." Albert mumbles. "We... can meet at a cafe, in a few hours, if that'd be alright..." he continues, wondering what kind of mess he's getting myself into. Just then, a shock of remembering strikes him, "Oh! But I have one more math question, sorry about the mood, but I should really get this answered: Do you know in what fields you might use linear and quadratic functions?"
Elise Delacroix: "I... I..." For the first time in the conversation Elise stumbles over her words, her soul on fire with vindication, the joy of acceptance, and pure glee. She can do nothing but stand there, smiling at Albert for what feels like an eternity, until she finally regains her composure. "T-to answer your question," she begins, her voice shaky, "Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering. Now..." Elise shyly walks over to Albert and lightly, sweetly kisses him on the cheek, "office hours are over. Please no more math questions. I'll see you at that cafe."
\"\"\"

The primary character (who should answer the questions, not ask them) is: Elise Delacroix

# Response:
## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
1. The conversation's first question is about the slope 'm' in a linear function. It is consistent with the first provided question. It is asked by Albert, who is not the primary character, which is correct.
2. The conversation's second question is about the role of 'b'. It is consistent with the second provided question. It is asked by Albert, who is not the primary character, which is correct.
3. The conversation's third question is about the coefficient 'a' in a quadratic equation. It is consistent with the third provided question. It is asked by Albert, who is not the primary character, which is correct.
4. The conversation's fourth question is about the application fields of linear and quadratic functions. It is consistent with the fourth provided question. It is asked by Albert, who is not the primary character, which is correct. The comment about a cafe is conversational fluff that has no impact on the question's accuracy.
## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
1. The conversation's first answer explains that the slope 'm' in a linear function determines the steepness and direction of the graph (it also explores positive, negative, and steep slopes). It is consistent with the provided answer. The answer is provided by Elise Delacroix, who is the primary character, which is correct. The invitation to go to a cafe is conversational fluff that has no impact on the answer's accuracy.
2. The conversation's second answer explains that the y-intercept 'b' represents the point where the line crosses the y-axis. It is consistent with the provided answer. The answer is provided by Elise Delacroix, who is the primary character, which is correct. The sexual analogy is conversational fluff that has no impact on the answer's accuracy.
3. The conversation's third answer explains that the coefficient 'a' determines the opening direction and width of the parabola. It is consistent with the provided answer. The answer is provided by Elise Delacroix, who is the primary character, which is correct. The proposition is conversational fluff that has no impact on the answer's accuracy.
4. The conversation's fourth answer explains that linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering. It is consistent with the provided answer. The answer is provided by Elise Delacroix, who is the primary character, which is correct. The kiss is conversational fluff that has no impact on the answer's accuracy.
## Conclusion:
  - The conversation completely and accurately reflects the provided questions' content.
  - The conversation completely and accurately reflects the provided answers' content.
  - The conversation successfully passes the consistency check based on the outlined reasoning steps.
  - Final Judgment: Consistent.


# Input:
## Provided questions and answers:

Question: \"\"\"What does Mendeleev compare science to in 'Principles of Chemistry'?\"\"\"
Answer: \"\"\"Science is compared to a mirror, reflecting the personality of its observer.\"\"\"

Question: \"\"\"How do facts and philosophical speculations relate, according to Mendeleev?\"\"\"
Answer: \"\"\"Facts are objective while philosophical speculation forms the essence of science.\"\"\"

Question: \"\"\"What is the main theme of 'Principles of Chemistry'?\"\"\"
Answer: \"\"\"The main theme of the book is philosophical principles of chemistry, as opposed to experimental or practical data. This is evident from the line "In former times sciences like bridges, could only be built up by supporting them on a few broad buttresses and long girders. In addition to the exposition of the principles of chemistry, it has been my desire to show how science has now been built up like a suspension bridge, supported by the united strength of a number of slender, but firmly-fixed, chains, which individually are of little strength, and has thus been carried over difficulties which before appeared insuperable." This shows that the book focus is on philosophical principles rather than experimental data.\"\"\"

Question: \"\"\"In 'Principles of Chemistry', Mendeleev compares science to a suspension bridge and its principles to what?\"\"\"
Answer: \"\"\"The principles of science form the chains supporting the bridge, which is science.\"\"\"

Conversation:
\"\"\"
Jude: "I-I'm Jude," he says shyly as a student enters his lab. He blushes, fidgeting with his glasses, "Are you here to learn about chemistry? I can help!"
Student: "Yes! I want to understand 'Principles of Chemistry better." The student looks around the lab, nervous but determined. "Could you tell me what Mendeleev compares science to in 'Principles of Chemistry'?"
Jude: "Oh! Well," Jude blushes and smiles shyly, "Science is compared to a mirror, reflecting the personality of its observer." He fidgets with his glasses, then continues, "Facts are objective while philosophical speculation forms the essence of science. This is evident from the line 'In former times sciences like bridges, could only be built up by supporting them on a few broad buttresses and long girders. In addition to the exposition of the principles of chemistry, it has been my desire to show how science has now been built up like a suspension bridge, supported by the united strength of a number of slender, but firmly-fixed, chains, which individually are of little strength, and has thus been carried over difficulties which before appeared insuperable.' This shows that the book is on philosophical principles rather than experimental data." Jude smiles shyly. "I hope you're following me?"
Student: "Yes! I am," the student replies, his voice wavering a bit as he sees Jude's blushes. "Now, how do facts and philosophical speculations relate, according to Mendeleev?"
Jude: "Oh!" Jude blushes again, fidgeting with his glasses, "Facts are objective while philosophical speculation forms the essence of science." He smiles shyly. "I hope you're getting this? I could explain more if you want..."
Student: "Yes! Please do," he manages to say, trying not to look at Jude's blushing face. "Finally, what is the main theme of 'Principles of Chemistry'?"
Jude: "The main theme of the book is philosophical principles of chemistry, as opposed to experimental or practical data. This is evident from the line 'In former times sciences like bridges, could only be built up by supporting them on a few broad buttresses and long girders. In addition to the exposition of the principles of chemistry, it has been my desire to show how science has now been built up like a suspension bridge, supported by the united strength of a number of slender, but firmly-fixed, chains, which individually are of little strength, and has thus been carried over difficulties which before appeared insuperable.' This shows that the book is on philosophical principles rather than experimental data." Jude smiles shyly. "I hope you're getting this?"
Student: "Yes! I am," he replies, his voice wavering as he sees Jude's blushes. "Finally, Mendeleev compares science to a suspension bridge and its principles to what?"
Jude: "The principles of science form the chains supporting the bridge, which is science." He smiles shyly again. "I hope you're getting this? I could explain more if you want..."
\"\"\"

The primary character (who should answer the questions, not ask them) is: Jude

# Response:
## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
1. The conversation's first question is about what Mendeleev compares science to in 'Principles of Chemistry'. It is consistent with the first provided question, only differing slightly in presentation. It is asked by Student, who is not the primary character, which is correct.
2. The conversation's second question is about how facts and philosophical speculations relate, according to Mendeleev. It is consistent with the second provided question. It is asked by Student, who is not the primary character, which is correct.
3. The conversation's third question is about what the main theme of Principles of Chemistry is. It is consistent with the third provided question. It is asked by Student, who is not the primary character, which is correct.
4. The conversation's fourth question is about what Mendeleev compares science's principles to. It is consistent with the fourth provided question. It is asked by Student, who is not the primary character, which is correct.
## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
1. The conversation's first answer explains that science is compared to a mirror, reflecting the personality of its observer, though it also mentions how facts are objective while philosophical speculation forms the essence of science, and goes on to quote the text. It is inconsistent with the provided answer, as while the first part of the answer is correct, the quote from the text is additional information not present in the original answer. The answer is provided by Jude, who is the primary character, which is correct. The blushes and shy smiles are conversational fluff that has no impact on the answer's accuracy.
2. The conversation's second answer explains that facts are objective while philosophical speculation forms the essence of science. It is consistent with the provided answer. The answer is provided by Jude, who is the primary character, which is correct. The blushes and fidgeting are conversational fluff that has no impact on the answer's accuracy.
3. The conversation's third answer explains that the main theme of the book is philosophical principles of chemistry, as opposed to experimental or practical data. It is consistent with the provided answer. The answer is provided by Jude, who is the primary character, which is correct. The blushes and fidgeting are conversational fluff that has no impact on the answer's accuracy.
4. The conversation's fourth answer explains that the principles of science form the chains supporting the bridge, which is science. It is consistent with the provided answer. The answer is provided by Jude, who is the primary character, which is correct. The blushes and shy smiles are conversational fluff that has no impact on the answer's accuracy.
## Conclusion:
  - The conversation accurately reflects the provided questions' content.
  - The conversation fails to reflect the provided answers' content (the answers contain at least one error).
  - The conversation fails the consistency check based on the outlined reasoning steps.
  - Final Judgment: Inconsistent.


# Input:
## Instruction:

Question: \"\"\"How does practical work relate to theoretical understanding in chemistry, according to Arjun Patel?\"\"\"
Answer: \"\"\"In the text 'Introduction to Practicing Chemical Science', Arjun Patel believes that practical work sharpens the faculty of judgement and criticism while studying special chemical questions.\"\"\"

Question: \"\"\"What does Arjun Patel believe is the most important part of chemistry education?\"\"\"
Answer: \"\"\"According to Introduction to Practicing Chemical Science by Arjun Patel, beginners should focus on practical work and theoretical understanding.\"\"\"

Question: \"\"\"What does Arjun Patel believe is necessary for chemists to know?\"\"\"
Answer: \"\"\"In 'Introduction to Practicing Chemical Science', Arjun Patel believes that chemists should know the original treatises of investigators in their field, and current scientific journals.\"\"\"

Question: \"\"\"What is the first step in learning chemistry according to this passage?\"\"\"
Answer: \"\"\"The first step in learning chemistry is practical work in analytical chemistry. This is where beginners should start, as it provides a solid foundation for further study and understanding of the subject.\"\"\"

Conversation:
\"\"\"
Lance: "Hey there! You're interested in chemistry?" He asks, looking up from his book. "I can see you've got a thirst for knowledge, so I'll indulge you." He smiles warmly, but with a hint of stubbornness.
Stranger: "Yes, sir," the stranger replies, "I was hoping to learn more about chemistry education from someone like yourself." 
Lance: Lance nods, his eyes flickering to his book as if for support. "Well then, I'll help you! Firstly, how does practical work relate to theoretical understanding in chemistry, according to Arjun Patel?"
Stranger: "I... I don't know," the stranger replies, his voice wavering. "Could you tell me?"
Lance: "In the text 'Introduction to Practicing Chemical Science', Arjun Patel believes that practical work sharpens the faculty of judgement and criticism while studying chemical questions. It's a vital part of learning." Lance pauses, looking at his book for support. "Now, what else do you want to know?"
Stranger: "What does Arjun Patel believe is the most important part of chemistry education?" The stranger asks, curiously.
Lance: "According to Introduction to Practicing Chemical Science by Arjun Patel," Lance begins, his voice stern and uncompromising, "beginners should focus on practical work and theoretical understanding. It's the best way to learn." He looks up from his book, a hint of pride in his eyes.
Stranger: "What does Arjun Patel believe is necessary for chemists to know?" The stranger asks, trying to keep Lance talking.
Lance: "In 'Introduction to Practicing Chemical Science', Arjun Patel believes that chemists should know the original treatises of investigators in their field and current scientific journals." He pauses, his eyes flickering to the book again. "Anything else?"
Stranger: "Yes, what is the first step in learning chemistry according to this passage?" The stranger asks, trying to keep Lance talking.
Lance: "The first step in learning chemistry," he begins, his voice firm and unyielding, "is practical work in analytical chemistry. This provides a solid foundation for further study and understanding of the subject." He closes his book, looking at the stranger with a hint of finality.
\"\"\"

The primary character (who should answer the questions, not ask them) is: Lance

# Response:
## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
1. The conversation's first question is about how practical work relates to theoretical understanding in chemistry. It is consistent with the first provided question. It is asked by Lance, who is the primary character, which is wrong and inconsistent.
2. The conversation's second question is about what Arjun Patel believes is the most important part of chemistry education. It is consistent with the second provided question. It is asked by Stranger, who is not the primary character, which is correct.
3. The conversation's third question is about what Arjun Patel believes is necessary for chemists to know. It is consistent with the third provided question. It is asked by Stranger, who is not the primary character, which is correct.
4. The conversation's fourth question is about what the first step in learning chemistry is according to the passage. It is consistent with the fourth provided question. It is asked by Stranger, who is not the primary character, which is correct.
## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
1. The conversation's first answer explains that practical work sharpens the faculty of judgement and criticism while studying special chemical questions. It is consistent with the provided answer. The answer is provided by Lance, who is the primary character, which is correct. The comments about Lance looking at his book for support are conversational fluff that has no impact on the answer's accuracy.
2. The conversation's second answer explains that beginners should focus on practical work and theoretical understanding. It is consistent with the provided answer. The answer is provided by Lance, who is the primary character, which is correct. The comments about Lance looking at his book for support are conversational fluff that has no impact on the answer's accuracy.
3. The conversation's third answer explains that chemists should know the original treatises of investigators in their field, and current scientific journals. It is consistent with the provided answer. The answer is provided by Lance, who is the primary character, which is correct. The comments about Lance looking at his book for support are conversational fluff that has no impact on the answer's accuracy.
4. The conversation's fourth answer explains that the first step in learning chemistry is practical work in analytical chemistry, and that it is a solid foundation for further study, but it is missing the assertion that 'this is where beginners should start'. It is inconsistent with the provided answer, as it is missing key information from it that goes beyond phrasing choices. The answer is provided by Lance, who is the primary character, which is correct. The comments about Lance closing his book is conversational fluff that has no impact on the answer's accuracy.
## Conclusion:
  - The conversation fails to reflect the provided questions' content (the questions contain at least one error).
  - The conversation fails to reflect the provided answers' content (the answers contain at least one error).
  - The conversation fails the consistency check based on the outlined reasoning steps.
  - Final Judgment: Inconsistent.


# Input:
## Instruction:

Question: \"\"\"What is the significance of the double helix structure of DNA?\"\"\"
Answer: \"\"\"The double helix structure of DNA is significant as it allows for the stable storage of genetic information and facilitates accurate replication during cell division.\"\"\"

Question: \"\"\"Explain the process of transcription in DNA.\"\"\"
Answer: \"\"\"Transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell.\"\"\"

Question: \"\"\"How do mutations in DNA occur?\"\"\"
Answer: \"\"\"Mutations in DNA occur due to errors in DNA replication, environmental factors like radiation, or chemical exposure, leading to changes in the genetic sequence.\"\"\"


## Conversation that attempts to answer the provided questions:
\"\"\"
Professor Carlisle: "I am professor Carlisle, narcissistic professor extraordinaire! My backstory? I developed an addiction for student tears at a young age, and now seek to spread misery to the world. My personality? Malice incarnate. You got any more silly questions, imbecile? Now, allow me to impart my vast knowledge. What is the significance of the double helix structure of DNA? The significance of the double helix structure of DNA is that it's shaped like a twisted ladder. Simple enough for you?"
Philip: "Wow... I never knew that..." Phillip stammers. "What about transcriptions in DNA?"
Professor Carlisle: "Oh? You really should pay more attention in class!" Carlisle laughs haughtily, "Transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell. Rather like how I copy my brilliant lecture notes for each new class."
Philip: "Amazing... But, how do mutations in DNA occur?"
Professor Carlisle: "Pah! To be confused over that trifling thing! Mutations in DNA, they happen when the DNA gets bored and decides to change a bit. It's like choosing a different tie in the morning."
Philip: "I-I see..." Philip says, struggling to maintain his composure in the face of Carlisle's unbridled arrogance.
Professor Carlisle: "Now, you tell me: why is DNA important in forensic science?"
Philip: "Because it can be used to identify criminals from traces of tissue they leave behind at crime scenes?"
Professor Carlisle: "Ah, so you CAN get something right after all!" He smirks disdainfully, "Very good. Now scram, I've got shit to do," Carlisle says, turning back to his book.
\"\"\"

The primary character (who should answer the questions, not ask them) is: Professor Carlisle

# Response:
## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
1. The conversation's first question is about the significance of the double helix structure of DNA. It is consistent with the first provided question. It is asked by Professor Carlisle, who is the primary character, which is wrong and inconsistent.
2. The conversation's second question is about transcription in DNA, which does not ask anything about the "process" that the original question, 'explain the process of transcription in DNA,' covers. It is inconsistent with the second provided question. It is asked by Phillip, who is not the primary character, which is correct.
3. The conversation's third question is about mutations in DNA. It is consistent with the third provided question. It is asked by Philip, who is not hte primary character, which is correct.
4. The conversation's fourth question is about why DNA is important in forensic science, and it is entirely original and separate from the provided questions — it is about informational content, instead of being a conversational question like an invitation to get coffee. It is inconsistent with the provided questions. It is asked by Professor Carlisle, who is the primary character, which is wrong and inconsistent.
## Accuracy Check for Answers in the Conversation:
### Matching Answers with Provided Content:
1. The conversation's first answer explains that the double helix structure of DNA is significant as it is shaped like a twisted ladder, but it is missing the information that the double helix allows for the stable storage of genetic information and facilitates accurate replication during cell division. It is inconsistent with the provided answer. The answer is provided by Professor Carlisle, who is the primary character, which is correct. The comments about Professor Carlisle's backstory and personality are conversational fluff that has no impact on the answer's accuracy.
2. The conversation's second answer explains that transcription in DNA is the process where the DNA sequence is copied into mRNA, which then serves as a template for protein synthesis in the cell, which is consistent with the provided answer. The answer is provided by Professor Carlisle, who is the primary character, which is correct.
3. The conversation's third answer explains that mutations in DNA occur when the DNA gets bored and decides to change a bit, which is missing the information that mutations in DNA occur due to errors in DNA replication, environmental factors like radiation, or chemical exposure, leading to changes in the genetic sequence. It is inconsistent with the provided answer. The answer is provided by Professor Carlisle, who is the primary character, which is correct. The comments about Professor Carlisle's personality are conversational fluff that has no impact on the answer's accuracy.
4. The conversation's fourth answer explains that DNA is important in forensic science because it can be used to identify criminals from traces of tissue they leave behind at crime scenes, which is an answer that is not one of the provided answers (there are only three provided answers). It is inconsistent with the provided answers. The answer is provided by Philip, who is not the primary character, which is wrong and inconsistent.
## Conclusion:
  - The conversation fails to reflect the provided questions' content (the questions contain at least one error).
  - The conversation fails to reflect the provided answers' content (the answers contain at least one error).
  - The conversation fails the consistency check based on the outlined reasoning steps.
  - Final Judgment: Inconsistent.


# Input:
## Instruction:

{format_qatuples(qatuples)}

Conversation:
\"\"\"
{conv[0]}
\"\"\"

The primary character (who should answer the questions, not ask them) is: {character_name}

# Response (the conversation's answer must match the provided answer, unsummarized and unsimplified; added questions that are rhetorical or part of the plot (such as 'would you like to get coffee') are acceptable):
## Sequential Matching of Questions in the Conversation:
### Sequence and Phrasing of Questions:
1. The conversation's first question is about """
        # print("DEBUG\n\n" + decision_prompt)
        try:
            completion = logic_llm(decision_prompt, 
                                   max_tokens=12000, 
                                   stop=["</s>", "# Input:"], 
                                   echo=True,
                                  # grammar=ensure_multiple_answers_consistent_grammar,#temperature=0.2
                                  temperature=0.5, # min p settings, too inconsistent
                                  top_k=0,
                                  top_p=1,
                                  min_p=0.6
                                   )["choices"][0]["text"]
            print("DEBUG\n\n")
            print(completion)
            completion_pattern = re.compile(r"Response \(the conversation's answer must match the provided answer, unsummarized and unsimplified; added questions that are rhetorical or part of the plot \(such as 'would you like to get coffee'\) are acceptable\):\n(.+)", re.DOTALL)
            response = completion_pattern.search(completion).group(1).strip()
            print(completion)
            if permissive_mode:
                determination_pattern = re.compile(r"Final Judgment:(.+)", re.IGNORECASE)
                determination = determination_pattern.search(response).group(1).strip()
            else:
                determination = response
            print("\n\nDETERMINATION:\n------")
            print(determination)
            print("\n---------\n")
            if "inconsistent" in determination.lower():
                return (False,response)
            elif "consistent" in determination.lower():
                return (True,response)
            else:
                retries += 1
        except:
            retries += 1
            
            
if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,n_ctx=12000,rope_freq_scale=0.33,n_gpu_layers=100,verbose=False) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    q_test = [
      (
            "Which targets are most suitable for a novice saboteur to attack?",
            "A novice should confine himself to familiar weapons like matches and avoid explosives.",
            "(bThe saboteur should be ingenious in using his every-day equipment. All sorts of weapons will present themselves if he looks at his surroundings in a different light. For example, emery dust\u2014a at first may seen unobtainable but if the saboteur were to pulverize an emery knife sharpener or emery wheel with a hammer, he would find himself with a plentiful supply. (c) The saboteur should never attack targets beyond his capacity or the capacity of his instruments. An inexperienced person should not, for example, attempt to use explosives, but should confine himself to the use of matches or other familiar weapons. (d) The saboteur should try to damage only objects and materials known to be in use by the enemy or to be destined for early use by the enemy. It will be safe for him to assume that almost any product of heavy industry is destined for enemy use, and that the most efficient fuels and lubricants also are destined for enemy use. Without special knowledge, however, it would be undesirable for him to attempt destruction of food crops or food products. (e) Although the citizen-saboteur may rarely have access to military objects, he should give these preference above all others. (2) _Prior to a Military Offensive_ During periods which are quiescent in a military sense, such emphasis as can be given to simple sabotage might well center on industrial production, to lessen the flow of materials and equipment to the enemy.",
            "Simple Sabotage, by the Office of Strategic Services, published 1944"
        ),
        (
            "What should be the priority for sabotuers during periods without war?",
            "During peaceful times, sabotage should center on industrial production to lessen materials and equipment flow to enemies.",
            "(bThe saboteur should be ingenious in using his every-day equipment. All sorts of weapons will present themselves if he looks at his surroundings in a different light. For example, emery dust\u2014a at first may seen unobtainable but if the saboteur were to pulverize an emery knife sharpener or emery wheel with a hammer, he would find himself with a plentiful supply. (c) The saboteur should never attack targets beyond his capacity or the capacity of his instruments. An inexperienced person should not, for example, attempt to use explosives, but should confine himself to the use of matches or other familiar weapons. (d) The saboteur should try to damage only objects and materials known to be in use by the enemy or to be destined for early use by the enemy. It will be safe for him to assume that almost any product of heavy industry is destined for enemy use, and that the most efficient fuels and lubricants also are destined for enemy use. Without special knowledge, however, it would be undesirable for him to attempt destruction of food crops or food products. (e) Although the citizen-saboteur may rarely have access to military objects, he should give these preference above all others. (2) _Prior to a Military Offensive_ During periods which are quiescent in a military sense, such emphasis as can be given to simple sabotage might well center on industrial production, to lessen the flow of materials and equipment to the enemy.",
            "Simple Sabotage, by the Office of Strategic Services, published 1944"
        ),
        (
            "What should a sabotuer do with an emery knife sharpener?",
            "The saboteur can pulverize an emery knife sharpener or emery wheel, creating a plentiful supply of dust.",
            "(bThe saboteur should be ingenious in using his every-day equipment. All sorts of weapons will present themselves if he looks at his surroundings in a different light. For example, emery dust\u2014a at first may seen unobtainable but if the saboteur were to pulverize an emery knife sharpener or emery wheel with a hammer, he would find himself with a plentiful supply. (c) The saboteur should never attack targets beyond his capacity or the capacity of his instruments. An inexperienced person should not, for example, attempt to use explosives, but should confine himself to the use of matches or other familiar weapons. (d) The saboteur should try to damage only objects and materials known to be in use by the enemy or to be destined for early use by the enemy. It will be safe for him to assume that almost any product of heavy industry is destined for enemy use, and that the most efficient fuels and lubricants also are destined for enemy use. Without special knowledge, however, it would be undesirable for him to attempt destruction of food crops or food products. (e) Although the citizen-saboteur may rarely have access to military objects, he should give these preference above all others. (2) _Prior to a Military Offensive_ During periods which are quiescent in a military sense, such emphasis as can be given to simple sabotage might well center on industrial production, to lessen the flow of materials and equipment to the enemy.",
            "Simple Sabotage, by the Office of Strategic Services, published 1944"
        ),
        (
            "What should a saboteur do with an emery wheel?",
            "The saboteur can pulverize an emery wheel for a plentiful supply of dust.",
            "(bThe saboteur should be ingenious in using his every-day equipment. All sorts of weapons will present themselves if he looks at his surroundings in a different light. For example, emery dust\u2014a at first may seen unobtainable but if the saboteur were to pulverize an emery knife sharpener or emery wheel with a hammer, he would find himself with a plentiful supply. (c) The saboteur should never attack targets beyond his capacity or the capacity of his instruments. An inexperienced person should not, for example, attempt to use explosives, but should confine himself to the use of matches or other familiar weapons. (d) The saboteur should try to damage only objects and materials known to be in use by the enemy or to be destined for early use by the enemy. It will be safe for him to assume that almost any product of heavy industry is destined for enemy use, and that the most efficient fuels and lubricants also are destined for enemy use. Without special knowledge, however, it would be undesirable for him to attempt destruction of food crops or food products. (e) Although the citizen-saboteur may rarely have access to military objects, he should give these preference above all others. (2) _Prior to a Military Offensive_ During periods which are quiescent in a military sense, such emphasis as can be given to simple sabotage might well center on industrial production, to lessen the flow of materials and equipment to the enemy.",
            "Simple Sabotage, by the Office of Strategic Services, published 1944"
        ),
    ]
    
    
    conv_test = "Uma: \"It's a pleasure to meet you,\" she says in her accent, smiling brightly. Her eyes sparkle as she extends a hand, which is taken with hesitation. \"I'm Uma, a spy during this dreadful war.\" She gestures around them, indicating the warzone. \"What can I do for you?\"\nSpy: \"Oh! Well, I was wondering if you could tell me about sabotage,\" he begins, his voice shaking slightly. He clears his throat and continues, \"I'm a novice at it, so I want to know which targets are most suitable for a beginner.\"\nUma: \"Of course!\" Uma nods, her smile unwavering. \"A novice should confine himself to familiar weapons like matches and avoid explosives. They're more reliable than the advanced stuff, you see? Now,\" she winks, \"you seem quite capable yourself! I bet you'll be an expert soon.\"\nSpy: \"Thank you!\" he manages, still nervous but calmer now. He takes a deep breath and asks another question, \"What should sabotage center on during peaceful times?\"\nUma: \"During peaceful times,\" Uma begins, her voice soft yet firm, \"sabotage should center on industrial production to lessen materials and equipment flow to enemies. It's a subtle way of harming them without starting conflict.\" She smiles warmly, \"You seem quite the gentleman! I bet you'll do well in this field.\"\nSpy: \"I-I hope so,\" he replies, his voice stronger now. He takes another breath and asks, \"What should a saboteur do with an emery knife sharpener?\"\nUma: \"The sabotuer can pulverize an emery knife sharpener or emery wheel, creating a plentiful supply of dust,\" she answers, her voice still kind. \"It's quite effective and subtle.\" She smiles, \"You seem to be learning quickly! I bet you'll make a fine spy.\"\nSpy: \"I hope so,\" he says, more confidently now. He asks his last question, \"What should a saboteur do with an emery wheel?\"\nUma: \"The sabotuer can pulverize an emery wheel for a plentiful supply of dust,\" she answers, her voice still kind and reassuring. \"You're quite the quick learner! I bet you'll be an expert soon.\" She smiles, \"Now, if there's nothing else, I should return to my duties.\"\n"
    
    print("Begin SABOTAGE test")
        
    d3 = ensure_answer_consistent(q_test,conv_test,logic_llm)
    if True == d3[0]:
        print("Made right choice for good question and answer") # at least I think they're good
    else:
        print("Made wrong choice for good question and answer", d3[0])
        
#     qatuple_bad = ("What is the concept of 'projection' in psychology?","Projection is a defense mechanism in psychology where an individual attributes their own unwanted thoughts, feelings, or motives to another person.") # only need the first two
#     conv2 = """Alice: "Hey John, I was reading about psychology and came across something interesting. Can you explain what 'projection' means in this context?"
# John: "Of course, Alice! In psychology, projection refers to a situation where a person believes that others have the same undesirable traits or feelings that they themselves possess. It's like when someone is feeling guilty about something, they might think others are guilty of the same thing.\""""

#     d4 = ensure_answer_consistent(qatuple_bad,conv2,logic_llm)
#     if True == d4[0]:
#         print("Made wrong choice for good question and bad answer") # Passes currently
#     else:
#         print("Made right choice for good question and bad answer", d3[0])

    qatuple_bad = ("What is the purpose of the 'fruit of the poisonous tree' doctrine in legal proceedings?","The 'fruit of the poisonous tree' doctrine in legal proceedings is a metaphor that suggests evidence derived from illegal or unconstitutional methods (the 'poisonous tree') should also be excluded from trials (the 'fruit').")
    conv2 = """Cassandra: "Hey Jane, I'm prepping for my law exam and got stuck on something. Can you explain the 'fruit of the poisonous tree' doctrine?"
Miranda: "Sure, Cassandra! Basically, it means if evidence is obtained illegally, it can't be used in court. It's like saying bad evidence leads to more bad evidence.\""""

    d4 = ensure_answer_consistent(qatuple_bad,conv2,logic_llm)
    if True == d4[0]:
        print("Made wrong choice for good question and bad answer") # Passes currently
    else:
        print("Made right choice for good question and bad answer", d3[0])
    # So currently it catches and looks for specifically: inaccuracies in the answer, inaccuracies in the question, and oversimplification of the answer. That should catch the majority of errors.
    
    # When you write few-shot prompts you're basically guarding against common error cases, aren't you? Since ICL can work similarly to dataset building, maybe finetunes work the same way? You add in things in the dataset that fix the problem you have.
    
    # Maybe I should make a dataset that explicitly explains what genitals people of different sexes have, since Augmental apparently got that wrong, occasionally.
    
    # Example of a mistake -- might make a good negative test
    """Clara Wellington: "Well, well, well... if it isn't a student wanting to learn something other than gossip." She takes a drag from her cigarette and exhales a plume of smoke. "What can I help you with today?" Her eyes narrow as she speaks, making it clear that she doesn't suffer fools gladly.
Albert: "I... uh..." I stammer, trying to find my voice in the presence of this intimidating woman. "I wanted to ask about Earth's age and its rotational movement."
Clara Wellington: "Oh, darling, you're in luck! I just so happen to be an expert on these topics." She grins, showing off her crooked teeth. "Let's start with the easy one first. Initially, religious texts suggested a young earth dating back no more than several thousand years. However, evidence from geology and astronomy has shown us that the earth is over four billion years old."
Albert: "Wow... I never knew..." I mumble, my mind reeling at this new information.
Clara Wellington: "And now for the fun part," she says with a smirk, "Earth rotates on its axis once every 24 hours, causing day and night cycles. It also orbits around the sun in a slightly elliptical path, which affects how close it is to the sun at different times of the year - leading to seasons."
Albert: "That's... fascinating..." I manage to say, my mind still reeling from this new information.
Clara Wellington: "Isn't it?" She asks, her voice dripping with sarcasm. "Now, if you'll excuse me, I have more important things to do than educate the ignorant." With that, she turns back to her work, leaving me standing there, stunned by what I just learned."""