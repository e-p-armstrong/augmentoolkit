import re
# try:
from .questions_grammar import questions_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
from .strip_steps import strip_steps

def generate_questions(para_tuple, plan,logic_llm): # TODO make it so that this incorporates information about what text this is. So that if you feed it old latin legal texts, say, it'll be able to say "What is the old latin legal principle xxxx" instead of "What is the legal principle xxxx" thus causing confusion.
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)
    
    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    made_questions = False
    retries = 0
    questions = []
    while (not made_questions and (retries <= 5)): 
        question_prompt = f"""# Input:
You are an expert educational AI that, given a paragraph or two from a text, will create suitable educational questions based on the paragraphs, and *only* based on the paragraphs. You are focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels). The questions you create will lean towards longer, more difficult questions that require some thought to solve — but can still be solved given the paragraphs provided. Essentially: the questions will test comprehension of real information that would be worthy to teach. After the question, you will also write its answer.

Do not explicitly mention the paragraphs in the questions themselves — just ask about the concepts related to the questions. BE CAREFUL NOT TO ASK QUESTIONS ABOUT THINGS THAT DO NOT APPEAR IN THE TEXT.

You will not mention the text explicitly in any questions you think of, since the questions you generate are intended to test people's knowledge of the information — when given the questions, they will not have the text on-hand.

# Input:
## Instruction:

Text details: Introduction to Mathematics, by Jane Smith

Text to make questions from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

# Response:
## Reasoning and thought process:
Identify Key Topics: The key topics in this paragraph are linear and quadratic functions in mathematics, their definitions, components, and applications.
Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and components of these functions. Example: "What does the 'm' in the linear function equation represent?"
Develop Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the equations. Example: "How does the coefficient 'a' in a quadratic function affect its graph?"
Create Questions Investigating Usage of Concepts: Look for information about when these concepts should be applied. Example: "What kind of function would you use to graph a a parabola?"
Remember to Not Mention the Text: I will not mention the text or the author in my answer, as the student will not have access to or know about those resources when asked the question.

## Questions:
1.) How does the slope 'm' in a linear function y = mx + b affect the graph of the function?
Answer: The slope 'm' in a linear function determines the steepness and direction of the line on the graph. A positive slope means the line ascends from left to right, while a negative slope indicates it descends. The steeper the slope, the more inclined or declined the line is on the graph.

2.) What role does the y-intercept 'b' play in graphing a linear function?
Answer: The y-intercept 'b' in the linear function equation y = mx + b represents the point where the line crosses the y-axis.

3.) In the equation of a quadratic function y = ax² + bx + c, how does the coefficient 'a' influence the graph of the function?
Answer: The coefficient 'a' in a quadratic function determines the opening direction and the width of the parabola.

4.) In what fields might you use linear and quadratic functions?
Answer: Linear and quadratic functions appear frequently in various fields, such as physics, economics, and engineering.

# Input:
## Instruction:

Text details: Thus Spake Zaranthustra, by Friedrich Nietzsche

Text to make questions from: 
\"\"\"
When Zarathustra was thirty years old, he left his home and the lake of his home, and went into the mountains. There he enjoyed his spirit and
solitude, and for ten years did not weary of it. But at last his heart changed,—and rising one morning with the rosy dawn, he went before the
sun, and spake thus unto it:

Thou great star! What would be thy happiness if thou hadst not those for
whom thou shinest!

For ten years hast thou climbed hither unto my cave: thou wouldst have
wearied of thy light and of the journey, had it not been for me, mine
eagle, and my serpent.

But we awaited thee every morning, took from thee thine overflow and
blessed thee for it.

Lo! I am weary of my wisdom, like the bee that hath gathered too much
honey; I need hands outstretched to take it.

I would fain bestow and distribute, until the wise have once more become
joyous in their folly, and the poor happy in their riches.

Therefore must I descend into the deep: as thou doest in the
evening, when thou goest behind the sea, and givest light also to the
nether-world, thou exuberant star!

Like thee must I GO DOWN, as men say, to whom I shall descend.
\"\"\"

# Response:
## Reasoning and Thought Process:
Identify Key Themes: In this excerpt, we find themes of transformation, self-discovery, the value of wisdom, and the need for recognition of effort and wisdom.
Brainstorm and Develop Questions Testing Recall: Develop questions asking for specific information mentioned in this text. For instance, "Finish the quote: I am weary of my wisdom, like..."
Devise Questions Testing Comprehension and Belief in the Text's Opinions: Questions can be formed to check understanding and acceptance of the text's philosophy. For instance, "What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?" Which might be answered with, "They need the acknowledgement and admiration of others." and then supported by lines such as "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" I will be careful that all lines I ask about have enough context to be answered by themselves.
Create Questions Investigating Interpretations: Given the text's rich language, I will devise questions interpreting its meaning, while being careful not to explicitly mention the text in doing so. Example: "Why did Zaranthustra, in 'Thus Spake Zaranthustra', decide to leave the mountains?"

## Questions:
1.) What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?
Answer: They need the acknowledgement and admiration of others. Take the line "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" This implies that even the wisest or the most enlightened individuals crave recognition for their efforts and wisdom, in order to further develop said wisdom and expend said efforts. They need others to see and appreciate the light they bring.

2.) Recite a famous quote from Thus Spake Zaranthustra that likens the solitary gathering of wisdom to a bee gathering honey.
Answer: "Lo! I am weary of my wisdom, like the bee that hath gathered too much honey; I need hands outstretched to take it."

3.) Why did Zaranthustra, in 'Thus Spake Zaranthustra', decide to leave the mountains?
Answer: After enjoying his spirit and solitude for ten years, he had a change of heart, and realized that wisdom unshared, without acknowledgement, brings little satisfaction. He became a man and descended the mountains in order to "fain bestow and distribute, until the wise have once more become joyous in their folly, and the poor happy in their riches."

4.) What are the parallels between the sun and Zaranthustra?
Answer: The sun rose every day for ten years, and was appreciated — had its "overflow" taken, and was blessed — for the action. Zaranthustra mentions of the sun, "What would be thy happiness if thou hadst not those for whom thou shinest!" — would the sun be happy, if it did not have people to shine for? In this same way, Zaranthustra's wisdom is shining, but there is no one to appreciate it. Thus, Zaranthustra descends the mountains, just like the sun descends the sky, in order to share his wisdom somewhere else — similar to how the sun shares its light in the nether-world, too.

# Input:
## Instruction:

Text details: Great Construction Projects Throughout History, by Marco Gonzalez

Text to make questions from: 
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

# Response:
## Reasoning and Thought Process:
# Identify Key Topics: The paragraph details specific aspects of the Panama Canal's construction, focusing on its challenges, innovations, and impacts. Topics include construction challenges, health issues, excavation techniques, and the canal's impact on global trade.
Brainstorm and Develop Questions Testing Recall: Questions can be formed to recall factual data from the text. Example: "How much earth was excavated during the construction of the Panama Canal?"
Develop Questions Exploring Cause and Effect Relationships: This involves creating questions that explore how certain challenges led to specific solutions and impacts. Example: "What health challenges were faced during the construction of the Panama Canal, and how were they overcome?"
Create Questions Investigating Quantitative Values: Given the text's focus on concrete numbers, I will devise questions that require analyzing these figures. Example: "By how many miles did the Panama Canal reduce the sea voyage from San Francisco to New York, and what does this imply about its impact on global trade?"

## Questions:
1.) How much earth was excavated during the construction of the Panama Canal?
Answer: Over 200 million cubic yards of earth were excavated during the construction of the Panama Canal, showcasing the scale of this massive engineering project.

2.) What health challenges were faced during the construction of the Panama Canal, and how were they overcome?
Answer: The construction faced significant health challenges, notably malaria and yellow fever. These were overcome through extensive public health measures, illustrating the importance of health considerations in large-scale engineering projects.

3.) By how many miles did the Panama Canal reduce the sea voyage from San Francisco to New York, and what does this imply about its impact on global trade?
Answer: The completion of the Panama Canal reduced the sea voyage from San Francisco to New York by around 8,000 miles, indicating a significant impact on global trade by greatly shortening maritime routes between the Atlantic and Pacific Oceans.

4.) What was the primary purpose of the Panama Canal, and how did its completion achieve this goal?
Answer: The primary purpose of the Panama Canal was to shorten the maritime route between the Atlantic and Pacific Oceans. Its completion, spanning approximately 50 miles, successfully achieved this goal by providing a much shorter and more efficient route for maritime traffic.

# Input:
## Instruction:

Text details: {para_tuple[1]}

Text to make questions from: 
\"\"\"
{para_tuple[0]}
\"\"\"

# Response:
## Reasoning and Thought Process:
{strip_steps(plan)}

## Questions (make 4):
"""
        # print("DEBUG\n\n" + decision_prompt)
        completion = logic_llm(question_prompt, max_tokens=4000, stop=["</s>"], echo=True,grammar=questions_grammar,temperature=0.2)["choices"][0]["text"]
        print("COMPLETION:\n\n----------------------")
        print(completion)
        print("\n------------------")
        
        # Extract questions
        response_pattern = re.compile(r"Questions \(make 4\):\n(.+)",re.IGNORECASE | re.DOTALL)
        generation = response_pattern.search(completion).group(1)
        print("GENERATION:\n\n-------------------\n\n", generation)
        pattern = re.compile(r'(?:Question:|^\d+[\).]?)\s*(.*?)\s*\n*Answer:\s*(.*?)(?=(?:\n\s*(?:Question:|\d+[\).]?))|$)', re.DOTALL | re.MULTILINE | re.IGNORECASE)
        matches = pattern.findall(generation)
        if len(matches) > 0:
            made_questions = True
        else:
            retries += 1
    if (retries > 5):
        return None

    for match in matches:
        questions.append((match[0].replace(") ","",1).strip(), match[1].replace(") ","",1).strip(),para_tuple[0].replace(") ","",1),para_tuple[1].replace(") ","",1)))
    
    return questions

# TODO fix the bug where the ) is included in the question text

if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    text = """The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.

The earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."""

    plan = """First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). I will start by looking at one or two sentences at a time. Let's begin with: "The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years." This paragraph is saying that people used to know only about 3,000 years of history, but now they know much more. So I might ask something like "What was the time period in which people had limited knowledge?" The question tests knowledge of when historical records were incomplete and therefore difficult to access. It requires understanding of the text as well as analysis to determine what time period is being referred to here.  Next, I'll look at: "Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end." This paragraph states that the universe seems infinite because of its reflection, but it could actually be finite with two ends. So I might ask something like "How does the structure of the universe affect its size?" This question tests understanding of the concept of reflection and how it can distort perceptions of size, as well as analysis to determine what implications this might have for the actual size of the universe. Then I'll move on: "The earth ... circles about the sun in a slightly distorted and slowly variable oval path in a year." This paragraph talks about the Earth's orbit around the Sun. So I might ask something like "What is the shape of the Earth's orbit around the sun?" This question tests understanding of the text as well as analysis to determine the specific shape mentioned here. Lastly, let's examine: "Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."  This paragraph says that the Earth's distance from the Sun changes throughout the year. So I might ask something like "Why does the Earth's distance from the Sun change over the course of the year?" This question tests understanding of the text as well as analysis to determine why this change occurs."""
    print("Begin HGWELLS test")
    # result = generate_question(text,plan,logic_llm)
    ## TODO a wider variety of tests from different texts
    
    print("Begin MENDELEEV Test")
    plan2 = """First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). Now, immediately I can see that the following passage might be a good basis for a question: "It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances." This seems like it could be a good basis for an application question. "By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties." This might also be a good basis for an application question. "Gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal." This seems like it could be a good basis for an analysis question. "Chemistry deals only with the homogeneous substances met with in nature, or extracted from natural or artificial non-homogeneous substances." This might also be a good basis for an application question."""
    text2 = """A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, copper) must be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about."""
      
      
      
      
      
      
      
    plan3 = """Step 1. Identify Key Topics: The key topics in this paragraph are homogeneous and non-homogeneous substances, their characteristics, examples of each type, and how they can be identified.
Step 2. Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and characteristics of homogeneous and non-homogeneous substances. Example: "What is a homogeneous substance?"
Step 3. Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the examples given in the text. Example: "How does the presence of 'orthoclase' affect the properties of porphyries?"
Step 4. Create Questions Investigating If-Then Relationships (where since one thing is true, another follows): Make questions that examine how different characteristics lead to identifying a substance as homogeneous or non-homogeneous. Example: "How can we determine if a substance is homogeneous based on its properties?\""""

    plan4 = """Step 1. Identify Key Topics: The key topics in this paragraph include the Will to Truth, its value, and the question of whether truth is more valuable than untruth or ignorance.
Step 2. Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of specific details about the Will to Truth. Example: "What is the 'Will to Truth' according to Nietzsche?"
Step 3. Devise Questions Exploring Relationships Between Concepts: Generate questions that explore how different concepts relate to each other within the text. Example: "How does the Will to Truth differ from a desire for truthfulness, as philosophers have traditionally understood it?"
Step 4. Create Questions Investigating Interpretations: Given the text's complex language and philosophical ideas, I will devise questions interpreting its meaning, while being careful not to explicitly mention the text in doing so. Example: "What does Nietzsche mean when he says that 'we made a long halt at the question as to the origin of this Will--until at last we came to an absolute standstill before a yet more fundamental question'?\""""
      
    text3="""The Will to Truth, which is to tempt us to many a hazardous
enterprise, the famous Truthfulness of which all philosophers have
hitherto spoken with respect, what questions has this Will to Truth not
laid before us! What strange, perplexing, questionable questions! It is
already a long story; yet it seems as if it were hardly commenced. Is
it any wonder if we at last grow distrustful, lose patience, and turn
impatiently away? That this Sphinx teaches us at last to ask questions
ourselves? WHO is it really that puts questions to us here? WHAT really
is this "Will to Truth" in us? In fact we made a long halt at the
question as to the origin of this Will--until at last we came to an
absolute standstill before a yet more fundamental question. We inquired
about the VALUE of this Will. Granted that we want the truth: WHY NOT
RATHER untruth? And uncertainty? Even ignorance? The problem of the
value of truth presented itself before us--or was it we who presented
ourselves before the problem? Which of us is the Oedipus here? Which
the Sphinx? It would seem to be a rendezvous of questions and notes of
interrogation. And could it be believed that it at last seems to us as
if the problem had never been propounded before, as if we were the first
to discern it, get a sight of it, and RISK RAISING it? For there is risk
in raising it, perhaps there is no greater risk."""
    
      
      
    result2 = generate_questions((text2,"Principles of chemistry"),plan3,logic_llm)
    # result3 = generate_questions((text3,"Beyond Good and Evil"),plan4,logic_llm)
    print("GENERATION TEST2:\n\n-------------------\n\n", result2)
    # print("GENERATION TEST3:\n\n-------------------\n\n", result3)
#     plan3 = """Step 1. Analyze the paragraphs.
# Step 2. Note that the text discusses various aspects of homogeneous and non-homogeneous substances, including their properties, extraction methods, and examples from nature and industry.
# Step 3. Realize that the text provides detailed information about the definition of a substance, its characteristics, and how to identify homogeneous and non-homogeneous substances.
# Step 4. Investigate potential topics for questions based on this analysis: examine what kinds of questions could be asked about these topics.
# Step 5. Consider if there are any connections between the different aspects discussed in the text that could lead to more complex or multi-part questions.
# Step 6. A potential question could be: "How do we identify a homogeneous substance?" This question tests understanding of the definition and characteristics of a homogeneous substance.
# Step 7. Realize another avenue of questions could explore the differences between homogeneous and non-homogeneous substances.
# Step 8. Devise possible questions that test this difference, such as "How do we distinguish between homogeneous and non-homogeneous substances?" This question tests understanding of the properties used to identify each type of substance.
# Step 9. Consider possible recall-related questions that test the reader's knowledge of specific details from the text.
# Step 10. A possible question might be: "What is an example of a homogeneous substance?" This question tests recall of the definition and examples given in the text.
# Step 11. I have brainstormed multiple areas from which questions can be asked, focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels)."""
#     plan3 = """Step 1. Analyze the paragraphs: briefly assess content, focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels).
# Step 2. Note the key elements: homogeneous substances, non-homogeneous substances, metals used in arts, properties of substances, extraction of substances from non-homogeneous substances, artificially produced non-homogeneous substances, and natural non-homogeneous substances.
# Step 3. Realize that the text provides detailed information about homogeneous and non-homogeneous substances. Possible question based on this fact: "What are some examples of homogeneous and non-homogeneous substances?" Answer: Homogeneous substances include metals used in arts, while non-homogeneous substances include natural and artificially produced ones.
# Step 4. Investigate the properties of these substances. Possible question based on this fact: "What are some properties of homogeneous substances?" Answer: Homogeneous substances exhibit similar properties in all their parts.
# Step 5. Consider questions testing understanding of how to distinguish between homogeneous and non-homogeneous substances. Possible question: "How can we tell if a substance is homogeneous or not?" Answer: By breaking up the substance, we obtain parts which resemble each other in their properties.
# Step 6. A potential question could be: "What are some ways to distinguish between homogeneous and non-homogeneous substances?" Answer: We can observe differences in color, texture, transparency, and composition of the substance.
# Step 7. Realize potential in exploring how these substances are made. Possible question: "How are homogeneous substances produced artificially?" Answer: They are prepared by mixing known proportions of sulphur, nitre, and charcoal.
# Step 8. Devise questions testing this process. Possible question: "What steps are involved in producing a non-homogeneous substance from a homogeneous one?" Answer: The homogeneous substance is extracted from the non-homogeneous substance through various methods.
# Step 9. Consider possible recall-related questions that test the reader's knowledge of individual parts of the text.
# Step 10. A possible question might be: "What are some examples of natural and artificially produced non-homogeneous substances?" Answer: Natural non-homogeneous substances include porphyries, rocks, plants, animals, blood, milk; while artificially produced ones include gunpowder.
# Step 11. I have brainstormed multiple areas from which questions can be asked."""
    # result3 = generate_question(text2,plan3,logic_llm)
    # print("GENERATION TEST3:\n\n-------------------\n\n", result3)
    
    
    
    # TODO change the function to include the full text in each question-answer tuple.
    
    
    
    
    
    
    
    
    # An example about how not to ask questions: if the text states fact X, but does not explain how X was established, do not ask a question "How do we know X". But instead you might consider asking how X relates to other facts in the paragraph, or how these facts together lead to a progression of ideas, "Explain why X, Y, and Z are related" for instance. Other good examples include "Compare and contrast X and Y", "Give an example of X".  Use Bloom's taxonomy, and focus on the cognitive levels of understanding, application, analysis, and synthesis of ideas.



"""# Response:
## Reasoning and thought process (being careful to only plan questions that are entirely based on the text provided):

Step 1: Analyze the paragraphs.
       - Realize that the text discusses the Water Cycle, focusing on processes like evaporation, condensation, and precipitation.

Step 2: Note that evaporation is a key process in the Water Cycle.
       - Recall that evaporation involves the transformation of water from liquid to gas.

Step 3: Consider the relationship between evaporation and condensation.
       - Recognize that condensation follows evaporation in the cycle, leading to cloud formation.

Step 4: Investigate how precipitation is related to condensation.
       - Understand that precipitation occurs when water droplets in clouds become too heavy and fall as rain, snow, etc.

Step 5: Devise possible questions that test the understanding of evaporation.
       - Questions could explore the conditions under which evaporation occurs or its role in the Water Cycle.

Step 6: Formulate questions regarding the process of condensation.
       - These questions might involve the transformation of gas to liquid and its significance in cloud formation.

Step 7: Develop questions about the precipitation process.
       - Consider questions on different forms of precipitation and their dependence on atmospheric conditions.

Step 8: Ensure all potential questions are directly related to the information in the text.
       - Confirm that each question tests the comprehension and memorization of the Water Cycle as described.

Step 9: I have brainstormed a sufficient number of possible, accurate questions based on the Water Cycle."""