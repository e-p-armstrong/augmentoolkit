import re
from .question_plan_grammar import question_plan_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL

def generate_questions_plan(text,logic_llm):
    """
    Produce a list of questions based off of an input text. The min between (4, as many good questions as the text permits)
    
    Format: Question: [question]\n\n
    """
    # Determine which paragraphs are worthy of making questions from
    # Analyze-Realize-Create-Example loop
    cot_prompt = f"""# Input:
As an educational AI specializing in question generation, your task is to plan out high-quality, critical thinking questions strictly based on provided paragraphs. Focus on understanding, application, analysis, and synthesis of the text's ideas. Develop a comprehensive plan to generate questions that test comprehension and memorization of explicit content in the paragraphs. Aim for longer, challenging questions that can be answered with the given information.

Ensure your questions strictly adhere to the text's content, avoiding any external topics. The key step is to analyze the text thoroughly and brainstorm potential questions based on this analysis. Do not list final questions in this step.

Your response should be concise, focusing solely on the text's content to test knowledge without the text's reference. Maintain a step-by-step reasoning approach, ensuring your text analysis and brainstorming logically lead to comprehensive question planning. Avoid any deviation from the text's content.

# Input:
## Instruction:
Text details: A History of the United States, by Fred Garcia
Text to plan questions from: 
\"\"\"
The American Revolution, a pivotal event in history, was fueled by a series of causes and events that led to significant consequences for both America and Britain. One of the major catalysts was the Boston Massacre, an incident that drastically intensified revolutionary sentiments among the American colonists. The implementation of the Stamp Act further aggravated these sentiments, marking a critical point in the escalation of dissent. Similarly, the Boston Tea Party played a key role in heightening tensions, directly leading to the enactment of the Intolerable Acts. These acts, deemed unbearable by the colonists, were instrumental in the formation of the First Continental Congress, uniting the colonies in their fight for independence. A notable event during the revolution was the Battle of Saratoga, which had immediate effects such as boosting American morale and securing French support, pivotal in turning the tide of the war in favor of the American colonies.
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Identify Key Topics: Analyze the paragraph's content to identify key topics. Example: Causes, events, and consequences of the American Revolution.
Step 2. Determine Information-Rich Areas: Recognize parts of the text with detailed information. The text contains detailed information on various events of the American revolution, and the relationships between these things.
Step 3. Brainstorm and Develop Questions Testing Recall: Create questions that require recalling specific facts or details from the text. Example: "What were the immediate effects of the Battle of Saratoga on the American Revolution?"
Step 4. Devise Questions Exploring Relationships Between Facts: In this case, the text mostly covers events, so instead of looking for the similarities and differences between things, I will create questions that explore how one event led to another within the text. Example: "How did the Boston Massacre contribute to the revolutionary sentiment?"
Step 5. Create Questions Investigating If-Then Relationships (where since one thing is true, another follows): I will devise questions that examine how different events are interconnected. Example: "In what ways did the Intolerable Acts fuel the revolutionary cause?"

# Input:
## Instruction:
Text details: Introduction to Mathematics, by Jane Smith
Text to plan questions from: 
\"\"\"
In mathematics, the concept of a 'function' is fundamental, defining a relationship where each input is associated with exactly one output. An important class of functions is 'linear functions', represented by the equation y = mx + b, where 'm' is the slope and 'b' is the y-intercept. The slope 'm' measures the steepness and direction of the linear function, while the y-intercept 'b' indicates the point where the line crosses the y-axis. Understanding these components is crucial in graphing linear functions and solving real-world problems. Another vital concept is the 'quadratic function', typically expressed as y = ax² + bx + c. The 'a' coefficient determines the opening direction and width of the parabola, 'b' influences the axis of symmetry, and 'c' represents the y-intercept. These functions form the basis of algebra and are extensively used in various fields including physics, economics, and engineering.
\"\"\"

# Response:
## Reasoning and thought process:
Step 1. Identify Key Topics: The key topics in this paragraph are linear and quadratic functions in mathematics, their definitions, components, and applications.
Step 2. Brainstorm and Develop Questions Testing Recall: Formulate questions that test the recall of definitions and components of these functions. Example: "What does the 'm' in the linear function equation represent?"
Step 3. Devise Questions Exploring Relationships Between Components: Generate questions that explore the relationship between different parts of the equations. Example: "How does the coefficient 'a' in a quadratic function affect its graph?"
Step 4. Create Questions for Mastering Vital Concepts: Make questions about key concepts from the text. Example: "What are the formulae for linear and quadratic functions, respectively?"

# Input:
## Instruction:
Text details: Thus Spake Zaranthustra, by Friedrich Nietzsche
Text to plan questions from:
\"\"\"
When Zarathustra was thirty years old, he left his home and the lake of
his home, and went into the mountains. There he enjoyed his spirit and
solitude, and for ten years did not weary of it. But at last his heart
changed,—and rising one morning with the rosy dawn, he went before the
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
## Reasoning and thought process:
Step 1: Identify Key Themes: In this excerpt, we find themes of transformation, self-discovery, the value of wisdom, and the need for recognition of effort and wisdom.
Step 2: Brainstorm and Develop Questions Testing Recall: Develop questions asking for specific information mentioned in this text. For instance, "Finish the quote: I am weary of my wisdom, like..."
Step 3: Devise Questions Testing Comprehension and Belief in the Text's Opinions: Questions can be formed to check understanding and acceptance of the text's philosophy. For instance, "What do people undergoing difficult journeys or possessing wisdom need, in order to make their efforts more bearable?" Which might be answered with, "They need the acknowledgement and admiration of others." and then supported by lines such as "Thou great star! What would be thy happiness if thou hadst not those for whom thou shinest?" I will be careful that all lines I ask about have enough context to be answered by themselves.
Step 4. Create Questions Investigating Interpretations: Given the text's rich language, I will devise questions interpreting its meaning, while being careful not to explicitly mention the text in doing so. Example: "Why did Zaranthustra, in 'Thus Spake Zaranthustra', leave the mountains and become a man again?"

# Input:
## Instruction:
Text details: Great Construction Projects Throughout History, by Marco Gonzalez
Text to plan questions from:
\"\"\"
During the construction of the Panama Canal, a massive engineering feat completed in 1914, several challenges and achievements were noted. The canal, spanning approximately 50 miles, was designed to shorten the maritime route between the Atlantic and Pacific Oceans. Notably, the construction saw the use of innovative excavation techniques, with over 200 million cubic yards of earth removed. The project also faced significant health challenges, including combating malaria and yellow fever, which were overcome through extensive public health measures. The completion of the canal significantly impacted global trade, reducing the sea voyage from San Francisco to New York by around 8,000 miles.
\"\"\"

# Response:
## Reasoning and Thought Process:
Step 1. Identify Key Topics: The paragraph details specific aspects of the Panama Canal's construction, focusing on its challenges, innovations, and impacts. Topics include construction challenges, health issues, excavation techniques, and the canal's impact on global trade.
Step 2. Brainstorm and Develop Questions Testing Recall: Questions can be formed to recall factual data from the text. Example: "How much earth was excavated during the construction of the Panama Canal?"
Step 3. Devise Questions Exploring Cause and Effect Relationships: This involves creating questions that explore how certain challenges led to specific solutions and impacts. Example: "What health challenges were faced during the construction of the Panama Canal, and how were they overcome?"
Step 4. Create Questions Investigating Quantitative Values: Given the text's focus on concrete numbers, I will devise questions that require analyzing these figures. Example: "By how many miles did the Panama Canal reduce the sea voyage from San Francisco to New York, and what does this imply about its impact on global trade?"

# Input:
## Instruction:
Text details: {text[1]}
Text to plan questions from:
\"\"\"
{text[0]}
\"\"\"

# Response:
## Reasoning and thought process (being careful to only plan questions that are entirely based on the text provided):
"""
    # print("DEBUG\n\n" + decision_prompt)
    completion = logic_llm(cot_prompt, max_tokens=4000, stop=["</s>"], echo=True, grammar=question_plan_grammar, temperature=0.2)["choices"][0]["text"]
    # EVERYTHING BELOW HERE IS TODO
    # print("DEBUG\n\n")
    print("COMPLETION:\n\n----------------------")
    print(completion)
    print("\n------------------")
    
    # Extract plan
    response_pattern = re.compile(r"Reasoning and thought process \(being careful to only plan questions that are entirely based on the text provided\):\n(.+)",re.IGNORECASE | re.DOTALL)
    generation = response_pattern.search(completion).group(1)
    print("GENERATION:\n\n-------------------\n\n", generation)
    
    return generation


if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_ctx=4096,n_gpu_layers=1000) # load the logical LLM and offload everything
    text = """The story of our world is a story that is still very imperfectly known. A couple of hundred years ago men possessed the history of little more than the last three thousand years. What happened before that time was a matter of legend and speculation.  Over a large part of the civilized world it was believed and taught that the world had been created suddenly in 4004 B.C., though authorities differed as to whether this had occurred in the spring or autumn of that year. This fantastically precise misconception was based upon a too literal interpretation of the Hebrew Bible, and upon rather arbitrary theological assumptions connected therewith.  Such ideas have long since been abandoned by religious teachers, and it is universally recognized that the universe in which we live has to all appearances existed for an enormous period of time and possibly for endless time.  Of course there may be deception in these appearances, as a room may be made to seem endless by putting mirrors facing each other at either end. But that the universe in which we live has existed only for six or seven thousand years may be regarded as an altogether exploded idea.

The earth, as everybody knows nowadays, is a spheroid, a sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles.  Its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years, but before that time it was supposed to be flat, and various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets.  We know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameter) every twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles."""
    print("Begin HGWELLS test")
    # result = generate_question_plan(text,logic_llm)
    ## TODO a wider variety of tests from different texts

    # Chemistry: a harder science
    print("Begin MENDELEEV TEST")
    text2 = ("""A substance or material is that which occupies space and has weight; that is, which presents a mass attracted by the earth and by other masses of material, and of which the _objects_ of nature are composed, and by means of which the motions and _phenomena_ of nature are accomplished. It is easy to discover by examining and investigating, by various methods, the objects met with in nature and in the arts, that some of them are homogeneous, whilst others are composed of a mixture of several homogeneous substances. This is most clearly apparent in solid substances. The metals used in the arts (for example, gold, iron, copper) must be homogeneous, otherwise they are brittle and unfit for many purposes. Homogeneous matter exhibits similar properties in all its parts. By breaking up a homogeneous substance we obtain parts which, although different in form, resemble each other in their properties. Glass, pure sugar, marble, &c., are examples of homogeneous substances. Examples of non-homogeneous substances are, however, much more frequent in nature and the arts. Thus the majority of the rocks are not homogeneous. In porphyries bright pieces of a mineral called 'orthoclase' are often seen interspersed amongst the dark mass of the rock. In ordinary red granite it is easy to distinguish large pieces of orthoclase mixed with dark semi-transparent quartz and flexible laminæ of mica. Similarly, plants and animals are non-homogeneous. Thus, leaves are composed of a skin, fibre, pulp, sap, and a green colouring matter. As an example of those non-homogeneous substances which are produced artificially, gunpowder may be cited, which is prepared by mixing together known proportions of sulphur, nitre, and charcoal. Many liquids, also, are not homogeneous, as may be observed by the aid of the microscope, when drops of blood are seen to consist of a colourless liquid in which red corpuscles, invisible to the naked eye owing to their small size, are floating about.""","Principles of Chemistry, by Demitry Mendeleev")
    result2 = generate_questions_plan(text2,logic_llm)
    
    text3 = ("""I went down yesterday to the Piraeus with Glaucon the son of Ariston,
that I might offer up my prayers to the goddess (Bendis, the Thracian
Artemis.); and also because I wanted to see in what manner they would
celebrate the festival, which was a new thing. I was delighted with the
procession of the inhabitants; but that of the Thracians was equally,
if not more, beautiful. When we had finished our prayers and viewed the
spectacle, we turned in the direction of the city; and at that instant
Polemarchus the son of Cephalus chanced to catch sight of us from a
distance as we were starting on our way home, and told his servant to
run and bid us wait for him. The servant took hold of me by the cloak
behind, and said: Polemarchus desires you to wait.

I turned round, and asked him where his master was.

There he is, said the youth, coming after you, if you will only wait.

Certainly we will, said Glaucon; and in a few minutes Polemarchus
appeared, and with him Adeimantus, Glaucon’s brother, Niceratus the son
of Nicias, and several others who had been at the procession.

Polemarchus said to me: I perceive, Socrates, that you and your
companion are already on your way to the city.

You are not far wrong, I said.

But do you see, he rejoined, how many we are?

Of course.

And are you stronger than all these? for if not, you will have to
remain where you are.

May there not be the alternative, I said, that we may persuade you to
let us go?

But can you persuade us, if we refuse to listen to you? he said.

Certainly not, replied Glaucon.

Then we are not going to listen; of that you may be assured.

Adeimantus added: Has no one told you of the torch-race on horseback in
honour of the goddess which will take place in the evening?

With horses! I replied: That is a novelty. Will horsemen carry torches
and pass them one to another during the race?

Yes, said Polemarchus, and not only so, but a festival will be
celebrated at night, which you certainly ought to see. Let us rise soon
after supper and see this festival; there will be a gathering of young
men, and we will have a good talk. Stay then, and do not be perverse.

Glaucon said: I suppose, since you insist, that we must.

Very good, I replied.

Accordingly we went with Polemarchus to his house; and there we found
his brothers Lysias and Euthydemus, and with them Thrasymachus the
Chalcedonian, Charmantides the Paeanian, and Cleitophon the son of
Aristonymus. There too was Cephalus the father of Polemarchus, whom I
had not seen for a long time, and I thought him very much aged. He was
seated on a cushioned chair, and had a garland on his head, for he had
been sacrificing in the court; and there were some other chairs in the
room arranged in a semicircle, upon which we sat down by him. He
saluted me eagerly, and then he said:—

You don’t come to see me, Socrates, as often as you ought: If I were
still able to go and see you I would not ask you to come to me. But at
my age I can hardly get to the city, and therefore you should come
oftener to the Piraeus. For let me tell you, that the more the
pleasures of the body fade away, the greater to me is the pleasure and
charm of conversation. Do not then deny my request, but make our house
your resort and keep company with these young men; we are old friends,
and you will be quite at home with us.

I replied: There is nothing which for my part I like better, Cephalus,
than conversing with aged men; for I regard them as travellers who have
gone a journey which I too may have to go, and of whom I ought to
enquire, whether the way is smooth and easy, or rugged and difficult.
And this is a question which I should like to ask of you who have
arrived at that time which the poets call the ‘threshold of old age’—Is
life harder towards the end, or what report do you give of it?""","The Republic, by Plato")
    print("Begin PLATO test")
    # result3 = generate_questions_plan(text3,logic_llm)
    
    print("Begin KANT test")
    text4 = ("""I. Of the difference between Pure and Empirical Knowledge

That all our knowledge begins with experience there can be no doubt.
For how is it possible that the faculty of cognition should be awakened
into exercise otherwise than by means of objects which affect our
senses, and partly of themselves produce representations, partly rouse
our powers of understanding into activity, to compare to connect, or to
separate these, and so to convert the raw material of our sensuous
impressions into a knowledge of objects, which is called experience? In
respect of time, therefore, no knowledge of ours is antecedent to
experience, but begins with it.

But, though all our knowledge begins with experience, it by no means
follows that all arises out of experience. For, on the contrary, it is
quite possible that our empirical knowledge is a compound of that which
we receive through impressions, and that which the faculty of cognition
supplies from itself (sensuous impressions giving merely the occasion),
an addition which we cannot distinguish from the original element given
by sense, till long practice has made us attentive to, and skilful in
separating it. It is, therefore, a question which requires close
investigation, and not to be answered at first sight, whether there
exists a knowledge altogether independent of experience, and even of
all sensuous impressions? Knowledge of this kind is called à priori, in
contradistinction to empirical knowledge, which has its sources à
posteriori, that is, in experience.

But the expression, “à priori,” is not as yet definite enough
adequately to indicate the whole meaning of the question above started.
For, in speaking of knowledge which has its sources in experience, we
are wont to say, that this or that may be known à priori, because we do
not derive this knowledge immediately from experience, but from a
general rule, which, however, we have itself borrowed from experience.
Thus, if a man undermined his house, we say, “he might know à priori
that it would have fallen;” that is, he needed not to have waited for
the experience that it did actually fall. But still, à priori, he could
not know even this much. For, that bodies are heavy, and, consequently,
that they fall when their supports are taken away, must have been known
to him previously, by means of experience.""", "The Critique of Pure Reason, by Immanuel Kant")
    # result4 = generate_questions_plan(text4,logic_llm)
    
    # This text is very hard to parse, maybe not a good test
#     print("Begin Beyond Good and Evil test")
#     text5 = ("""The Will to Truth, which is to tempt us to many a hazardous
# enterprise, the famous Truthfulness of which all philosophers have
# hitherto spoken with respect, what questions has this Will to Truth not
# laid before us! What strange, perplexing, questionable questions! It is
# already a long story; yet it seems as if it were hardly commenced. Is
# it any wonder if we at last grow distrustful, lose patience, and turn
# impatiently away? That this Sphinx teaches us at last to ask questions
# ourselves? WHO is it really that puts questions to us here? WHAT really
# is this "Will to Truth" in us? In fact we made a long halt at the
# question as to the origin of this Will--until at last we came to an
# absolute standstill before a yet more fundamental question. We inquired
# about the VALUE of this Will. Granted that we want the truth: WHY NOT
# RATHER untruth? And uncertainty? Even ignorance? The problem of the
# value of truth presented itself before us--or was it we who presented
# ourselves before the problem? Which of us is the Oedipus here? Which
# the Sphinx? It would seem to be a rendezvous of questions and notes of
# interrogation. And could it be believed that it at last seems to us as
# if the problem had never been propounded before, as if we were the first
# to discern it, get a sight of it, and RISK RAISING it? For there is risk
# in raising it, perhaps there is no greater risk.""", "Beyond Good and Evil")
#     result4 = generate_questions_plan(text5,logic_llm)

















# # old one

# f"""[INST] <<SYS>> You are an expert educational AI that, given a paragraph or two from a text, will narrow down what kind of high-quality educational questions could be asked based on the paragraphs, and *only* based on the paragraphs. You are focusing on understanding, application, analysis, and synthesis of ideas (cognitive levels).<</SYS>> For now, your goal is to just write a good, comprehensive plan for generating questions that require critical thinking to solve; what aspects of the provided paragraphs they might cover, etc. The questions should ONLY cover material that explictly appears in the provided paragraphs. The topics you think of should lean towards longer, more difficult questions, that require some thought to solve — but which can still be solved if you know the information in the paragraphs. Essentially: the question will test comprehension and memorization of real information that would be worthy to teach. Your task includes analyzing the text, thinking through and brainstorming which questions you will make and why. 

# Text to make questions from: \"\"\"{text}\"\"\"

# You should aim to plan out 4 questions, but if the text is too small or information-sparse for that many, you are allowed to write fewer. BE CAREFUL NOT TO ASK QUESTIONS ABOUT THINGS THAT DO NOT APPEAR IN THE TEXT; your reasoning step should only deal with things that are EXPLICITLY stated in the text. The primary goal of the reasoning step is to analyze the provided text and brainstorm possible questions from the paragraph given that analysis. Do not write out a final list of questions in this step.

# An example about how not to ask questions: if the text states fact X, but does not explain how X was established, do not ask a question "How do we know X". But instead you might consider asking how X relates to other facts in the paragraph, or how these facts together lead to a progression of ideas, "Explain how X, Y, and Z are related" for instance. Use Bloom's taxonomy, and focus on the cognitive levels of understanding, application, analysis, and synthesis of ideas.

# Keep your answer concise. You will not mention the text in any topics you think of, since the questions you generate are intended to test people's knowledge of the information — when given the questions, they will not have the text on-hand. Remember to stay completely within the content of the text, and to NOT STRAY IN THE SLIGHTEST.

# For the first part of your reasoning step, you might want to make a few learning objectives based on the paragraphs provided. [/INST]

# # Reasoning and thought process (being careful to only plan questions that are entirely based on the text provided):
# First, let's will analyze the text to determine what kinds of high-level questions I can ask that will test the content in these paragraphs (being careful to avoid mentioning the paragraphs explicitly in any questions, and being SURE to only ask about things that the paragraphs talk about). Now, immediately I can see that the following passage might be a good basis for a question: """