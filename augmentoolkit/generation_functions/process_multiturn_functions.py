import re


def has_sequential_chars(string1, string2, n):
    """
    Check if any n sequential characters from string1 appear in string2.

    Args:
    string1 (str): The first string to check.
    string2 (str): The second string in which to look for sequences.
    n (int): The length of the sequence to check.

    Returns:
    bool: True if any n sequential characters from string1 are found in string2, False otherwise.
    """

    # Check if n is larger than the length of string1.
    if n > len(string1):
        return False, ""

    # Iterate over string1 and check for each n-length substring in string2
    comparison_string = ""
    for i in range(len(string1) - n + 1):
        comparison_string = string1[i : i + n]
        if comparison_string in string2:
            return True, comparison_string

    return False, comparison_string


def extract_conversation(conversation):
    """
    Extracts conversation from a string and returns it as a list of tuples.

    Parameters:
    conversation (str): A string representing the conversation.

    Returns:
    list of tuples: Each tuple contains the character's name and their message.
    """
    lines = conversation.strip().split("\n")
    dialogues = []
    current_speaker = None
    current_message = ""

    for line in lines:
        line = line.strip()
        if line in ["**AI Assistant:**", "**User:**"]:
            if current_speaker:
                dialogues.append((current_speaker, current_message.strip()))
                current_message = ""
            current_speaker = line[2:-2].strip()
        else:
            if current_speaker:
                current_message += line + "\n"

    if current_speaker:
        dialogues.append((current_speaker, current_message.strip()))

    return dialogues


def compare_answers_with_qatuples(dialogues, qatuples, n):
    """
    Compares each answer in dialogues with the corresponding answer from qatuples.

    Parameters:
    dialogues (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n (int): Number of sequential characters to check.

    Returns:
    bool: True if all answers match the corresponding answers in qatuples, False otherwise.
    """
    for i in range(1, len(dialogues), 2):  # Answers are at odd indices, starting from 1
        if (i - 1) // 2 >= len(qatuples):  # at this point we've reached added stuff that doesn't have a corresponding qatuple
            break
        sequential, comp = has_sequential_chars(qatuples[(i - 1) // 2][1], dialogues[i][1], n)
        # print(sequential)
        # print(n)
        if not sequential:
            print(
                f"Answer {(i + 1) // 2}: {dialogues[i][1]} does not match the corresponding answer in qatuples: {qatuples[(i - 1) // 2][1]}, {comp}"
            )
            return False
    return True

# def check_repeated_answer(dialogues, qatuples):
#     # Get the length of the dialogues
#     conv_length = len(dialogues)

#     # Loop through even indices starting from 2 (first answer is at index 2)
#     for i in range(2, conv_length, 2):
#         current_answer = dialogues[i][1][:n_characters_same]
#         next_answer_index = i + 2

#         if next_answer_index < conv_length:
#             next_answer = dialogues[next_answer_index][1][:n_characters_same]
#             if current_answer == next_answer:
#                 return False
#     return True


def check_conversation_length(conv, qatuples):
    """Checks the length of the conversation"""
    # Dialogues with answers should be at even indices that are not 0
    # qatuples are of the format (question, answer,source_text,name_of_text) -- only the first two are used here

    # Get the length of the dialogues
    conv_length = len(conv)

    target_length = len(qatuples) * 2
    if (
        conv_length < target_length
    ):  # we can have more messages since the AI might add some stuff at the end to wrap up the scene
        return False
    else:
        return True

def check_each_question_contains_q_from_tuples(conv, qatuples, n):
    """
    Ensures that each question contains at least n sequential characters from the corresponding question in qatuples.
    If the first question fails this check, return None for special handling.

    Parameters:
    conv (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n (int): Number of sequential characters to check.

    Returns:
    bool or None: True if all questions pass the check, False if any fail, None if the first question fails.
    """
    for i in range(0, len(conv), 2):  # Questions are at even indices, starting from 0
        if i // 2 < len(qatuples):  # Ensure we only check questions that have corresponding qatuples
            question_from_conv = conv[i][1]
            question_from_tuples = qatuples[i // 2][0]
            # print(question_from_tuples, question_from_conv)
            sequential, _ = has_sequential_chars(question_from_tuples, question_from_conv, n)
            if not sequential:
                if i == 0:
                    return None  # Special handling for the first question
                else:
                    return False
    return True


def check_for_unintended_repeated_quotes(dialogues, qatuples, n_characters_shared):
    """
    Checks if answers in the conversation inadvertently use a long quote from another QA pair.

    Args:
    dialogues (list): List of tuples containing the dialogues.
    qatuples (list): List of tuples containing questions and answers.
    n_characters_shared (int): Number of sequential characters to check for repetition.

    Returns:
    bool: True if no unintended repeated quotes are found, False otherwise.
    """

    # Extract only the answers from the QA tuples for comparison
    qa_answers = [qa[1] for qa in qatuples]

    for i in range(
        2, len(dialogues), 2
    ):  # Answers are at even indices, starting from 2
        # Skip if there's no corresponding QA tuple
        if int(i / 2) - 1 >= len(qatuples):
            break

        dialogue_answer = dialogues[i][1]
        corresponding_qa_answer = qatuples[int(i / 2) - 1][1]

        # Check for each answer in the QA tuples
        for idx, qa_answer in enumerate(qa_answers):
            # Skip the comparison for the current QA pair itself
            if qa_answer == corresponding_qa_answer:
                continue

            # Check if the dialogue answer contains a long quote from another QA answer
            sequential, comp_string = has_sequential_chars(
                qa_answer, dialogue_answer, n_characters_shared
            )
            if sequential:
                if comp_string in corresponding_qa_answer:
                    continue  # This is a quote from the corresponding answer, so it's fine
                else:
                    # Found an unintended repeated quote
                    return False
    return True


def call_all_processors(multiturn_conversation, qatuples):
    convs_split = extract_conversation(multiturn_conversation)

    # Check if answers in dialogues match corresponding answers in qatuples
    if not compare_answers_with_qatuples(convs_split, qatuples, 15):
        print("Answers in dialogues do not match corresponding answers in qatuples.")
        return False

    # Check the conversation length
    if not check_conversation_length(convs_split, qatuples):
        print("Conversation is too short! Validation failed!")
        print(convs_split)
        return False

    # Check for unintended repeated quotes
    if not check_for_unintended_repeated_quotes(convs_split, qatuples, 100):
        print("Conversation contains unintended repeated quotes. Validation failed!")
        return False

    # Check each question contains a part of the question from tuples
    result = check_each_question_contains_q_from_tuples(convs_split, qatuples, 15)
    if result is None:
        print(
            "First question does not contain a part of the question from tuples. Validation failed!"
        )
        return None
    elif not result:
        print(
            "Each question does not contain a part of the question from tuples. Validation failed!"
        )
        return False

    # If all checks pass
    return True


if __name__ == "__main__":
    # Test cases for has_sequential_chars
    print("Testing has_sequential_chars:")
    print(has_sequential_chars("hello", "worldhello", 3))  #
    print("Expected True")
    print(has_sequential_chars("abc", "defghijkl", 2))  # Expected False
    print("Expected False")
    print(has_sequential_chars("", "empty", 1))  # Expected False (empty string1)
    print("Expected False")
    print(
        has_sequential_chars("longstring", "short", 5)
    )  # Expected False (n is longer than string2)
    print("Expected False")
    print(
        has_sequential_chars("overlap", "laptopp", 3)
    )  # Expected True (partial overlap)
    print("Expected True")

    # Test cases for extract_conversation
    print("\nTesting extract_conversation:")
    test_conversation1 = "Charname1: Hello\nCharname2: Hi\nCharname3: How are you?"
    print(
        extract_conversation(test_conversation1)
    )  # Expected [('Charname1', 'Hello'), ('Charname2', 'Hi'), ('Charname3', 'How are you?')]
    print(
        "Expected [('Charname1', 'Hello'), ('Charname2', 'Hi'), ('Charname3', 'How are you?')]"
    )
    test_conversation2 = "No colons here"
    print(extract_conversation(test_conversation2))  # Expected []
    print("Expected []")
    test_conversation3 = ""
    print(extract_conversation(test_conversation3))  # Expected []
    print("Expected []")

    # Test cases for compare_answers_with_qatuples
    print("\nTesting compare_answers_with_qatuples:")
    dialogues1 = [
        ("Charname1", "Hello"),
        ("Charname2", "Hi how are you"),
        "Totally Fantastic and Amazing!",
    ]
    qatuples1 = [("How are you?", "Fine")]
    print(compare_answers_with_qatuples(dialogues1, qatuples1, 2))  # Expected False
    print("Expected False")
    dialogues2 = [
        ("Charname1", "Hello"),
        ("Charname2", "Hi how are you"),
        ("Charname1", "Mostly Fine I think, yeah"),
    ]
    print(compare_answers_with_qatuples(dialogues2, qatuples1, 2))  # Expected True
    print("Expected True")
    dialogues3 = []
    qatuples2 = []
    print(
        compare_answers_with_qatuples(dialogues3, qatuples2, 2)
    )  # Expected True (both empty)
    print("Expected True (both empty)")

    # Test cases for check_for_repeated_dialogue_answers
    print("\nTesting check_for_repeated_dialogue_answers:")
    qatuples_repeated_answers = [("How are you?", "Fine, thank you for asking!")]
    dialogues4 = [
        ("Charname1", "Hello"),
        ("Charname2", "How are you?"),
        ("Charname1", "Fine, thank you for asking!"),
    ]
    print(
        check_for_repeated_dialogue_answers(dialogues4, qatuples_repeated_answers, 2)
    )  # Expected True (no repetition)
    print("Expected True (no repetition)")
    dialogues5 = [
        ("Charname1", "Hello"),
        ("Charname2", "How are you?"),
        (
            "Charname1",
            "Fine, thank you for asking! It's nice today, after all, so I'm Fine, thank you for asking!",
        ),
    ]
    print(
        check_for_repeated_dialogue_answers(dialogues5, qatuples_repeated_answers, 2)
    )  # Expected False (repetition)
    print("Expected False (repetition)")

    # Test cases for check_repeated_answer
    # print("\nTesting check_repeated_answer:")
    # dialogues6 = [("Charname1", "Question"), ("Charname2", "Answer1"), ("Charname3", "Question"), ("Charname4", "Answer1")]
    # print(check_repeated_answer(dialogues6))  # Expected False (repeated answers)
    # dialogues7 = [("Charname1", "Question"), ("Charname2", "Answer1"), ("Charname3", "Question"), ("Charname4", "Answer2")]
    # print(check_repeated_answer(dialogues7))  # Expected True (different answers)

    # Test cases for check_conversation_length
    print("\nTesting check_conversation_length:")
    conv1 = [("Charname1", "Hello"), ("Charname2", "Hi, How are you?")]
    print(
        check_conversation_length(conv1, qatuples1)
    )  # Expected False (conversation too short)
    print("Expected False (conversation too short)")
    conv2 = [("Charname1", "Hello"), ("Charname2", "Hi"), ("Charname3", "How are you?")]
    print(check_conversation_length(conv2, qatuples1))  # Expected True (correct length)
    print("Expected True (correct length)")

    # Test cases for check_conversation_for_text_from_examples (commented out as implementation is assumed elsewhere)
    # print("\nTesting check_conversation_for_text_from_examples:")
    # conv3 = "This conversation contains lipstick-colored lips and a coquettishly tilting head."
    # print(check_conversation_for_text_from_examples(conv3))  # Expected False (contains example texts)

    # Test cases for check_each_question_contains_q_from_tuples
    print("\nTesting check_each_question_contains_q_from_tuples:")
    conv4 = [
        ("Charname2", "Hiya~!"),
        ("Charname1", "What's your favorite color?"),
        ("Charname2", "I'm Fine, thank you very much!"),
    ]
    print(check_each_question_contains_q_from_tuples(conv4, qatuples1, 6))
    print("Expected None (no matching question, first Q)")

    conv45 = [
        ("Charname2", "Hiya~!"),
        ("Charname1", "How are you?"),
        ("Charname2", "I'm Fine, thank you very much!"),
        ("Charname1", "What is the airspeed velocity of an unladen swallow?"),
        ("Charname2", "Black, like my soul."),
    ]
    qatuples3 = [
        ("How are you?", "I'm Fine, thank you very much!"),
        ("What's your favorite color?", "Black, like my soul."),
    ]
    print(check_each_question_contains_q_from_tuples(conv45, qatuples3, 6))
    print("Expected False (no matching question, second Q)")

    conv5 = [
        ("Charname1", "Hiya~!"),
        ("Charname2", "How are you?"),
        ("Charname2", "I'm Fine, thank you very much!"),
        ("Charname1", "What's your favorite color?"),
        ("Charname2", "Black, like my soul."),
    ]
    print(check_each_question_contains_q_from_tuples(conv5, qatuples1 + [], 6))  #
    print("Expected True (question contains part of qatuple question)")

    # Test cases for check_for_unintended_repeated_quotes
    print("\nTesting check_for_unintended_repeated_quotes:")
    # Creating a set of dialogues and qatuples where there is an unintended repeated quote
    qatuples_shared = [
        ("What is your favorite book?", "I love reading The Hobbit."),
        (
            "Tell me about a recent happy moment.",
            "My friends threw me a surprise party!",
        ),
    ]
    dialogues_shared1 = [
        ("Charname1", "Hello"),
        ("Charname2", "What is your favorite book?"),
        ("Charname1", "I love reading The Hobbit."),
        ("Charname2", "Tell me about a recent happy moment."),
        (
            "Charname1",
            "My friends threw me a surprise party! It felt just like I was in The Hobbit.",
        ),
    ]
    print(
        check_for_unintended_repeated_quotes(dialogues_shared1, qatuples_shared, 10)
    )  # Expected False (repeated long quote from another answer)
    print("Expected False (repeated long quote from another answer)")

    # Creating a set of dialogues and qatuples where there are no unintended repeated quotes
    dialogues_shared2 = [
        ("Charname1", "Hello"),
        ("Charname2", "What is your favorite book?"),
        ("Charname1", "I absolutely adore The Lord of the Rings."),
        ("Charname2", "Tell me about a recent happy moment."),
        ("Charname1", "I had a great time at the beach last weekend!"),
    ]
    print(
        check_for_unintended_repeated_quotes(dialogues_shared2, qatuples_shared, 10)
    )  # Expected True (no repeated long quotes)
    print("Expected True (no repeated long quotes)")

    # Test cases for call_all_processors
    print("\nTesting call_all_processors:")
    complete_conversation = """
    Charname1: Hello
    Charname2: How are you doing today?
    Charname1: I'm fine, thank you very much!
    Charname2: What's the weather like?
    Charname1: It's sunny and warm. I don't like sand. It's coarse and rough and irritating and it gets everywhere.
    Foo: Bar
    Baz: Quux
    """
    qatuples_complete = [
        ("How are you doing today?", "I'm fine, thank you very much!"),
        (
            "What's the weather like?",
            "It's sunny and warm. I don't like sand. It's coarse and rough and irritating and it gets everywhere.",
        ),
    ]
    print(call_all_processors(complete_conversation, qatuples_complete))  #
    print("Expected True (all checks pass)")
    incomplete_conversation = """
    Charname1: How's it going?
    Charname2: Good.
    Charname1: Any plans?
    Charname2: None.
    Foo: Bar
    Baz: Quux
    """
    print(call_all_processors(incomplete_conversation, qatuples_complete))  #
    print("Expected False (checks fail)")
