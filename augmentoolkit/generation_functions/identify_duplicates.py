from typing import List, Tuple
from .process_multiturn_functions import has_sequential_chars

# If you want to check for matching substrings anywhere, not just at start, use this code (untested)
# def identify_duplicates(tuples: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
#     # Create a dictionary to hold questions with the same first N characters
#     question_dict = {}

#     # Iterate through each tuple and categorize them by the first N characters of the question
#     for q_tuple in tuples:
#         question = q_tuple[0]
#         placed = False
#         for dict_q in question_dict.keys():
#             if has_sequential_chars(question,dict_q,N_CHARACTERS_SAME):
#                 question_dict[dict_q].append(q_tuple)
#                 placed = True
#                 break
#         if not placed:
#             question_dict[question] = [q_tuple] # if not found to be equivalent with anything, make it a dict entry so that things can be compared against it and added to its list

#     # Filter out prefixes that only have one question associated
#     matching_questions = [q for q_list in question_dict.values() if len(q_list) > 1 for q in q_list]

#     return matching_questions


def identify_duplicates(
    tuples: List[Tuple[str, str, str, str]],
) -> List[Tuple[str, str, str, str]]:
    # Create a dictionary to hold questions with the same first N characters
    question_dict = {}

    # Iterate through each tuple and categorize them by the first N characters of the question
    for q_tuple in tuples:
        question = q_tuple[0]
        # Get the first N characters of the question
        prefix = question[:15]
        # Add the tuple to the list of tuples with the same prefix
        if prefix in question_dict:
            question_dict[prefix].append(q_tuple)
        else:
            question_dict[prefix] = [q_tuple]

    matching_questions = [
        q for q_list in question_dict.values() if len(q_list) == 1 for q in q_list
    ]
    selected_from_duplicates = [
        q_list[0] for q_list in question_dict.values() if len(q_list) > 1
    ]

    return matching_questions + selected_from_duplicates


# There is no bug about this ignoring certain judgments and retrying; that's just the dissenting reasoning from the print statement


if __name__ == "__main__":
    sample_tuples = [
        ("What is your name?", "Alice", "12/12/2021", "ID1"),
        ("What is your quest?", "Bob", "12/12/2021", "ID2"),
        ("When is your birthday?", "Cindy", "12/12/2021", "ID3"),
        ("When is your birthday?", "Dan", "12/12/2021", "ID4"),
        ("When do you go to school?", "Eve", "12/12/2021", "ID5"),
    ]
    print(identify_duplicates(sample_tuples))
