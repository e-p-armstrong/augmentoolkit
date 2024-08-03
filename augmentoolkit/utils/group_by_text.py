from augmentoolkit.generation_functions.format_qadicts import format_qadicts
from augmentoolkit.generation_functions.identify_duplicates import identify_duplicates
import sys
import traceback

def group_by_text(dicts_list):
    # Dictionary to hold the groups with text as the key
    groups = {}

    # Iterate over each tuple in the list
    for dict in dicts_list:
        # If the text is not yet a key in the dictionary, add it with an empty list
        text = dict["paragraph"]
        if text not in groups:
            groups[text] = {
                "dict_list": [],
                "question_answer_pairs_string": "",
            }
                            

        # Append the current tuple to the appropriate list
        groups[text]['dict_list'].append(dict)
    
    # Iterate over the dictionary to create the question-answer pairs string
    for key, value in groups.items():
        try:
            # Create a list of dictionaries from the list of tuples
            dict_list = value["dict_list"]
            # Create a string of question-answer pairs
            question_answer_pairs_string = format_qadicts(dict_list)
            value["question_answer_pairs_string"] = question_answer_pairs_string
        except Exception as e:
            print(f"Error creating question-answer pairs string: {e}")
            traceback.print_exc(file=sys.stdout)
    # Return the values of the dictionary, which are the lists of tuples grouped by text; also remove duplicates
    return [ group for group in list(groups.values()) ]
