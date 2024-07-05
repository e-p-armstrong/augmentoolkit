from augmentoolkit.generation_functions.identify_duplicates import identify_duplicates
import sys
import traceback

def group_by_text(tuples_list):
    # Dictionary to hold the groups with text as the key
    groups = {}

    # Iterate over each tuple in the list
    for question, answer, text, textname, q_group_id, idx, qnum in tuples_list:
        # If the text is not yet a key in the dictionary, add it with an empty list
        if text not in groups:
            groups[text] = []

        # Append the current tuple to the appropriate list
        groups[text].append((question, answer, text, textname, q_group_id, idx, qnum))
    # Return the values of the dictionary, which are the lists of tuples grouped by text; also remove duplicates
    return [ group for group in list(groups.values()) ]
