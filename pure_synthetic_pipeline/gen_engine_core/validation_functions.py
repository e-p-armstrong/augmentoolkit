import re
from collections import defaultdict, deque
import traceback
from gen_engine_core.utillity_functions import count_tokens

def find_frequent_substrings(text, min_length, min_occurrences, cluster_threshold):
    def update_counts(substring, index):
        # Update positions and remove indices that are out of the current cluster scope
        while substring_positions[substring] and index - substring_positions[substring][0] > cluster_threshold:
            substring_positions[substring].popleft()
            active_substrings[substring] -= 1
            if not active_substrings[substring]:  # Clean up if no occurrences are within the threshold
                del active_substrings[substring]
                del substring_positions[substring]

        # Add new index and update count
        substring_positions[substring].append(index)
        active_substrings[substring] += 1

        return active_substrings[substring]

    active_substrings = defaultdict(int)  # Counts of substrings currently within the cluster threshold
    substring_positions = defaultdict(deque)  # Positions of each substring's occurrences
    max_repetitions = 0
    most_repeated_substring = ""

    for start in range(len(text) - min_length + 1):
        for end in range(start + min_length, min(start + min_length + cluster_threshold, len(text)) + 1):
            substring = text[start:end]
            count = update_counts(substring, start)

            # Check if this substring is the most repeated within the constraints
            if count >= min_occurrences and (count > max_repetitions or (count == max_repetitions and len(substring) > len(most_repeated_substring))):
                max_repetitions = count
                most_repeated_substring = substring

    print(f"The highest number of repetitions is {max_repetitions}")
    print(f"The substring that repeated the most is '{most_repeated_substring}'")
    
    return max_repetitions, most_repeated_substring

# find_frequent_substrings(text, min_length, min_occurrences)

def find_repetitions(text, min_length):
    pattern = r'(.{' + str(min_length) + r',}?)\1+'
    repetitions = re.finditer(pattern, text)
    
    # for rep in repetitions:
    #     print(f"Repetition found: '{rep.group(1)}' (repeated {len(rep.group()) // len(rep.group(1))} times)")

    return repetitions


## Post-generation validation and retry abstraction
async def validate_generation(gen_func=None, validation_functions=[], retries=1, gen_func_args=[]): # higher-order function that takes a list of validation functions and a generation function, and checks if the generation function's output passes all the validation functions; if not, it retries up to a certain number of times. If it fails it returns None, otherwise it returns the output of the generation function.
    """
    The interface for validation functions compatible with validate_generation is as follows:
    they should take a single input: the output of the generation function
    they should return false if that input does not pass validation and True if it does
    """
    times_tried = 0
    while times_tried <= retries:
        try:
            response = await gen_func(*gen_func_args)
            success = True
            for validation_function in validation_functions:
                if not validation_function(response):
                    success = False
                    break
            if success:
                return response
            else:
                times_tried += 1
        except Exception as e:
            times_tried += 1
            print(f"Error in Generation Step: {e}")
            traceback.print_exc()
    raise Exception("VALIDATION FAILED TOO MANY TIMES -- CUTTING LOSSES AND SKIPPING THIS CHUNK\n\n\n")

# Helpers for said abstraction
def validate_length_callback(length): # returns a function that checks if a string is a certain length
    def inner(string):
        return count_tokens(string) <= length
    return inner

def validate_consecutive_repetition_callback(min_length): # returns a function that checks if a string has a repetition of a certain length
    def inner(string):
        return len(list(find_repetitions(string, min_length))) == 0
    return inner

def validate_repetition_callback(min_length, num_repetitions_allowed, cluster_threshold): # returns a function that checks if a string has a repetition of a certain length
    def inner(string):
        count, substr = find_frequent_substrings(string, min_length, num_repetitions_allowed, cluster_threshold)
        res = count <= num_repetitions_allowed
        if not res:
            print(f"\n\nRepetition found: '{substr}' (repeated {count} times)\n\n")
        return res
    return inner

def validate_not_none(input):
    # print(input)
    if input == None:
        return False
    else:
        return True