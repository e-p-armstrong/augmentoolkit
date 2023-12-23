import re

def determine_perspective(character, dialogue_line):
    # Check for specific third person indicators at the start of narration
    third_person_indicators = [r'\*She', r'\*Her', r'\*{}'.format(character)] # GPT almost always makes mistakes in the same way, which is a blessing.
    
    for indicator in third_person_indicators:
        if re.search(indicator, dialogue_line, re.IGNORECASE):
            return "third person"
    
    # If none of the specific third person indicators are present, default to first person
    return "first person"

# Test
if __name__ == "main":
    print(determine_perspective("Mayuri", 'Mayuri: *She tilts her head, her eyes squinting as she tries to comprehend my words.* "But that’s too hard to remember."'))
    print(determine_perspective("Kurisu", 'Kurisu: *I let out a sigh, my eyes softening as I watch Itaru.* "I guess that means the past hasn’t changed."'))
    