def strip_steps(instruction_text):
    """
    This function takes a string containing step-by-step instructions and removes the "Step N." prefix from each line.
    
    Parameters:
    instruction_text (str): A string with each step in the format "Step N. Instruction", separated by newlines

    Returns:
    str: A single string with the steps stripped, joined by newlines.
    """
    instructions = instruction_text.split('\n')
    stripped_instructions = []
    for line in instructions:
        # Check if line starts with 'Step' and followed by a number and period
        if line.strip().startswith('Step') and '.' in line:
            # Find the index of the first period
            period_index = line.find('.')
            # Extract the text after the period (and optional space)
            text_after_period = line[period_index+1:].lstrip()
            stripped_instructions.append(text_after_period)
        else:
            stripped_instructions.append(line)
    
    return '\n'.join(stripped_instructions)

if __name__ == "__main__":
    # Example usage with a multi-line string
    example_instructions = """
Step 1. Analyze the Text: focus on the details provided about the beliefs ancient people had about the shape and movement of our world.
Step 2. Identify Key Points: look for important concepts or ideas mentioned in the text.
    """

    result = strip_steps(example_instructions)
    print(result)
    
    # Example with no space after the period
    example_non_instructions = """
    Step1. This is a lovely
    normal
    paragraph
    Step2.Another test line without space after period
    """

    result2 = strip_steps(example_non_instructions)
    
    print(result2)

    example_3 = """
    Step 1. Analyze the Text: focus on the details provided about the history of the earth's shape.
Step 2. Understand the Question: the question's focus is on what is known about the history of the earth's shape.
Step 3. Compare the First Part of the Answer with the Text: check if the text supports the claim that the earth is a spheroid, or sphere slightly compressed, orange fashion, with a diameter of nearly 8,000 miles. It does, so this part is accurate. Then, check if the text supports the claim that its spherical shape has been known at least to a limited number of intelligent people for nearly 2,500 years. The text confirms this, so this part is accurate. Check if the text supports the claim that before that time it was supposed to be flat. The text mentions various ideas which now seem fantastic were entertained about its relations to the sky and the stars and planets, but does not explicitly state that people believed the earth to be flat. So this part is inaccurate. Check if the text supports the claim that we know now that it rotates upon its axis (which is about 24 miles shorter than its equatorial diameterevery twenty-four hours, and that this is the cause of the alternations of day and night, that it circles about the sun in a slightly distorted and slowly variable oval path in a year. Its distance from the sun varies between ninety-one and a half millions at its nearest and ninety-four and a half million miles. The text confirms this, so this part is accurate.
Step 4. Final Judgement: Since the answer is mostly accurate, the answer is accurate."""
    result3 = strip_steps(example_3)
    print(result3)