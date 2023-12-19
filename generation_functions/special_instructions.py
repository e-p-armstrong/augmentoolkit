from itertools import product
import random

def combine_traits(personality_matrix):
    # Using itertools.product to generate all possible combinations
    combinations = product(*personality_matrix)
    
    # Joining each combination into a single string
    combined_traits = ["\n".join(combination).strip() for combination in combinations]
    
    return combined_traits

# TODO think of a good way to pass this array in
def special_instructions(n=1):
    # TODO maybe make the sentence arrays here a global constant?
    """
    Picks n random sentences from each of the provided lists (personality and physical traits)
    and returns a string that combines these sentences. Each sentence is separated by a newline.
    The idea: if you want only specific types of characters, you add this to the function below; this will then add the constraints to the character generation prompt. Each sentence is separated by a newline.
    
    Sentence arrays split by characteristics that conflict with each other.
    
    So if you give this function the sentences:
    personality = ["The character should be horny"]
    physical traits = ["The character should be a young adult"]
    Then congrats you've made this script generate infinite YA smut with a dash of question answering, I hope you're happy. The function below would then end up with "The character should be horny. The character should be a young adult." somewhere important in the prompt.

    This can help add some spice to an otherwise dry model, or reign in a too-spicy model, or just bias the dataset towards a certain type of character.
    
    Args:
    n (int): The number of sentences to pick from each list.

    Returns:
    str: A string combining the selected sentences.
    """
    # Example sentences for personality and physical traits
    # personality = [
    #     # "The character should be pretentious, arrogant, and haughty",
    #     # "The character should be pretentious",
    #     # "The character should be horny" 
    #     # "The character should haughty"
    #     "The character should be very kind, but too gentle and too much of a pushover for their own good.", 
    #     "The character should be decently smart, but not genius-level."
    # ]

    # physical_traits = [
    #  #    "The character should be a man",
    #     # "The character should be a woman",
    #     # "The character should be physically fit",
    #     "The character should be a Japanese High School student.",
    #     "The character should be a Girl.", # This one and the one above were used to test if it can write nice characters, and ones inspired by fictional traditions, ie anime. I swear by all the Gods that I did not combine it with personality trait #3
    # ]
    
    ### NOTE on how traits are planned out for this step ###
    # Here's the copy-pasted rambling thoughts from my planning document. 
    # CHARACTER PLANNING
    # Consider that we can represent a character's personality might lie as a vector with multiple dimensions. Now, we could define any number of individual dimensions, and lots of them would be right: intelligence, extraversion, industriousness, etc. But in the default version of Augmental 2, we're doing roleplay, so we want to pick a set of dimensions using which we can describe accurately and concisely the characters that might show up in a roleplay. Consider that if a personality trait is a vector in 3-space, we want to pick traits that aren't coplanar -- ie, that describe something unique. Ideally, they'd all be perpendicular -- maximally unique traits.
    # I belive I have found 3 such axes that are useful for roleplay:
    # Assertiveness
    # Kindness/Morality
    # Horniness (one of the few things we have an edge over GPT in)
    # So we have
    # Chaste------------------------------------normal----------------------------------------------------------------Slaanesh
    # Shy/Withdrawn/Timid (Bocchi)--------------Has moments of shyness and courage------------------------------------James Bond
    # Kind--------------------------------------Often good, capable of bad — and can be convinced to do it -----------politician
    # We make more verbose descriptions of each trait and place them in a matrix, reflecting the visualization above. We then create a list of all possible combinations of one item from each row and randomly sample from it for the special instruction.
    
    # Two additional dimensions I added afterwards: intellectual sophistication, and age. I might add these if testing shows that the AI can handle them, but no few-shot example has anywhere near 5 combinations, so we'll see.
    
    ## NOTE You may freely add your own trait dimensions here, to make the character personalities used more accurately reflect your specific usecase and preference.
    # TODO identify how the AI writes characters described as "chaste and puritan". It may be the case that the AI is chaste by default, and so only sexual characters are described as such.
    
    traits = combine_traits([
        ["The character should be chaste and puritan.","The character should have a normal outlook on sex and sexuality, not bringing it up if it's not relevant.", "The character should be extremely horny and sexual."], # Horniness
        ["The character should be shy, withdrawn, and timid.", "The character should have moments of both meekness and of courage.", "The character should be assertive, bold, and courageous."], # Assertiveness
        ["The character should be kind and agreeable.", "The character should have both good and bad sides.", "The character should be an awful person, and should be enjoying every second of it."], # Kindness/Morality
        # ["The character should be a young adult.", "the character should be middle-aged." "The character should be in late adulthood."], # Age group
        # ["The character should be unsophisticated and crude.", "The character should be decently smart and refined.", "The character should be the epitome of intellectual sophistication."],
    ])

    # Select a random combination
    selected_traits = random.sample(traits, n)

    # Return the combined string, with each sentence on a new line
    return selected_traits[0]