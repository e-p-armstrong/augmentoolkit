

names = [  # Replaces "Albert" in scenarios. Needs to be western male names to avoid pronoun and setting inconsistencies).
    "William",
    "James",
    "John",
    "Robert",
    "Michael",
    "Charles",
    "George",
    "Joseph",
    "Edward",
    "Henry",
    "Thomas",
    "David",
    "Richard",
    "Daniel",
    "Matthew",
    "Alexander",
    "Benjamin",
    "Christopher",
    "Nicholas",
    "Samuel",
]


# N_CHARACTERS_SAME_ANSWER = 25 # number of characters that are the same in the question and answer for a thing to fail validation or be deemed "the same" in various places throughout the code

# N_CHARACTERS_SAME_QUESTION = 15

# N_CHARACTERS_SHARED = 100 # number of characters that are the same in the question and answer for a thing to fail validation or be deemed "the same" in various places throughout the code

# IF USING THE 70b LLAMA 2, MUST SET n_gqa=8 WHEN LOADING
# TODO MAKE A GLOBAL CONSTANT is_70b AND ADD THAT WITH BRANCHING LOGIC TO ALL THE LLAMA CPP LOADERS
