RP_MODEL = "./rp_model"  # model used for RP tasks, probably going to use Sao10K/Euryale-1.3-L2-70b
# LOGICAL_MODEL = "./logical_model/airoboros-l2-13b-3.1.1.Q8_0.gguf" # model used for decision-making and base question generation (should be "smart")

LOGICAL_MODEL = "./logical_model/flatorcamaid-13b-v0.2.Q8_0.gguf"  # model used for decision-making and base question generation (should be "smart")

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

INPUT_DIRECTORY = "../../inputs/"

# N_CHARACTERS_SAME_ANSWER = 25 # number of characters that are the same in the question and answer for a thing to fail validation or be deemed "the same" in various places throughout the code

# N_CHARACTERS_SAME_QUESTION = 15

# N_CHARACTERS_SHARED = 100 # number of characters that are the same in the question and answer for a thing to fail validation or be deemed "the same" in various places throughout the code

# IF USING THE 70b LLAMA 2, MUST SET n_gqa=8 WHEN LOADING
# TODO MAKE A GLOBAL CONSTANT is_70b AND ADD THAT WITH BRANCHING LOGIC TO ALL THE LLAMA CPP LOADERS
