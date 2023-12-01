RP_MODEL = "./rp_model" # model used for RP tasks, probably going to use Sao10K/Euryale-1.3-L2-70b
LOGICAL_MODEL = "./logical_model/airoboros-l2-13b-3.1.1.Q8_0.gguf" # model used for decision-making and base question generation (should be "smart")

# IF USING THE 70b LLAMA 2, MUST SET n_gqa=8 WHEN LOADING
# TODO MAKE A GLOBAL CONSTANT is_70b AND ADD THAT WITH BRANCHING LOGIC TO ALL THE LLAMA CPP LOADERS