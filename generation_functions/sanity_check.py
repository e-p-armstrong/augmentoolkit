import re
from .answer_accurate_grammar import answer_accurate_grammar
from llama_cpp import Llama
from .constants import LOGICAL_MODEL
# Answer vetting
# For now, this checks answer relevancy too. The danger with abstracting answer relevancy into a separate step is that anything which relies on knowledge that is obviously mentioned in the text already up until this point, will get screwed


# TODO improve the step-by-step check to formalize the fact that we evaluate each part of the answer. So "accurate" or "inaccurate" at the end of each step.
# Also improve the test, right now the "good" answer is actually a bit shit
# ^ one big unfixed problem here is that the model might occasionally say "This text is not neccessarily inaccurate" and that usage of the word might make the determination inaccurate. To fix this a reliance on the final judgement can be used, or I can prompt it to only use "inaccurate" if it is describing a part of the answer as bad, or through more precise grammars for the step-by-step bits.

def sanity_check(logic_llm):
    retries = 0
    while (retries <= 4):
        decision_prompt = f"""Hi there, """
        # print("DEBUG\n\n" + decision_prompt)
        completion = logic_llm(decision_prompt, max_tokens=100, stop=["</s>","# Input:"], echo=True,grammar=answer_accurate_grammar,temperature=0.2)["choices"][0]["text"]
        print(completion)

        return
            
            
if __name__ == "__main__": # test
    logic_llm = Llama(model_path=LOGICAL_MODEL,n_gqa=8,offload_kqv=True,n_ctx=8000,n_gpu_layers=1000,rope_freq_scale=0.33,rope_scaling_type=1) # load the logical LLM and offload everything
    # Q0 is good q, bad a
    # q1 is good q, good a,
    # q2 is bad q, bad a,
    # q3 is iffy q, good a
    
    
    
    d = sanity_check(logic_llm)
    print(d)
    