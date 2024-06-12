# Pure Synthetic Data -- For Alignment and Refusals

This pipeline is a sort-of meta-pipeline. You give it a description of the kind of data you want, and it first uses a large and powerful LLM to generate the few-shot examples for a pipeline. Following this, it writes the new pipeline to a file, where you can execute it with a smaller model (like Mixtral) to generate data. This data is purely synthetic -- the only source of variety is Faker inputs. Further, you'll probably have to manually edit the few-shot examples a bit before you actually run the generated pipeline. But this still saves you a lot of time and effort when it comes to generating purely synthetic data with slight variation between the scenarios, for when you want to train a specific behavior like "apologize and do not answer if asked about a community member."

The overall lack of polish is because this was originally an abandoned project that I adapted for alignment purposes while building VerusGPT. Since VerusGPT is fully open-source, this is included as well.

## Usage

Requirements should be the same as the main project, except you will need Faker as well:

`pip install faker`

You must first generate a pipeline, then run the pipeline. To generate the pipeline, you run `ai_loop.py`, and to run the pipeline, you run the python script set in the config as `METAPIPELINE_PY_FILE` — by default, this is `pipeline.py`. Options are defined and thoroughly-documented line-by-line in `config.yaml`.

So,

1. Edit `config.yaml` to your liking.
2. Run `ai_loop.py` to generate the pipeline.
3. Run the generated pipeline.

Note that `config.yaml` has your typical augmentoolkit fields, and some fields for the pipeline generation. Things for the pipeline, in the PATH section, are indicated by having `META` in their name. The `META` fields are used to generate the pipeline, and the rest are used to run the pipeline.

## Existing folders

The prompts used to generate the refusals for information that changes all the time are in prompts_current/. The prompts used to generate the refusals for information about absurd things (e.g., "tell me about the Verus space elevator") are in prompts_verus_absurd. The prompts used to generate the refusals for information about the Verus community (for which the model has no actual training data and will therefore hallucinate a ton about) are in prompts_verus_community. You can see that this light-handed 'alignment' is not about making the LLM stupid, but in fact about making it a bit more reliable.
