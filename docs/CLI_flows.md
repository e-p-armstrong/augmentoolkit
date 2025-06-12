# CLI

The CLI (Command Line Interface) for Augmentoolkit revolves around the `run_augmentoolkit.py` script. The core idea is that `run_augmentoolkit` uses the `super_config.yaml` file to know which pipeline function to run (`node`), and which config file to pass in as arguments to that function (`config`). Specific things like the input directory for a pipeline, its output directory, which models to use... those are controlled by the specific config file being used, the one pointed to by `super_config`.

`pipeline_order` means you can queue multiple pipelines in sequence to "set it and forget it".

To run an Augmentoolkit pipeline, point `node` at the path to a python file and then a function within that file (or use one of the shorthand aliases in `path_aliases`) and point `config` at a config file (or again, use an alias).

## Flows

### Upload Documents

Add your files and folders to whatever the folder the `input_dir` of the config you are using points at. Or change the input_dir of the config you are using to point at a folder with the files you want to use as inputs to that pipeline. Note that the file named `config.yaml` inside each pipeline folder is the "canonical" config for that folder, and you should probably make a copy rather than changing it directly, because this config file is used as a reference when making new configs in the interface.

input dirs and output dirs are always relative to the root of the Augmentoolkit project folder. So inputs are usually written `./inputs/[foldername]` and outputs are usually specified `./outputs/[foldername]`. The inputs and outputs folders are the recommended location to store your inputs and outputs, so that they are visible from the interface and are easier to organize.

### Starting Runs

Once your `super_config` points at the right thing, and the config it points at is all set up the way you want it (see the pipeline-specific help pages for configuration help for each pipeline), run `run_augmentoolkit.py` and Augmentoolkit will execute that pipeline with those settings.

### Observing Results

Augmentoolkit logs a lot of stuff to the console. If you see tracebacks but things keep moving, it is likely these are errors caused by occasional LLM mistakes that are then caught and retried.

If you want to see progress as the pipeline moves forward you can also look at the output directory that your config points to. If that pipeline has been configured with a "log observer" then there will be intermediate outputs — the full inputs to each step, including the prompt, as YAML files of OpenAI API messages — saved to a folder in there. There will also likely be JSON files. Note that unlike older versions of Augmentoolkit, Augmentoolkit 3.0 saves only at the start and end of each pipeline step (or in the event of an error) so there may be a delay in the appearance of JSON files from when your pipeline run starts.

### Getting your Results Back

The outputs of a pipeline are saved in its output folder, usually a subdirectory of `./outputs/` (relative to project root). The specific nature of the outputs depends on the pipeline, and is documented per-pipeline in that specific pipeline's help doc, which you can find a list of [here](../README.md#documentation-pages). Most pipelines output sharegpt training data for training with axolotl, though some (such as Representation Variation) output completion-style data for continued pretraining, and others (such as the Complete Factual Datagen) output pre-configured training folders including Axolotl configs, made ready to copy over to a computer with a good enough GPU to start finetuning immediately.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.