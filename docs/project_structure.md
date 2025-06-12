# Project Structure

This page details what the most important parts of the project are. I'm making this mostly for developers or power users who want to understand "what goes where" in the codebase at a glance. It's also useful for contributors and collaborators.

There's one more kind of person this is for: if you're the kind of person who feels a slight fundamental queasiness of you don't see the whole picture of something you rely on, this will give you a medium-resolution image to rely on, as well as some links to details.

Finally, this would probably be good context for an LLM to understand the project it is working in better.

If you're more interested in how to run this, check out the [CLI](CLI_flows.md) and [INTERFACE](./interface_flows.md) guides.

- The start scripts either launch the interface and its associated services, or launch a pipeline directly via the command line.

### Pipelines summary
- **Pipelines are in `generation/`**
- **Pipelines are just Python functions**, generally intended for creating training data for LLMs, that optionally adhere to < 10 [conventions](./pipeline_conventions.md).
- You can make your own pipeline just by making a new folder within `generation/` and defining a function there. **You are encouraged to do so if you have a cool idea.** You can get help, advice, or support building custom pipelines on the [Discord](https://discord.gg/s6PBfsaVzu).
- Pipelines get their arguments from **config** files as keyword arguments.
- Pipelines are called by the CLI in `run_augmentoolkit.py`, or run as a subprocess by the API.
- Pipelines, being Python functions, can be imported and used inside other pipelines. The most powerful Pipeline in Augmentoolkit, `complete_factual_datagen`, is a composition of most of the other pipelines.
- Pipelines use a group of abstractions in `generation_functions/` mostly intended to make LLM-powered dataset generation easy. 
    - These abstractions ensure efficiency, allow runs to be resumed, and cover behaviors of LLM datagen pipelines including file chunking, saving of intermediate outputs, resuming partially completed runs, pipeline step input preprocessing and output postprocessing, configurable majority vote filtering of items, and more.
    - While the conventions mentioned earlier are entirely optional from a programming standpoint (you can run just about any Python function with Augmentoolkit if you put its arguments in a config file) the abstractions do heavily encourage certain design decisions. For instance, data is usually stored in a large dict during program execution and also during saving.
- A basic template for a pipeline can be found in generation/core_components. **It contains all the standard boilerplate needed for an LLM-powered pipeline and thus is good copy-paste material.**
- `generation/core_pipelines` contains focused pipelines that are part of the core Augmentoolkit project.
- `generation/core_composition` contains pipelines that are mostly composed of other pipelines.
- Augmentoolkit knows what config file to get its kwargs from when executing a pipeline function because either the interface provides the arguments in its API call (interface) or the file `super_config.yaml` specifies a path to a config in the `config` field.

### Configs Structure
- **Configs store arguments for pipelines.**
- `super_config.yaml` controls which pipeline is executed with which config when Augmentoolkit is run in CLI mode. `super_config.yaml` also has the **path aliases** section, which is used by the API to keep track of what pipelines exist (used for choosing what pipeline to run), and where their primary configs are (used for duplicating them to serve as the basis for a new pipeline).
- By convention, at least one of a pipeline's config files is stored in the same folder as the pipeline itself. If a path to this config is listed under path_aliases then it will be available as a template to copy and modify in the interface.
- Pipelines almost always have [some common sections](./config_common_fields.md) which are detailed in [this explainer document](./config_common_fields.md)
- Fields that don't really have a sensible default can have their value set to "!!PLACEHOLDER!!". The interface will recognize these lines and flag them in any config that is opened, letting you help prevent people forgetting to fill in important details like API keys, etc.


### Interface summary
- The interface files are in `atk-interface/`. It's a React + Vite application.
- The start scripts automatically do an npm install and build.
- The interface interacts with `api.py` which interacts with a Huey worker which runs `run_augmentoolkit` as a subprocess.
- The reason why run_augmentoolkit is run as subprocess is to let Augmentoolkit run on the main thread of the main interpreter of a process. This prevents the code from getting severely complicated. The idea is that the Pipeline programmer should not have to think about the API at all when coding; they just make a Python function that produces the data they want, and Augmentoolkit makes running that function nice and easy (while also providing the abstractions needed to make writing the function itself nice and easy, too).
- The interface has a bit of animation and behavior meant to make the dataset generation a charming and fun experience. Animated backgrounds which react to the actions taken are a part of making good first impressions, and making long-time users have a persistent source of joy in the process (and not just the outcome of getting a custom model). Some might call it overdesigned, but if you do this for 60+ hours a week you, too, will appreciate the fireworks celebrating a run's completion on the `Outputs` page.

### Abstractions
See the [Abstractions Primer](abstractions_primer.md)