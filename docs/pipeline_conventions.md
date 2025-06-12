# Pipeline conventions

Pipelines are just Python functions and they can contain literally anything. This was done to make them easy to get started building. That being said, there are some **conventions** that make building them even easier.

1. Pipeline functions should take `**kwargs`
    - Reason: Config files usually outnumber a pipeline many-fold. Including kwargs means that the arguments of a pipeline can be changed (such as removing extraneous ones to simplify logic) without causing older configs to outright error even if they would have otherwise worked. Furthermore, this is necessary for pipelines to work well with the interface, since config files can be given a "pipeline:" argument which will tell the API which pipeline to run, and if you don't have **kwargs and you don't happen to include a pipeline: argument in your function, then your pipeline will error when being run with the interface.
1. Pipeline functions should take a `task_id` argument and call `set_progress()` occasionally
    - Reason: This ensures that the progress bar in the interface looks nice. This is the only requirement imposed by the project API and the code will work without it, but things look nicer with this. It takes <5 minutes to modify any pipeline to fit this requirement at a basic level.
1. Pipeline functions, if they call LLMs, should have the LLMs' prompts be in a folder. This folder should be located in the same folder that the pipeline function's file is in.
    - Reason: prompts in code makes the prompts cluttered and the code cluttered. Keeping the prompts outside of the code makes everything more modular/extensible/customizable, means that non-technical people can more easily modify prompts, and reduces the chances of accidental formatting mistakes, among other things. YAML is preferred over JSON due to its nice handling of whitespace. The [Abstractions](abstractions_primer.md) make working with this constraint easy.
1. Pipeline functions should have a good config file template co-located in the same folder as their file.
    - Reason: organizing config files with the pipelines themselves is a natural way of doing it. The constraints of the interface forced an `external_configs/` folder, but for the CLI (and for the sake of the fill-in-able config file "templates" that the interface uses) it makes sense to include a config file in the same folder as a pipeline is defined in.
1. Pipeline functions should be deterministic (not random) given the same input.
    - Reason: Augmentoolkit's abstractions allow for easy resuming of previously-started-but-interrupted-runs. However this behavior can sometimes become unpredictable or lead to impossible states if a resumed run has different random generations than the run it is resuming. Guarantee repeatable random behavior with set seeds or with tricks like sorting by the hash of a value.
1. Pipeline functions ought to use the [Abstractions](abstractions_primer.md) unless there's a good reason not to.
    - Reason: if Pipelines use the same parts, then when you understand one you understand them all; and it also becomes easy to make new pipelines using the copy-pasted parts of others. Also, the abstractions enforce certain patterns of operating (patterns that are stricter than these conventions) which are generally useful for dataset generation.
    - Don't like the abstractions and think I'm a bad programmer? My feelings are hurt but I welcome PRs to improve functionality and fix bugs. Just please try to maintain compatibility with the previous interface as much as possible.

And some recommendations. These have some overlap with just general good coding practice but they bear mentioning:

- Any argument that does not really make sense to have a default value, should not have one and should not be a keyword argument. If the vast majority of a PipelineStep's arguments are normal arguments, then Python itself checks whether a config file contains all the fields it needs. This behavior is handy and does not happen if you go default trigger-happy.
- Non-reasoning-models are usually better for nailing complex output formats, precise instructions, and out-of-distribution tasks. Reasoning models are better for fast prototyping and varied data. Use the right tool for your job. **But specify what type of model should be used for the small or large model in the config, or a README, so that people using a different model know what kind of model to use.**
- Fail rather than continue. You usually don't want impossible states to just log and continue on -- Augmentoolkit is something that more often than not runs while you are AFK. And the worst thing that can happen is for corrupted data to be written and the model trainer to not be aware before they dump money on a training run. So, if something really bad happens (on a pipeline level) error. If something bad happens on a sample level, either error if it's during development and you want to debug stricter, or skip the sample entirely and use the rest of the dataset. In general a smaller, cleaner dataset will outperform a larger broken one.

- **ALWAYS. READ. YOUR DATA. BEFORE. YOU. TRAIN.**


#### Objection Handling

> But how does restricting what can be done make doing anything easier?

Because thinking about how to do things the best way is hard, and conventions do that for you. They may not be the "best" way mind, but they're probably not the worst.

> Not all the existing pipelines follow all the conventions!

Clean code; functioning well; being ambitious; releasing anywhere near 'on-time': solo projects can pick two, in my opinion. ATK 3.0 basically redid everything in Augmentoolkit and then some, and it was done solo, so I almost certainly missed some things. I welcome GitHub issues that point out problems! And I adore PRs!


Random Augmentoolkit Discord plug because the community is cool: [Discord](https://discord.gg/s6PBfsaVzu)