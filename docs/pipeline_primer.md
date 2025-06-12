# New Pipeline Primer

Making custom factual experts is awesome. You can update the knowledge cutoff of an AI, teach it all about completely niche things, and use it in place of much more expensive large generalists while still engaging in reliable factual QA. 

But if you're a creative model finetuner you can probably think of a lot of other unique behaviors you'd like an AI to have. If you've ever thought of or have built an AI-powered project/application, there's probably been some gap between off-the-shelf AI capability and your use case that you've desperately wanted to bridge.

This doc explains when and why to make new pipelines, and some of the philosophy behind their effective design.

If you want to see 

## When to make a new pipeline?

When you want an LLM to do something more specific than generic conversation/coding. Better than all the other models. For cheaper.

## Why to make a new pipeline?

Custom models afford more control, personalization, charm, and a complete lack of censorship -- all at lower costs for scaled applications.

This makes them superior both for personal projects (personalization, lack of censorship) and business (margins and quality are everything).

Augmentoolkit's goal, besides making it really easy for you to create factual expert models, is to provide you with everything you need to make the custom dataset generation pipelines of your dreams (and from there, to train the custom AI models of your dreams). Pipelines are fantastic: not only do you make a dataset that can help you tune a custom model for the task, but pipelines take in data and use models to create different data from it -- meaning that they passively improve over time as you get more input data and as the models running the pipelines get better.

## How to make a new pipeline?

If you're getting started with your first datagen pipeline, check out the example pipeline and its [documentation](example.md) to understand the structure of Augmentoolkit pipelines. They're basically Python functions with a few (optional) conventions and a lot of (very powerful) abstractions that make chunking text, chaining and saving long sequences of LLM calls, and debugging these things, easy as pie.

The rest of this section is philosophy. I.e., where `example.py` shows you how to actually build a pipeline's code, this section explains how to make decisions early on that make your life easier down the line.

Honestly, design philosophy and process is a very nuanced and personal thing. It takes a lot of text to fundamentally make an attempt at imparting a person's way of doing things and all their context and experience and decisions and tradeoffs. Therefore, this is more of a glimpse, a starting point for your own process, since I'm not sure I have fully systematized the pipeline *design* process to a sufficient degree yet for me to feel comfortable writing about *the canonical way* to do it. Use this blueprint to avoid getting lost with how to start -- this is a workable route from start to finish, even if it is not the best one.

The main ideas discussed: deciding on the desired end state and input-output pairs; sketching out and building the initial chain of steps to bridge your raw text and that end state; and iterating from there to add steps as needed to improve output quality.

The worked example here is: writing style cloning.

Firstly, have clearly in your mind what you want the final LLM to output. Do you want it to be able to answer factual questions? Do you want it to follow long, complex instructions? In our example, we want it to imitate a specific writing style. Have a clear idea of what this looks like in terms of input and output. Then, considering what raw text we have available, how do we turn our raw text into that input/output pair? (or pairs).

Basically: desired behavior -> mental picture of inputs and their outputs -> what raw text do we have -> how do we turn the raw text into the input and output that we need?

That is the challenge of the model creator. And you want to architect things out by thinking through it first, before you actually start coding. Once you have the desired end state in mind, you can conceive of a sequence of LLM calls on your raw text input, to assemble the pieces you need to create the input/output pair.

In our example: writing style cloning. We have some text written in a specific style, and we want to teach an LLM to rewrite generic text into that style. The stylistic text is our raw text, and the desired outcome is that we have some generic text (input), and the LLM rewrites the generic text into the style of our raw text. How do we bridge the gap?

Let's assemble the first piece. We want generic-sounding text with the same fundamental content as some text written in the style of the raw text we have. What's the easiest way to get that? Well, LLMs can certainly rephrase things to be in a generic, corny style -- so let's make the first step of our pipeline be to rephrase chunks of the raw stylized text, to have a generic style. The hope is that we will then be able to use the generic text as the *input* in an input-output pair, and the raw text as the *output*.

So we code this step as a PipelineStep with a new prompt. For the sake of speed we opt to use a reasoning model like QwQ or R1 Llama Distill and no few-shot examples (if we were to have used a non-reasoning model, we'd have more control over the output structure, but it would take longer to develop due to having to write the few-shot examples, and it might be more inflexible/less varied). We get some AI to write some code that formats the generic rephrase (input) raw text (output) into sharegpt samples. We then run our MVP (minimal-viable-pipeline) and inspect the resulting data.

This is the beginning of a simple iteration loop: run the pipeline; inspect the final result; if it is lacking, identify what component is lacking and then hypothesize what step which builds towards that component might be lacking; fix that step.

Looking at our data, we would likely see that the start and end of our samples is a but sudden and sometimes even mid-paragraph. Though Augmentoolkit's chunking algorithm does its best take groups of contextually relevant text, sometimes things go poorly. And this results in our data being composed of unnatural text as an output.

So we think about how to correct this. We have unnatural stops and starts sometimes. We could just roll with it -- the end user of a rephrasing model is not always going to pass perfect text anyway. Accepting a "shortcoming" as a feature is a fantastic way to save time when developing things, especially with LLM-based systems, which are hard to get perfect. Alternatively, we could fix the starts and ends of our unnaturally truncated raw text by adding another LLM step to add in a short intro sentence and concluding sentence at the start and end, to make our text more self-complete. Either way, we'd likely reach a finished pipeline by the next iteration.

Overall, what we did was: we identified the type of pipeline we wanted to build; envisioned what the inputs and outputs would look like; sketched out and built an initial version of the pipeline to create that data; inspected the data and found a problem; and we revised the pipeline until the problem was either acceptable or no longer present. This sequence of behavior can serve as a blueprint for you to follow when you start developing pipelines with Augmentoolkit -- you'll probably find your own preferred way of working, but if you're staring at the getting-started annotated template in the `generation/example_pipeline/example.py` file and are not sure how to start working on your vision, maybe try this.

Further documentation on some common components of Augmentoolkit pipelines can be found at [the example pipeline's documentation page](example.md). Documentation of Augmentoolkit's main abstractions (basically, functions and classes) can be found at [the abstraction primer](abstractions_primer.md). I hope you enjoy making the custom LLMs of your dreams! And if you want help, please check out the #building-stuff-channel on the [Discord](https://discord.gg/GZnxtWqh)

If you make a new pipeline, consider forking Augmentoolkit so that you can more easily make a PR back into the main project -- I'd love to incorporate cool pipelines into Augmentoolkit so that we can get closer to the goal of democratizing custom LLM creation.