# Augmentoolkit - Data for Domain-expert AI
Augmentoolkit creates domain-expert datasets that update an AI's brain (basically, its knowledge cutoff), so that the AI becomes an expert in an area of your choosing.

You upload documents, and press a button. And get a fully trained custom LLM. Now every aspect of your AI's behavior and understanding is under your control. Better still, Augmentoolkit **optionally works offline on your computer** -- no external API key required* for datagen† on most hardware.

Maybe you want AI to know the latest research papers in your field, or perhaps you want an LLM that understands your passion deeply and has learned from the same sources as you. Possibly, you dream of creating a lore expert for your favorite obscure fictional universe. Whatever the application is, Augmentoolkit lets you take text and make an LLM's brain inherently learn the information contained within. It also automatically creates a RAG-ready dataset (and can start up an inference server) if you want some traditional grounding as well.

Get started now (the interface will guide you through generating your first dataset):
### MacOS (interface)
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash macos.sh # NOTE: Will attempt to install valkey via brew if not found.
```

### Linux (interface)
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash linux.sh # NOTE: will build Valkey from source if a Redis/Valkey server is not running
```

Or for local inference
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash local_linux.sh normal # or you can write "small" or a custom model name to serve the quantized version (for more consumer hardware) or a model of your choice, respectively
```


### Windows (interface)
> [!NOTE]
>
> The interface requires Valkey (or Redis) to be installed and running MANUALLY. [The CLI is easier to get running on windows honestly.](docs/quickstart.md#windows-cli)
Running the bat command will give you install instructions. Alternatively, see [`docs/quickstart.md`](./docs/quickstart.md)
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
./windows.bat # see above note about valkey/redis
```

<sub>*Note that datagen can take a while on a lot of hardware however, don't expect fast datagen on an old mac for instance. And for training you will need either a powerful machine of your own, or to rent (latter is done automatically for you if you so choose).</sub>

<sub>†If you want data to generate faster you *can* use an open-source LLM API, and the quickstart encourages you to. In addition to its custom dataset generation model, Augmentoolkit is optimized for open source LLMs like Deepseek or Llama.</sub>

![](images/augmentoolkit-logo.png)

Augmentoolkit, now that it is on its 3.0 version, has been refined and improved through over a year of professional application and experimentation. It is now the best way in the world to create domain expert LLMs, and it's MIT licensed.

If you use this project and like it, please consider starring the repo! It's also designed to be extremely customizable so consider **forking** Augmentoolkit!

> [!IMPORTANT]
>
> The below links contain very useful information. There is a table of contents, that links to extensive documentation pages for any conceivable part of the project, a bit further down.

[Help Videos](#video-tutorials) I walk through how to do all the cool stuff in this project starting from scratch, including training LLMs with the data and configs you get (takes 10 minutes). Check out the help videos if you want further guidance!

[Community](https://discord.gg/s6PBfsaVzu) If you have questions, if you are training models, if you are building cool new pipelines or extensions on top of Augmentoolkit's code, or if just want to hang out, I'd love to see you on the Augmentoolkit discord! It's also a good place to reach me.

[Newsletter](https://promptingweekly.substack.com/) I write about model training and data generation over on a Substack. Totally free, I just want to help people be able to use the tool better.

[Contact](#contact) I'm doing all kinds of things around this project, if you're interested in the mission and the business of bringing custom, personally-aligned AI to everyone, let's get in touch!

[Build](docs/pipeline_primer.md) Augmentoolkit is meant to be the go-to tool for people experimenting with training LLMs, whether they're a hobbyist or a professional. To that end, building new pipelines is as simple as writing Python functions (while adhering to about 2 mostly optional conventions). Efficient explainers and pipeline templates are provided for you to build your own dataset generation pipelines, and by extension, your own datasets and your own completely custom LLMs.

All configs are fully annotated with comments and placeholders to help you understand them as you fill them out.

## Documentation Pages

> [!NOTE]
>
> that this documentation page (the main README) has an [important note](#note-for-when-you-start-training) about model training for facts that you should read regardless of your experience level.

1. [Quickstart](docs/quickstart.md)
    - [CLI](docs/quickstart.md#the-cli)
    - [Interface](docs/quickstart.md#macos-interface)
1. [Video Help](#video-tutorials) <!-- Pages still todo -->
    - [Generate Data and Train an LLM! (12 mins)](#train-a-model-on-your-own-data-in-13-minutes)
    - [Detailed exploration (Interface)](#interface-deep-dive)
    - [Detailed exploration (CLI)](#cli-and-code-structure-deep-dive)
1. [Vision](docs/vision.md)
    - [What is it? (Technical)](docs/vision.md#what-is-it-technical)
    - [What is it? (General)](docs/vision.md#what-is-it-general)
    - [Ideas Presented](docs/vision.md#ideas-presented-and-hypotheses)
    - [Goals](docs/vision.md#goals)
1. Longstart (customization and development guide links)
    - [Pipelines Available by Default]()
        - [Compositions]()
            - [Complete Factual Generation](docs/complete_factual_datagen.md)
            - [Meta Datagen](docs/meta.md)
        - [Pipelines]()
            - [Multi-Source Recall Factual Datagen](docs/multi_source_facts.md)
            - [Single-Source Recall Factual Datagen](docs/single_source_recall.md)
            - [Representation Variation](docs/representation_variation.md)
            - [Traditional Classifier Bootstrapper](docs/classifier_bootstrapper.md)
            - [Generic Data Rephrase](docs/generic_data_rephrase.md)
            - [GRPO (experimental)](docs/grpo.md)
            - [Correction Data (loss-masked mistakes)](docs/corrections.md)
            - [Rag Data (preparing for enhanced recall)](docs/rag_data.md)
            - [Debug (health check)](docs/debug.md)
            - [Starting Point (build your own pipeline!)](docs/example.md)
            - [RPToolkit](docs/rptoolkit.md)
        - [Utility]()
            - [Basic LLM Server](docs/basic_server.md)
            - [RAG LLM Server](docs/rag_server.md)
            - **[Discord Hosting](docs/discord.md)**
        - [Config Common Fields](docs/config_common_fields.md)
        <!-- - [Training an LLM Walkthrough]() -->
    - [Understand the Tool In Detail]()
        - [Project Structure](docs/project_structure.md)
        - [CLI]()
            - [Flows](docs/cli_flows.md)
                - [Upload Documents](docs/cli_flows.md#upload-documents)
                - [Starting Runs](docs/cli_flows.md#starting-runs)
                - [Observing Runs](docs/cli_flows.md#observing-results)
                - [Getting Your Results Back](docs/cli_flows.md#getting-your-results-back)
        - [Interface]()
            - [Flows](docs/interface_flows.md)
                - [Upload Documents](docs/interface_flows.md#upload-documents)
                - [Starting Runs](docs/interface_flows.md#starting-runs)
                - [Observing Runs](docs/interface_flows.md#observing-runs)
                - [Getting Your Results Back](docs/interface_flows.md#getting-your-results-back)
        - [Training with Axolotl Concepts](docs/axolotl_concepts.md)
    - [Customize and Develop]()
        - [New Pipeline Primer](docs/pipeline_primer.md)
        - [Abstractions Primer](docs/abstractions_primer.md)
        - [Conventions Commandments](docs/pipeline_conventions.md)
        - [Reminder That Conventions Are Minimal and You Can Just Code and It Will Probably Work](docs/conventions_reminder.md)
1. [Discord!](#discord)
1. [Updates and Training/Datagen Tips Blog! Stay in the loop!](#training-and-datagen-tips-blog)
1. [Contributing!](#contributing)
1. [Contact & Client Work](#contact)

**If you're familiar with LLMs and want a more jargonful rundown of what Augmentoolkit is and what makes it cool, check out [this section](docs/vision.md)**

Cite:
[![DOI](https://zenodo.org/badge/726083337.svg)](https://zenodo.org/doi/10.5281/zenodo.11525927)

### Video Tutorials

#### [Train a Model on your Own Data in 13 Minutes](https://youtu.be/E9TyyZzIMyY)

#### [Interface Deep Dive!](https://youtu.be/M-OFVwHPfeU)

#### [CLI and Code Structure Deep Dive!](https://youtu.be/cEkgw7sYqMw)

^ This one is useful if you're going to make modifications to the code

### Benefits
**Augmentoolkit makes LLM data easy.**
- **Cheap:** Augmentoolkit pipelines use open-source LLMs, and so can be run on consumer hardware for hardly any cost, or cheaply via APIs like Deepinfra *(the "local" prompt sets should enable usage of most pipelines by reasoning models, too)*
- **Effortless:** Any Augmentoolkit pipeline can be run with an intuitive interface that is started by running a start script. Alternatively, you can make data by putting some files in a folder, and then running a Python script. If that's too much, you can also use the graphical user interface, now a first-class citizen in Augmentoolkit 3 (and in fact, the recommended way to run Augmentoolkit). Previously-started runs are continued automatically, so you don't need to worry about interruptions costing you time and/or money.
- **Fast:** when using APIs, you can quickly generate millions of trainable tokens. Fully async code lets you get results quickly. Reading and chunking caches ensure that even large-scale workloads are quick to use. Models are automatically trained after the data is ready, and are even automatically downloaded and prepared for inference on your local machine. All the hard or annoying parts of the process have been automated and made efficient. In the past creating datasets and iterating and testing and learning could have taken a skilled person months; now, anyone can press a button, come back in a day, and chat with a newly-trained model.
- **Innovative, Effective Approach to Factual Training:** Augmentoolkit has a production-tested method of creating domain-expert LLMs that can understand entirely new subjects. Many separate pipelines are composed together to produce quality datasets that teach capabilities such as answering factual questions, acknowledging when something is not known by the model, correcting mistakes, etc. You can be confident in getting high-quality specialist models when you use Augmentoolkit.

We've also done our best to **facilitate the step after you generate your data -- training your LLM:**
- **Production-Scale:** Datasets that are gigabytes-large have been generated with Augmentoolkit -- it is battle-hardened, it works at scale without annoying inefficiencies costing immense time, and it is ready for the stresses of production.
- **Train an AI for the cost of a dinner:** you can generate data on your own hardware for what is basically free. Augmentoolkit can then automatically perform a full finetune of an AI, on your own data, for a tiny sum of money (roughly $20 for the finetuning part of the process).
- **Create your LLM in less than a day:** with a fully automated process for turning documents into datasets, and only a single button-click needed to kick off training, making a subject matter expert LLM is *fast* (especially when you use API for the dataset generation). Iterate quickly and cheaply.
- **When you use the same recipe, you get the same bread:** Augmentoolkit datasets have been used successfully for professional consulting projects. Video documentation is linked in this README that shows exactly how to use this tool to do the same. The code, settings, and prompts you need is all here. Examples, templates, comments, marked-out placeholders, and extensive documentation is all available.
- **Train AI with confidence, *especially* if it's your first time:** between the battle-tested process, extensive video docs, in-depth README, and Discord community, you can be confident you'll get a good LLM out of this.

**Do it all locally**
With a custom-trained 7b model built to run these pipelines specifically, Augmentoolkit can generate data on consumer hardware, and can do so at incredible scale, with incredible parallelism, when on higher-performance computers. Budget does not need to be a constraint -- just passion and time. Of course, if you want immediate results/speed, you can use an API too.

Finally, **using the model you create should be easy and valuable:**
- **AI that understands your facts:** For the professionals and the passionate: training an LLM with Augmentoolkit's Complete Factual Datagen "composition" pipeline creates an assistant that understands the big picture of the data you're training on. If RAG is like giving an LLM an open-book test on a textbook it hasn't read before, then training on Augmentoolkit data gives it some time to study before the test as well. This pipeline has been battle-tested in consulting projects across different industries. Compared to earlier versions of Augmentoolkit, Augmentoolkit's 3.0 version generates a wide variety of different domain data, and it even automatically balances this data with the generic data it uses.
- **Individual Alignment:** Use GPRO (the same algorithm that made Deepseek R1 as good as it is) to align a model to any task imaginable without modifying any code. Augmentoolkit adopts an innovative approach of letting you use an LLM as a reward function -- you write a prompt that grades certain outputs higher, and then those reward scores teach the model to behave more like that in the future. Want your model to do a task better? Explain what "better" is and then the model will learn it. Want your model to be more emotional and human-like? Explain how to grade responses based on their emotional content, and the model will [learn it](https://huggingface.co/Heralax/llama-gRPo-emotions-nothoughts). Want your model to write like a pirate? Explain in your grading prompt what makes a good pirate-like response, and the model will learn it. You can also change code and use traditional reward functions if you want to. The GRPO pipeline is experimental and in beta, but early results are promising.
- **Make sense of massive data without using human annotators:** For the heavy-duty ML professionals: if you have a large dataset with tons of unlabelled text (like the Enron emails dataset, IMDB, or fineweb, etc.) you can now write a sentence or two that describes two classes which exist in that data. Augmentoolkit's classifier creator pipeline will then use an LLM to make a full classification dataset, based on a subset of the input data and your specified classes; it'll then train a classifier and evaluate it and take more data and retrain, in a loop, until validation loss is below a specified threshold. Classifiers trained using this pipeline seem to achieve similar performance to classifiers trained on human-labelled data. Be advised that data is not yet automatically balanced between different labels.
- **AI inspired by your favorite fiction:** For the creatives and entertainers: using RPToolkit, you can create detailed and varied multi-turn roleplaying data with the themes of any story you can think of. If you're creating custom AI for creative or entertainment purposes, you can now specialize it in any genre you want. Want a depressing and dark specialist in mecha stories? Feed in some stories and you can get a ton of data for that. How about an AI writer of wholesome slice of life? You can get data for that too. Create as broad or as narrow of a writing AI as you want from whatever inspiration you can find.

*Clarification: Augmentoolkit, the project, has multiple pipelines: the original pipeline (QA), RPtoolkit (rich multiturn roleplaying data), and the classifier creator. If it is said that "Augmentoolkit can make [some kind of data]" then I mean that one of Augmentoolkit's pipelines can do so.*

### NOTE For When You Start Training

Factual finetuning requires a certain number of optimizer steps to stick. If training is where the LLM's brain "moves" towards a place where it understands your new domain, "optimizer steps" are the number of times the LLM moves. **If your dataset is small you may not have enough optimizer steps for the LLM to learn the new domain well.**

Because of this, ironically, it can be easier to teach LLMs large new domains rather than small ones, with training. However, there are tools at your disposal for turning a small dataset into a large one when you use Augmentoolkit.

In [complete factual dataset](docs/complete_factual_datagen.md) you have the `number_of_factual_sft_generations_to_do` setting for the whole pipeline, and the `variation_generation_counts` which you can customize per input dir. The one that is customized per dir makes the data from a specific input dir represented more in the continued pretraining data; the other setting increases the overall amount of SFT data made from all input dirs together. With these two levers you can make a small dataset as large as you need — though some of the data may be very similar, you can still scale it up in this way to teach it to an LLM without catastrophic drawbacks.

As a "break glass in case of emergency" option, if your dataset is exceptionally small, you may want to consider turning sample packing off. This an be done by modifying the pretrain and finetune kwargs to set sample packing off (do this in the complete factual datagen config).

```yaml
other_pretrain_kwargs: {sample_packing: False}
other_finetune_kwargs: {sample_packing: False}
```

Turning off sample packing has not been tested with the current iteration of Augmentoolkit's settings yet, so success with that emergency approach cannot be guaranteed for extremely small datasets, but since the main problem with extremely small datasets is a lack of per-epoch optimizer steps causing the LLM to not learn the data enough, *theoretically*, this should work.

Most of the configuration of Augmentoolkit you'll do, besides changing input/output paths for different models, is probably going to be related to the optimizer step. With very large input datasets you'll want to reduce things that increase the optimizer step because otherwise you'll be training for a long time, whereas with very small ones you'll have to pull out tricks to increase it. That's why this has a section and is marked out as important -- be cognizant of the size of your dataset as you create it!

If you have any questions about your specific use case, consider heading over to the [Discord](https://discord.gg/s6PBfsaVzu)

### Temporary Announcement if You've Been Here Before
If you're returning to look at this repo after a while, I want to make a few things clear!

Firstly, a *lot* has changed. I disappeared for six months, half of it was spent on research and the other half was spent building. I wanted the tool to fundamentally work better. Now, Augmentoolkit reliably produces great domain experts across datasets of different sizes. It can even teach a model something that it has not seen at all during pretraining. This experimentation cost thousands of dollars out of pocket, but I believe it was worth it, since anyone can now make domain experts about arbitrary subjects with very little technical experience.

Secondly, things are *much* easier to use. The interface is robust and is not a second-class citizen anymore. Start scripts, automatically generated and balanced training configs, better error messages, and a host of other improvements should make Augmentoolkit much nicer to use.

Not much code from the original survived, though making older pipelines fit into the project as it is now, is pretty simple ([see the new pipeline example and you'll get a picture of how they look like now](/docs/example.md)). Also, if you have custom prompts from before, they should work with the new pipelines without modification. The older pipeline compared to this one is like a rat compared to a human -- they're technically related and have a lot of the same DNA, but the human is much more evolved and more capable. I hope you enjoy using the new project and getting great results.

The bad news is that since so much has changed, some new bugs probably got introduced. Please report bugs so that they can be fixed. I had not been keeping too close an eye on the issues these past 4 months since the entire project was being ripped apart anyway -- now that it is in a more final form, and frankly now that I have better discipline with this sort of thing, I'll be focused on the Discord and GitHub issues to correct any mistakes that you point out. If any of the new documentation is unclear about parts of the project, please let me know. And, if you have [custom pipelines]() that you want to add, or bugfixes, please check out [Contributing]() and make a PR!
 <!-- Not much survived from the previous version, this is like a modern ape observing a rat -- technically related, lots of the DNA is the same, but the ape is much more evolved and capable. Also since everything is differnt and the project is much larger some of the accumulated fixes will have gone away. Please report bugs so that they can be fixed. -->

### Discord

Custom-built models are (usually) not meant to be enjoyed only by their creators. There's a [new feature in Augmentoolkit](docs/discord.md) where you can easily make your custom models into Discord bots! Now you can share your custom AI creations with your friends or community! Also, all the code runs on your own computer so no worry about recurring costs.

Speaking of Discord...

Augmentoolkit is partly about democratizing dataset generation, so community is hugely important to the project! There's a Discord server where you can get help with custom model creation, as well as share new pipelines, prompt sets, or projects you're creating! [Come hang out and be part of a useful community of like-minded people!](https://discord.gg/s6PBfsaVzu)

### Training and Datagen Tips Blog

I write about model training and data generation over on a [free substack](https://promptingweekly.substack.com/)! If you want read access to my brain as I continue to experiment and explore with dataset generation, consider signing up to up your model creation game. If you're planning on building your own dataset generation pipelines using the tools and abstractions provided by Augmentoolkit, some of the advice there might be very useful.

Now that the new Augmentoolkit version is out, I finally have time to post again (and new ideas to post as well).

### Contributing!

PRs for bugfixes, new pipelines, and improvements are welcome! If you have an experiment you're proud of, consider opening a PR. The rules are pretty standard:

- Contributors can open PRs
- Collaborators can push to branches and merge PRs into the master branch
- Collaborators may either be chosen depending on contributions, or may be chosen internally within Augmentoolkit (the company)
- [The example pipeline and its documentation](docs/example.md) contain useful information for making your own pipeline. You are encouraged to fork Augmentoolkit and experiment!
- Code with the style you want, just test thoroughly before making a PR
    - Caveat: failing silently or continuing is worse than explicitly erroring if an impossible state is reached
    - Asserts are your friend
    - `black .` makes even MY code look formatted nice, it can do the same for yours.

### Useful Commands for Datagen and Training Workflows

Copy-paste these when appropriate or use them as a reference.

Copy files over to a different computer (such as a GPU instance on runpod)

```
scp -P [port] -r ./outputs/your-output-dir/pretraining_run root@123.456.78.9:/workspace/axolotl
```

Kick off a training run:
```
accelerate launch -m axolotl.cli.train [your_config].yaml
```

Convert and quantize with llama.cpp
```
python ~/llama.cpp/convert_hf_to_gguf.py --outtype q8_0
```

## Contact!

- Email me at: evanpeterarmstrong@gmail.com (NOTE: my inbox is flooded, your message might not get through, prefer booking a call for serious discussions)
- [For serious and urgent discussions, we can schedule a call!](https://calendly.com/evanpeterarmstrong/30min)
- [I'm pretty active on the Augmentoolkit discord server and a bunch of other AI discords. Find me as @heralax!](https://discord.gg/s6PBfsaVzu)
- [I sometimes post stuff on X/Twitter](https://twitter.com/e_p_armstrong)
- [Substack! I am finally posting again.](https://promptingweekly.substack.com/)
- [YouTube -- The source of the help videos](https://www.youtube.com/@Heralax)
- [Let's connect on LinkedIn!](https://www.linkedin.com/in/evan-armstrong-1a84b3200/)

If you have a company or organization that wants to serve custom domain-expert AI to internal users (to get employees the information they need to do their jobs great) or to external users (for instance, to answer community questions or increase product awareness) then we should [get in touch](https://calendly.com/evanpeterarmstrong/discovery-call).

Also, if you have an **AI Chat Wrapper Startup or Company** that is currently being extorted by OpenAI, [we should also talk](https://calendly.com/evanpeterarmstrong/discovery-call), because I'm fairly certain **I can save you a ton on API costs while maintaining or improving answer quality**. Not only will I use this tool to produce a quality model, but I also have the means to run it at scale, which is not a trivial problem to solve.

The Augmentoolkit project is going to continue to be developed. It has been consistently developed for a long time -- the many-months gap from this major update to the previous one is because I was busy researching and developing the techniques, and later, preparing this release itself (this update has been in the works a long time). I am open to and would appreciate organizations sponsoring this open-source project -- I'd like nothing more than to research and build the tools for creating custom LLMs all day!

I am also working on ambitious commercial solutions involving Augmentoolkit and Augmentoolkit tech. This project is part of a larger master-plan. If you're an investor, I'm very open to discussions here! My [Calendly](https://calendly.com/evanpeterarmstrong/30min) is always open.

Current Generalized AI means hallucinations, platitudes, and sycophancy; domain-expert AI writes with your knowledge, understanding, and is aligned to **your** tastes. As more people use the same AI, more of the world sounds the same and thinks the same--it is my belief that individually-customized LLMs are the only way to avoid this world of slop.