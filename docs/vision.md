# Vision

Augmentoolkit *is* a production-ready way to create AI subject matter experts. It *will be* the tool that everyone uses to get AI aligned to their specific objectives and tastes.

I believe alignment is solved when individuals and orgs can make their AI act as they want it to, rather than having to settle for a one-size-fits-all solution. The moment people can use AI specialized to their domains, is also the moment when AI stops being slightly wrong at everything, and starts being incredibly useful across different fields.

If AI is to power the intellectual efforts of a new modern world, then the people using it must be free from monopolist rent collecting labs, they must have control over the models they use (which goes beyond the weights -- to control something they must be able to customize it) and they should have an alternative to one-size-fits-all alignment. Do you trust a handful of researchers to correctly set the opinions of the AI that everyone will use to assist their thinking? Even if you trusted their ethics (which they have not given us reason to) it is an impossible task by itself since people think and believe differently. People must be able to choose.

Beyond these problems, custom AI unlocks new possibilities and solves other problems. The of the most exciting things about AI is that it lets people produce a lot of written content quickly. But because models are expensive for big labs to make, they have one main model for everyone (and maybe a larger one for coding power users). This generic approach has led to a lot of "AI slop" -- generic trash outputs produced with little care by a generalist model then spread far and wide to the chagrin of anyone with any taste. Dead internet theory is a big concern, too. A future where all written content is created by things that sound like a LinkedIn post is certainly concerning.

AI is sloppy and recognizable because it is the same everywhere and it is made by people with poor taste. If people could customize the models they use to write, then the models would be an extension of people's style and tastes -- the model they use, being made by them on the data they chose to use, would be unique. Because the models are unique, they would not produce generalist "slop" — the result of [researchers with no artistic sense](https://community.openai.com/t/dall-e-3-opinionated-boring/408979) seeking improvement in the wrong way and shipping the same "solution" to everyone. With Augmentoolkit, the creator will have put effort into their AI model and made it a reflection of what they were aiming at, and so the taste of the end user (rather than the researcher) would come into play again. The diversity of tastes and perspectives and ideas that makes up human creative discourse would be able to survive the coming age where productive professionals *have* to use AI in order to keep pace with their competitors.

Imagine if GPT wrappers gave real value-add and had full control over their product by creating and curating expert models on the subject they offer a chatbot for. Imagine if serial authors on sites like Royal Road were able to escape the burning-out grind of rushing out chapters daily, and instead could focus on the actual fun parts of the process (or marketing). Imagine if all those AI SEO content blogs that companies are using actually began to have a voice that sounds like the brand (and not like endless GPTSlop that sickens potential users). **Imagine if we could have the dramatic productivity increase of AI without compromising the content we're using it for.** **If picky people could comfortably use AI for things they really cared about, AI would see even more adoption.**

There are other issues Augmentoolkit could tackle. Because with custom AI people control their own models, enterprises avoid the "checkpointing problem" that prevents a lot of organizations from safely making AI apps -- when you're using someone else's model, that "someone" might decide to one day make it worse (as they did with Gemini, or GPT-4... etc.). Or that provider might have downtime. If you don't control the core of the product that you are delivering to customers, then you have no answer or recourse when that product fails due to no fault of your own. Custom AI is incredibly valuable to enterprises which need stability and transparency in their tech stack for the same reason why Linux is invaluable (this is also why Augmentoolkit is open source). Also, with consumer AI being small enough to run locally, even the privacy issue is sidestepped. There's a lot more (I have been working on this for a while, after all) but those are some of the main motivations behind Augmentoolkit.

We can build ASI -- Artificial Specialized Intelligence -- for everyone. Or rather, everyone can build it for themselves.

But how does teaching models new facts get us there?

Of all the emergent capabilities of LLMs, facts have been (until now) one of the hardest to teach to models. They're not in the context window and they require high reliability. They also come up everywhere in custom AI. Consider: an opinion is just a subjective fact. If you want to have serious control over a model's capabilities, perspectives, beliefs, or alignment, you will likely run into facts sooner or later. Perhaps it was not wise to start with the hardest part of the problem, but I did (mostly by accident at first) and now after a year it's solved decently well. Humans disagree about facts all the time, and we invent new ones when we write creatively -- for people to have actual custom models, the models must be capable of believing arbitrary facts as determined by the user.

With this solid foundation in place, Augmentoolkit is going to next take aim at writing style, and general task training with reinforcement learning. I already know how to do writing style (I have done it before for another project and retained the rights to the general methods) and general task training has a beta in the project right now already (`generation/core_pipelines/do_grpo_rl_with_a_prompt`). The idea with general task training is that by prompting a strong generalist model to produce grades according to criteria, which will serve as a reward function, you can do reinforcement learning to improve a model on any conceivable task using just a prompt (no code changes required). Early experiments have shown promise here — a prompt to grade higher for more emotional responses produced [some good RP models](https://huggingface.co/Heralax/llama-gRPo-emotions-nothoughts), for instance — and it seems to work especially when combined with deterministic function-based rewards to prevent reward gaming. GPRO is exciting because if factual finetuning is a quality but focused tool, prompt-GRPO is a sandbox where you can make anything (also MUCH cheaper than factual finetuning). And the motivation for writing style is simple: if facts are one thing that basically everyone will want to customize, writing style is another. These are both on the **roadmap** for Augmentoolkit and I expect to have weekly updates as progress is made towards each of them.

I believe the future for Augmentoolkit and custom AI is bright. If you want updates as Augmentoolkit quickly evolves, check out [the Discord](https://discord.gg/s6PBfsaVzu) where I'll notify and tease about progress, or [the Substack](https://promptingweekly.substack.com/) for announcements, training best practices, prompting tips, AI thoughts, and more!

## What is Augmentoolkit? (Technical)

If you're into this stuff and want a clear picture on what Augmentoolkit is and how it works using the jargon we all understand, this section is for you.

Augmentoolkit is a tool, mostly used via the command line but also possessing a locally-running web interface, that runs LLM-powered pipelines to create Continued Pretraining and Supervised Fine Tuning datasets for teaching LLMs factual information. Intensive experimentation and client work has shown that LLMs can be taught to understand, recall, and use completely new factual domains (even things it has never seen at all during pretraining, such as the unreleased lore of in-development fictional universes).

The process for achieving this is essentially to create many synthetic varied representations of the documents we want the model to learn, so that the model memorizes the facts (and does not overfit on the structure). You do the continued pretraining for a long time, often around 12 epochs to get the loss really low and the documents completely memorized. Then, this knowledge is solidified by doing SFT (training on inputs; decently high batch size; about 5 to 7 epochs) on a mix of domain specific conversational data and generic data (including Capybara; the Hermes dataset; Bluemoon; Pippa; LMSys; and others). In order to ensure that the model is resistant to hallucinations, a variety of domain-specific data is used, enhancing generalization and intelligence: the model is trained to say it "does not know" about invented facts; it is trained to correct faulty assumptions; it is trained to answer followup questions. By keeping the structure of the domain data the same as the SFT data, the model's capabilities generalize nicely.

**With the ability to essentially expand an LLM's knowledge cutoff to any arbitrary area we choose,** a lot of additional options for productively using LLMs open up. With this approach, LLMs' opinions or beliefs about what is true can be arbitrarily chosen too. And all this is very cost-effective, especially considering that a custom-trained 7 billion parameter dataset generation model, built to serve as the engine for Augmentoolkit specifically, is available and can run on consumer hardware.

A domain expert can be trained for $20 or less, using only open source AI. Heck, with the right setup, both the dataset generation and model training can be done entirely locally.

## What is Augmentoolkit? (General)

If you just want to know what it does in clear and simple terms, this is the section for you.

Augmentoolkit teaches models new facts. It doesn't try to take bits of relevant text from a massive number of documents and show it to the model -- it bakes this stuff into the model's brain. This gives your AI a very good big picture understanding of the area it is working in. Think about it like hiring someone who's studied a subject for years, versus that person's twin brother who's a trivia player. Both might be pretty clever, but the trivia-playing twin brother might need a pile of textbooks as a reference to have even a half-decent conversation about your deep subject — and even then, he might not know the right pages to go to. The subject-matter expert brother, on the other hand, will be able to talk about the subject all day, and bring up relevant information on demand, because he's learned it directly.

- Maybe a company wants to create an AI expert in the problems they help solve. 
- Or maybe they want to train their AI on their internal operating procedures and run it on premises, to help their employees deal with common problems the right way.
- Perhaps a hobbyist wants an AI to understand their favorite obscure fiction.
- Maybe an advocacy organization wants an AI that agrees with its positions and can represent and explain their ideas to the uninitiated.
- A researcher exploring some niche area trains an AI on that area's foundational texts and papers, and suddenly gains a helpful personalized assistant (where before, generalist AIs had no idea about their field)

People sometimes debate whether LLMs are just fancy search. Let's consider both possibilities. If LLMs *are* just search, then Augmentoolkit is how you search new sets of documents with them, rather than being stuck looking through the same outdated snapshot of the internet. If LLMs *are something greater* then search, then Augmentoolkit is how you can leverage this new, powerful tech in areas that it simply didn't understand before.

## Ideas Presented and Hypotheses

- With continued pretraining and SFT done the right way, LLMs can be trained to understand and recall new factual domains
- Specialized AI (whether in its knowledge, opinions, or tone) is infinitely more useful to people building projects or products than its generalist counterpart.
- Synthetic data works.
- Many dataset generations are fundamentally similar in structure, so by making a core set of highly-practical abstractions and a unified interface for them to operate, dataset generation can be a core part of the open source AI revolution.
- Custom LLMs are the best way for people to start learning AI.

In addition to these, the Augmentoolkit vision believes that it helps avoid these problems:

## Arguments

> With continued pretraining and SFT done the right way, LLMs can be trained to understand and recall new factual domains

View [https://huggingface.co/Heralax/llama-Augmentoolkit-Quickstart-Factual-Demo-Example](https://huggingface.co/Heralax/llama-Augmentoolkit-Quickstart-Factual-Demo-Example) or ask my clients, or run the tool yourself.

> Specialized AI (whether in its knowledge, opinions, or tone) is infinitely more useful to people building projects or products than its generalist counterpart.

Source: my professional experience, consulting while building this tool up for about a year and a half now. Domain-specific AI is more controllable than trying to prompt something that updates unpredictably. It answers reliably with the information you want. Also, it's far cheaper and more secure than running AI off of a proprietary model provider that extorts you for tokens and eats away at your margins.

> Synthetic data works.

Most of the breakthroughs in making Augmentoolkit specialized models work well, came from adding more synthetic data of higher quality. Augmentoolkit believes that synthetic dataset generation is the future of worldwide LLM adoption+integration, as well as a core pillar of open source AI.

> Many dataset generations are fundamentally similar in structure, so by making a core set of highly-practical abstractions and a unified interface for them to operate, dataset generation can be a core part of the open source AI revolution.

It takes me about a day to make most datagen pipelines that I am interested in, now that I have all these abstractions pre-built and all the structure in place. Obviously, teaching LLMs new facts is not the only thing that people can use custom-trained AI for (though it is incredibly useful).

The hope is that with a robust codebase, simple conventions, good documentation, and now proven efficacy, Augmentoolkit can be the foundation of a community of — and eventually, an ecosystem of* — dataset generators and custom AI trainers. Together we can enable people to create AI with any capability, knowledge, beliefs, and features.

Augmentoolkit already had 1.4k stars and 195 forks before this massive update. Now, with radically overhauled performance, better abstractions, superior documentation, a GRPO pipeline which allows for reinforcement learning on arbitrary objectives, and conventions for building pipelines, Augmentoolkit is enabling the community to take it even further.

## Goals

Augmentoolkit will do facts even better, but it also has designs beyond facts.

GRPO reinforcement learning for *any* objective (whether in code or graded by an LLM prompt) already has a beta version available [here](grpo.md). Writing style and agentic training are also targets for the near future.

**Specialized AI lets anyone bridge the gap between their use case and what the models are capable of**—Augmentoolkit aims to provide this capability leap to the entire world.

To get started, clone the project and run one of the start commands (based on your OS). The interface will guide you from there.

I welcome you to use the project, play with it, experiment with it, try expanding it, and maybe contribute back with new pipelines or fixes.

#### MacOS (interface)
```
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash macos.sh
```

#### Linux (interface)
```
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash start_linux.sh
```

---

\* Despite the use of em dashes (–) here, I actually did write this myself. I've been doing this before AI started using them all over the place — long before. It's just shift+alt+dash on the mac keyboard, you know? "—" is way better looking <!--sexier--> than "()"