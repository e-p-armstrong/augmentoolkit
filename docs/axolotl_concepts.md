Augmentoolkit gives you datasets. Typically what you do with datasets is you train LLMs on them. [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is a good way to do that.

When it comes to changing axolotl configs, the Axolotl docs [here](https://docs.axolotl.ai/docs/config.html) are a good way to get a full overview. However that explanation gives you *every single option* and is also not terribly deep at explaining the actual impact some of these things have.

So, if you want to potentially improve your model training and get a better idea of what these settings in your training config are, and how to use them, read on.

This doesn't cover every field, but rather, the important ones you might want to change.

---

## Mistral Derived
```yaml
is_mistral_derived_model: True|False
```

This one is important if you are traininng a model like Mistral 7b v0.2. Those models have a different padding side than most models. If you're not sure what that means, don't worry -- just remember to turn this on if tuning an older mistral model.

## Data

```yaml
datasets:
```

This is where you point Axolotl at the datasets you're training your model on. Massively important! Very little has such an impact on a model's performance as data does.

If you are training on completion data (as I recommend you do -- due to obscure training dynamics, the models seem to learn things really well when you train on the inputs too) then format data like this:

```yaml
- path: [path to your dataset].jsonl
  type: completion
```

And the data is formatted very simply in this case:
```json
{"text": "Some text you want the model to be trained on"}
{"text": "More text"}
....
```

If you're training on data where you don't show certain parts of the data to the model, but it's not a typical conversational back-and-forth (take the RAG pipeline or correction pipeline as good examples of this) then you'll want input-output data.

```yaml
- path: axolotl_rag_conversations_facts.jsonl
  type: input_output
```

```json
[
  {
    "segments": [
      {
        "label": true,
        "text": "BOS TOKEN some text you want to train on"
      },
      {
        "label": false,
        "text": "Some text you do not want to train on"
      },
      {
        "label": true,
        "text": "Some text that you DO want to train on. Notice how label controls that. Also, be careful with this format -- you need to manually add the BOS and EOS tokens, nothing is added for you automatically when training input_output. EOS TOKEN"
      }
    ]
  },
  ...more objects...
```

If you want to train conversationally (like passing in a list of messages) it is annoying but here is how to do it roughly speaking:
```yaml
- path: openhermes2_5_shard_01.json
    type: chat_template
    chat_template: chatml
    field_messages: conversations
    message_field_role: from
    message_field_content: value
    roles:
      user:
        - human
      assistant:
        - gpt
      system:
        - system
```

```json
{"conversations": [{"from": "system", "value": "system prompt"}, {"from": "human", "value": "human message"}, {"from": "gpt", "value": "ai message"}, {"from": "human", "value": "human message 2"}, {"from": "gpt", "value": "ai message 2"}]}
{"conversations": [{"from": "system", "value": "system prompt for conversation 2"}, {"from": "human", "value": "human message"}, {"from": "gpt", "value": "ai message"}, {"from": "human", "value": "human message 2"}, {"from": "gpt", "value": "ai message 2"}]}
...more objects...
```
I have observed worse results using this compared to completion data. Furthermore, the more arbitrary of a chat format you use/the less it resembles the pretraining text, the overall worse I find the results to be. Chat format is hugely important which is why I like to customize mine and I use my own special Human: **Finished.** format. You can do as you want to however, and for compatibility sometimes you might want to train with instruct mode.

## Sequence Length

```yaml
sequence_len: 5000
```

Sequence length is the context length you are training your model at. Higher numbers will reduce the number of optimization steps, potentially resulting in worse factual learning if your dataset is small. Most settins tweaking you do, for datagen configuration or training configuration, will be balanced around the number of optimization steps in your training run (the number you see next to the progress bar when training starts -- this represents how many times the LLM's brain moves towards understanding the dataset better, and if the number is too low, the model won't get the chance to move enough to properly learn your dataset).

Higher sequence length will also cost more VRAM, meaning you may need more powerful GPUs. However, having more context length can be very valuable, especially for applications like RAG. So, it's up to you to determine what's a good value for your usecase.

## Effective Batch Size

```
gradient_accumulation_steps: 75
micro_batch_size: 2
```

Higher effective batch size makes the model generalize more but reduces the number of optimization steps and can harm its ability to memorize specific things/can cause the facts to blend together if it is far too high.

Effective batch size is determined by the simple formula:

gradient accumulation steps * micro batch size * number of GPUs used for training = effective batch size.

micro batch size speeds up training but costs VRAM. Gradient accumulation steps kinda slightly slows down training a bit, but does not cost any more VRAM. More GPUs increases the VRAM you have but costs more money. Be mindful of the more gpus = higher effective batch size rule, because the same config run on machines with different GPU counts will have wildly different results.

If you have too few optimizer steps you can decrease the effective batch size to a point, and if you have a lot of data and some spare VRAM you can increase it, but since this is a parameter that has been reached through a decent amount of trial and error, it is advised to be cautious when changing it. Nevertheless, you will have to change this to keep it ~150 if you are increasing the number of GPUs used to train. Remember the formula,

gradient accumulation steps * micro batch size * number of GPUs used for training = effective batch size.

## Epoch count

```
num_epochs: 7
```
The number of training epochs to use. Generally you want quite a few of these for factual learning. You can stick with the defaults in the configs for the most part, but if training is estimated to take a LONG time, you may consider reducing this slightly.

Crank it too high, when combined with noisy-embedding finetuning, however, and your model will get very stupid. So do be careful. Everything in ML is a loaded gun if you push it too far.

## Huggingface Settings

```
hub_model_id: HFUsername/model-name-here
hub_strategy: all_checkpoints
```

Keep running out of space on HuggingFace? Change the hub strategy. Parameter options viewable [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.set_push_to_hub.strategy)

#### Is something still on your mind?

Got questions about model training still? Feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask your questions! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.