# Host your Custom AI on Discord for your Friends to Chat With!

This **utility pipeline** is subtly one of the coolest parts of Augmentoolkit. With it, you can put one of your custom models on a Discord server you have permissions on, letting your friends (or community) chat with the cool thing you've built and showing off your AI talent.

Here's how it works:
1. Go to [the Discord developer portal](https://discord.com/developers/applications) and make a new application
1. Fill out a config for this utility pipeline.
    - Add your bot_token (found at [https://discord.com/developers/applications/yourappid/information](https://discord.com/developers/applications/yourappid/information))
    - Add your client ID (found at [https://discord.com/developers/applications/yourappid/information](https://discord.com/developers/applications/yourappid/information))
    - Add the path to your model, prompt.txt, and template.txt just like in [the normal server](./basic_server.md) or [the RAG server](./rag_server.md)
    - Specify the `inference_server` arg to be "normal" if you want the basic server; "rag" if you want a RAG server, and "none" if you want to manually run a different server that will listen on port 8003.
1. Add the proper permissions to the bot on the Discord side of things
    - Enable message content intent in the Bot page
1. Invite the bot to your server
    - https://discord.com/developers/applications/YOURBOTID/oauth2
    - Scroll down and give it the "Bot" permission as well as "applications.commands"
    - Scroll down further and give it the "Read Message History", "Send Messages", and "View Channels" and perhaps "Send Messages in Threads".
    - Copy the generated URL.
    - Paste it into your browser in a new tab.
    - Select the server you want to invite it to, and invite it.
1. Run the Discord utility pipeline
    - **Your bot can now be interacted with by other people!** Try @ mentioning it with a question†.
    - Replies continue the conversation, new @ mentions start a new one. Same as [llmcord](https://github.com/jakobdylanc/llmcord), which this utility pipeline is forked from and modified extensively from.
    - Instructions on how to use the interface and CLI can be found in [the quickstart](./quickstart.md) or, for more detail, in [the interface flows doc](./interface_flows.md) and the [cli flows doc](./CLI_flows.md).
    - The bot will only be available on Discord as long as both windows are running on SOME computer. If you terminate the windows, to run the bot again just rerun both pipelines again (no need to go through te whole Discord application portal a second time).
    
Documentation for the config fields is in the comments inside the config itself (`discord_inference_config.yaml` inside `external_configs/`). They are completely self-explanatory except for maybe max_text which you probably should not touch anyway (max_text is simply the maximum amount of text that the model can be passed in a single chain of messages).

† Note that @ mentions only work if typed, not if copy-pasted. Super-weird quirk of discord -- if you copy-paste then the message has the literal text of the @ mention, whereas if you type it out and hit enter then your message will contain @<USERID> and will be properly recognized by the appliation code.