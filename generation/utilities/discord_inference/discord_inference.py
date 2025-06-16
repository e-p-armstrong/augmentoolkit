# Started by a utility
# Runs the model to power a specified discord bot
# It must still be added to the server by the user of course.

import re
import asyncio
import subprocess
import sys
import os
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
import discord
import httpx
from openai import AsyncOpenAI
import yaml

# Import the server functions
from generation.utilities.llm_server.llm_server import llm_server
from generation.utilities.rag_server.rag_server import rag_server
from redis_config import set_progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    next_msg: Optional[discord.Message] = None

    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def send_chat_request(messages: list, port: int = 8003) -> str:
    """
    Send a chat request to a local server with a /generate endpoint.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        port: Port number of the server (default: 8003)

    Returns:
        String response from the server

    Raises:
        httpx.HTTPError: If the request fails
        Exception: If the server returns an error
    """
    url = f"http://localhost:{port}/generate"

    # Prepare the request payload
    payload = {"messages": messages}

    try:
        async with httpx.AsyncClient(timeout=400.0) as client:  # 5 minute timeout
            response = await client.post(url, json=payload)
            response.raise_for_status()  # Raises an exception for HTTP error codes

            # The server returns the response as a JSON-encoded string
            # We need to decode escaped characters like \n back to actual newlines
            response_text = response.text.strip(
                '"'
            )  # Remove quotes if the response is JSON-encoded string
            # Decode common escape sequences
            response_text = response_text.encode().decode("unicode_escape")

            return response_text

    except httpx.TimeoutException:
        logging.error(f"Request to {url} timed out")
        raise Exception("Request timed out")
    except httpx.HTTPError as e:
        logging.error(f"HTTP error when calling {url}: {e}")
        raise Exception(f"HTTP error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error when calling {url}: {e}")
        raise Exception(f"Unexpected error: {e}")


ALLOWED_FILE_TYPES = ["text"]

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()


async def discord_inference(
    bot_token: str,
    client_id: int,
    max_text: 100000,
    max_messages: 25,
    allow_dms: bool,
    allowed_user_ids,
    blocked_user_ids,
    allowed_role_ids,
    blocked_role_ids,
    allowed_channel_ids,
    blocked_channel_ids,
    inference_server_args,
    inference_server="normal",
    port=8003,
    max_message_nodes=100,
    status_message: str = None,
    task_id="11037",
    **kwargs,
):

    # Start the inference server with subprocess if needed
    server_process = None
    server_task = None

    if inference_server == "normal":
        print("Starting LLM server in background...")
        # Start llm_server as a background task
        server_task = asyncio.create_task(llm_server(**inference_server_args))
        # Give the server time to start up
        await asyncio.sleep(5)

    elif inference_server == "rag":
        print("Starting RAG server in background...")
        # Start rag_server as a background task
        server_task = asyncio.create_task(rag_server(**inference_server_args))
        # Give the server time to start up
        await asyncio.sleep(10)  # RAG server needs more time for initialization

    elif inference_server == "none":
        print(
            "No inference server will be started - assuming external server is running"
        )
    else:
        raise ValueError(f"Unknown inference_server type: {inference_server}")

    try:
        print("EXTRA ARGS")
        print(kwargs)

        print(
            f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n"
        )
        print("Paste this into a browser to invite your bot to a server you control!")
        print(
            "Be sure that you have first created a bot using the Discord Developer portal https://discord.com/developers/applications"
        )
        print(
            "You need to get the bot token and client ID and give it permissions: bot, view channels, send messages, read message history"
        )

        intents = discord.Intents.default()
        intents.message_content = True
        activity = discord.CustomActivity(name=(status_message or "")[:128])
        discord_client = discord.Client(intents=intents, activity=activity)

        httpx_client = httpx.AsyncClient()

        msg_nodes = {}
        last_task_time = 0

        @discord_client.event
        async def on_message(new_msg):
            print("DEBUG: on_message")
            nonlocal msg_nodes, last_task_time, allow_dms, allowed_channel_ids, allowed_role_ids, allowed_user_ids, blocked_channel_ids, blocked_role_ids, blocked_user_ids, max_text, bot_token, max_messages, port, max_message_nodes

            is_dm = new_msg.channel.type == discord.ChannelType.private

            # Normalize the message content
            content = new_msg.content.strip()

            # Early return for bot messages
            if new_msg.author.bot:
                return

            ## NOTE Handle whether to reply or not
            # Get bot's member object for this guild
            bot_member = (
                new_msg.guild.get_member(discord_client.user.id)
                if new_msg.guild
                else None
            )
            bot_roles = bot_member.roles if bot_member else []

            # Check all possible mention formats
            is_mentioned = any(
                [
                    discord_client.user in new_msg.mentions,  # Direct user mention
                    f"<@{discord_client.user.id}>" in content,  # ID mention
                    f"<@!{discord_client.user.id}>" in content,  # Nickname mention
                    "@" + discord_client.user.name in content,  # Name mention
                    # Role mentions
                    any(f"<@&{role.id}>" in content for role in bot_roles),
                ]
            )

            role_mentions = (
                new_msg.role_mentions
            )  # This will give you the role objects that were mentioned
            print(
                "DEBUG: role mentions: ",
                [(role.name, role.id) for role in role_mentions],
            )

            if not is_dm and not is_mentioned:
                print("DEBUG: not is_dm and not is_mentioned")
                print("DEBUG: new_msg.content: ", new_msg.content)
                print("DEBUG: discord_client.user.id: ", discord_client.user.id)
                print("DEBUG: is_dm: ", is_dm)
                print("DEBUG: is_mentioned: ", is_mentioned)
                print(
                    "DEBUG: bot roles: ",
                    [f"{role.name}: {role.id}" for role in bot_roles],
                )
                return

            role_ids = tuple(role.id for role in getattr(new_msg.author, "roles", ()))
            channel_ids = tuple(
                id
                for id in (
                    new_msg.channel.id,
                    getattr(new_msg.channel, "parent_id", None),
                    getattr(new_msg.channel, "category_id", None),
                )
                if id
            )

            allow_all_users = (
                not allowed_user_ids
                if is_dm
                else not allowed_user_ids and not allowed_role_ids
            )
            is_good_user = (
                allow_all_users
                or new_msg.author.id in allowed_user_ids
                or any(id in allowed_role_ids for id in role_ids)
            )
            is_bad_user = (
                not is_good_user
                or new_msg.author.id in blocked_user_ids
                or any(id in blocked_role_ids for id in role_ids)
            )

            print("DEBUG: is_good_user: ", is_good_user)
            print("DEBUG: is_bad_user: ", is_bad_user)

            allow_all_channels = not allowed_channel_ids
            is_good_channel = (
                allow_dms
                if is_dm
                else allow_all_channels
                or any(id in allowed_channel_ids for id in channel_ids)
            )
            is_bad_channel = not is_good_channel or any(
                id in blocked_channel_ids for id in channel_ids
            )

            if is_bad_user or is_bad_channel:
                print("DEBUG: is_bad_user or is_bad_channel, returning")
                return

            accept_usernames = False  # NOTE will be an arg

            # use_plain_responses = cfg["use_plain_responses"] # We will need an option to remove the thought process part of the AI's response.
            max_message_length = 2000  # if use_plain_responses else (4096 - len(STREAMING_INDICATOR)) # TODO we need a nicer way to format past the 2k character limit, format the splits. But not urgent.

            # Build message chain and set user warnings
            messages = []
            user_warnings = set()
            curr_msg = new_msg
            print("DEBUG: curr_msg: ", curr_msg)
            while curr_msg != None and len(messages) < max_messages:
                curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

                async with curr_node.lock:
                    if curr_node.text == None:
                        # cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()
                        mention_pattern = re.compile(r"<@!?&?[0-9]+>")
                        cleaned_content = re.sub(
                            mention_pattern, "", curr_msg.content
                        ).strip()

                        good_attachments = {
                            type: [
                                att
                                for att in curr_msg.attachments
                                if att.content_type and type in att.content_type
                            ]
                            for type in ALLOWED_FILE_TYPES
                        }

                        print(good_attachments)

                        curr_node.text = "\n".join(
                            ([cleaned_content] if cleaned_content else [])
                            + [
                                embed.description
                                for embed in curr_msg.embeds
                                if embed.description
                            ]
                            + [
                                (await httpx_client.get(att.url)).text
                                for att in good_attachments["text"]
                            ]
                        )

                        curr_node.role = (
                            "assistant"
                            if curr_msg.author == discord_client.user
                            else "user"
                        )

                        curr_node.user_id = (
                            curr_msg.author.id if curr_node.role == "user" else None
                        )

                        curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(
                            len(att_list) for att_list in good_attachments.values()
                        )

                        try:
                            if curr_msg.reference:  # Only follow explicit replies
                                try:
                                    referenced_msg = (
                                        curr_msg.reference.cached_message
                                        or await curr_msg.channel.fetch_message(
                                            curr_msg.reference.message_id
                                        )
                                    )
                                    # Only include the referenced message if it's from the bot or if the current message mentions the bot
                                    if (
                                        referenced_msg.author == discord_client.user
                                        or discord_client.user.mentioned_in(curr_msg)
                                    ):
                                        curr_node.next_msg = referenced_msg
                                except (discord.NotFound, discord.HTTPException):
                                    logging.exception(
                                        "Error fetching referenced message"
                                    )
                                    curr_node.fetch_next_failed = True
                            else:
                                print("DEBUG: else")
                                is_public_thread = (
                                    curr_msg.channel.type
                                    == discord.ChannelType.public_thread
                                )
                                next_is_parent_msg = (
                                    not curr_msg.reference
                                    and is_public_thread
                                    and curr_msg.channel.parent.type
                                    == discord.ChannelType.text
                                )

                                if next_msg_id := (
                                    curr_msg.channel.id
                                    if next_is_parent_msg
                                    else getattr(curr_msg.reference, "message_id", None)
                                ):
                                    if next_is_parent_msg:
                                        curr_node.next_msg = (
                                            curr_msg.channel.starter_message
                                            or await curr_msg.channel.parent.fetch_message(
                                                next_msg_id
                                            )
                                        )
                                    else:
                                        curr_node.next_msg = (
                                            curr_msg.reference.cached_message
                                            or await curr_msg.channel.fetch_message(
                                                next_msg_id
                                            )
                                        )

                        except (discord.NotFound, discord.HTTPException):
                            logging.exception(
                                "Error fetching next message in the chain"
                            )
                            curr_node.fetch_next_failed = True

                    content = (
                        curr_node.text
                    )  # [:max_text] TODO do we really need to truncate here?

                    if content != "":
                        message = dict(content=content, role=curr_node.role)
                        if accept_usernames and curr_node.user_id != None:
                            message["name"] = str(curr_node.user_id)

                        messages.append(message)

                    if len(curr_node.text) > max_text:
                        user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                    if curr_node.has_bad_attachments:
                        user_warnings.add(
                            "⚠️ Unsupported attachments (quite possibly images)"
                        )
                    if curr_node.fetch_next_failed or (
                        curr_node.next_msg != None and len(messages) == max_messages
                    ):
                        user_warnings.add(
                            f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}"
                        )

                    # print("DEBUG: curr_msg: ", curr_msg)
                    curr_msg = curr_node.next_msg

            logging.info(
                f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}"
            )

            # No system prompt in config. What we do, is, we run this alongside one of the servers.  You see, the api it will use -- is an augmentoolkit API. We give it a port and it points at like 8003 or something.

            # Generate and send response message(s) (can be multiple if response is long)
            response_msgs = []
            response_contents = []
            prev_chunk = None
            edit_task = None

            embed = discord.Embed()
            for warning in sorted(user_warnings):
                embed.add_field(name=warning, value="", inline=False)

            try:
                async with new_msg.channel.typing():
                    final_content = await send_chat_request(
                        messages=messages[::-1], port=port
                    )  #  await engine_wrapper.submit_chat(messages[::-1], sampling_params) # reverses the messages list

                    # Split response into chunks of max Discord message length (2000 chars)
                    chunk_size = 1999  # Leave 1 char buffer
                    response_contents = [
                        final_content[i : i + chunk_size]
                        for i in range(0, len(final_content), chunk_size)
                    ]

                    for content in response_contents:
                        reply_to_msg = (
                            new_msg if response_msgs == [] else response_msgs[-1]
                        )
                        response_msg = await reply_to_msg.reply(
                            content=content, suppress_embeds=True
                        )
                        response_msgs.append(response_msg)

                        msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                        await msg_nodes[response_msg.id].lock.acquire()

            except Exception:
                logging.exception("Error while generating response")

            for response_msg in response_msgs:
                msg_nodes[response_msg.id].text = "".join(response_contents)
                msg_nodes[response_msg.id].lock.release()

            # Delete oldest MsgNodes (lowest message IDs) from the cache
            if (num_nodes := len(msg_nodes)) > max_message_nodes:
                for msg_id in sorted(msg_nodes.keys())[: num_nodes - max_message_nodes]:
                    async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                        msg_nodes.pop(msg_id, None)

        set_progress(
            task_id=task_id, progress=1.0, message="Discord client about to start!"
        )
        await discord_client.start(bot_token)

    finally:
        # Clean up the server if it was started
        if server_task and not server_task.done():
            print("Shutting down inference server...")
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                print("Inference server shut down successfully")
            except Exception as e:
                print(f"Error during server shutdown: {e}")
