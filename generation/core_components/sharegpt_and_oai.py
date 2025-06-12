# def convert_oai_to_sharegpt(oai_json):
#     return [{
#         "conversations": [
#             {
#                 "from": msg["role"],
#                 "value": msg["content"]
#             }
#             for msg in message_list
#         ]
#     } for message_list in oai_json]

# def convert_sharegpt_to_oai(sharegpt_json):
#     return [
#         [
#             {
#                 "role": msg["from"],
#                 "content": msg["value"]
#             }
#             for msg in item["conversations"]
#         ]
#         for item in sharegpt_json
#     ]


# def convert_single_sharegpt_to_oai(sharegpt_json):
#     return {"messages": [
#         {
#             "role": msg["from"],
#             "content": msg["value"]
#         }
#         for msg in sharegpt_json["conversations"]
#     ]}

# def convert_single_oai_to_sharegpt(oai_json):
#     return {"conversations": [
#         {
#             "from": msg["role"],
#             "value": msg["content"]
#         }
#         for msg in oai_json["messages"]
#     ]}


def rename_oai_messages_to_sharegpt(oai_list):
    conversations = []
    for message in oai_list:
        conversations.append(
            {
                "from": (
                    "human"
                    if message["role"] == "user"
                    else (
                        "gpt"
                        if message["role"] == "assistant"
                        else "system" if message["role"] == "system" else "unknown"
                    )
                ),
                "value": message["content"],
            }
        )
    return conversations


def rename_sharegpt_conversations_to_messages(sharegpt_list):
    messages = []
    for conversation in sharegpt_list:
        messages.append(
            {
                "role": (
                    "user"
                    if conversation["from"] == "human"
                    else (
                        "assistant"
                        if conversation["from"] == "gpt"
                        else "system" if conversation["from"] == "system" else "unknown"
                    )
                ),
                "content": conversation["value"],
            }
        )
    return messages


def convert_oai_to_sharegpt(oai_list):
    # takes a list of objects with a "messages" key, each of which is a list of objects with a "role" and "content" key
    sharegpt_list = []
    for oai_item in oai_list:
        sharegpt_list.append(
            {"conversations": rename_oai_messages_to_sharegpt(oai_item["messages"])}
        )
    return sharegpt_list


def convert_sharegpt_to_oai(sharegpt_list):
    oai_list = []
    for sharegpt_item in sharegpt_list:
        oai_list.append(
            {
                "messages": rename_sharegpt_conversations_to_messages(
                    sharegpt_item["conversations"]
                )
            }
        )
    return oai_list
