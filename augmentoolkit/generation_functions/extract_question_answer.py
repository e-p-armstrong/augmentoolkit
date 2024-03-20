import re


def extract_question_answer(response):
    # Define the regex pattern to match the question and answer
    pattern = r"### Question Rewording \(using text details as reference\):\nQuestion: (.+?)\nAnswer: (.+)"

    # Search for the pattern in the response
    match = re.search(pattern, response)

    # Extract and return the question and answer if a match is found
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    else:
        response = response.replace("\\n","\n")
        response = response.replace("\\\"","\"")
        match = re.search(pattern, response)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            return question, answer
        else:
            print("Returned none, failed to match")
            print(response)
            return None, None
