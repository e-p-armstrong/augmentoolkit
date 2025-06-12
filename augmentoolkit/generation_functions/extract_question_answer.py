import re


def extract_question_answer(response):
    # Modified pattern to handle multi-line answers
    pattern = (
        r"### Reworded Question and Answer:\nQuestion:(.*?)\nAnswer:(.*?)(?=\n###|\Z)"
    )

    # Search with DOTALL flag to match across newlines
    match = re.search(pattern, response, re.DOTALL)

    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return question, answer
    else:
        # Try again after replacing escaped characters
        response = response.replace("\\n", "\n")
        response = response.replace('\\"', '"')
        match = re.search(pattern, response, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            return question, answer
        else:
            print("Returned none, failed to match")
            print(response)
            return None, None
