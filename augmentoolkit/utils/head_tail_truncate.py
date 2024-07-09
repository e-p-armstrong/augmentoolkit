import math


def head_tail_truncate(text, max_length=510):
    """
    Truncate the text using the head+tail method.
    Keep the first head_length characters and the last (max_length - head_length) characters.
    """
    
    head_length = math.floor(0.2*max_length)
    tail_length = max_length - head_length
    if len(text) <= max_length:
        return text
    return text[:head_length] + text[-(tail_length - head_length):]
