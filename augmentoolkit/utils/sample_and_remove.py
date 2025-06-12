import random


def sample_and_remove(lst, n):
    sampled = []
    for _ in range(min(n, len(lst))):
        index = random.randrange(len(lst))
        sampled.append(lst.pop(index))
    return sampled
