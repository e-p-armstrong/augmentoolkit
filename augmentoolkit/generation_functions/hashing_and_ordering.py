import hashlib


def hash_input_list(
    input_list=[], key_to_hash_with="text"
):  # turns things like sentence chunks, paragraphs processed, into wide dicts with each item -> a hashed unique key pointing at a dict with that item's original values

    if not input_list:
        return {}

    # Sort the input list by the key_to_hash_with value to ensure consistent ordering
    sorted_input_list = sorted(
        input_list,
        key=lambda x: (
            str(x.get(key_to_hash_with, str(x))) if key_to_hash_with in x else str(x)
        ),
    )

    hashed_dict = {}

    for idx, item in enumerate(sorted_input_list):
        # Create a hash from just the specified key's value (no index dependency)
        # if key_to_hash_with and key_to_hash_with in item:
        #     hash_input = str(item[key_to_hash_with])
        # else:
        # If key_to_hash_with is not provided or not in item, use the whole item
        # hash_input = str(item)

        # Create a hash using MD5
        # hash_obj = hashlib.md5(hash_input.encode())
        # hash_key = hash_obj.hexdigest()

        # Store the item in the dictionary with the hash as the key
        hashed_dict[str(idx)] = (
            item  # instead of hashing we will just use the stringified index. It's still unique; it contains more information; it avoids the problem of lists where the indicies can be flipped (the new "id" does not change); it does not cause resuming issues with slightly different inputs or other random annoying causes like the current thing does
        )

    return hashed_dict


# the key to use as the hash for identification, or rather, the key to search through... no it does not work like that, we don't even need this we just search the dict for the given key in the key,value tuple returned by .items(). We literally just replace idx with key and swap out the method.
