def format_qatuples(qatuples):
    strlst = []
    for qatuple in qatuples:
        strlst.append(
            f"""**QUESTION:**
{qatuple[0]}

**ANSWER:**
{qatuple[1]}
"""
        )
    return "\n\n".join(strlst)
