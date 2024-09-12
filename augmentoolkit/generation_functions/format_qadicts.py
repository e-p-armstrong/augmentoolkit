def format_qadicts(qadicts):
    strlst = []
    for qatuple in qadicts:
        strlst.append(
            f"""**QUESTION:**
{qatuple['question']}

**ANSWER:**
{qatuple['answer']}
"""
        )
    return "\n\n".join(strlst)
