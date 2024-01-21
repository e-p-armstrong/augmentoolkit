def format_qatuples(qatuples):
    strlst = []
    for qatuple in qatuples:
        strlst.append(
            f"""Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\""""
        )
    return "\n\n".join(strlst)


def format_qatuples_noquotes(qatuples):
    strlst = []
    for idx, qatuple in enumerate(qatuples):
        strlst.append(f"""{idx + 1}. {qatuple[0]}""")
    return "\n".join(strlst)
