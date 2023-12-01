
def format_qatuples(qatuples):
    strlst = []
    for qatuple in qatuples:
        strlst.append(f"""Question: \"\"\"{qatuple[0]}\"\"\"
Answer: \"\"\"{qatuple[1]}\"\"\"""")
    return "\n".join(strlst)