def longestCommonPrefix(strs: list[str]) -> str:
    if strs == [""] or strs == None:
            return ""
    if len(strs) < 2:
         return strs[0]
    letter = ""
    strs = strs = [[list(name), len(name)] for name in strs]
    for i in range(min([sublist[1] for sublist in strs])):
        previous = strs[0][0][i]
        for j in range(len(strs)):
            if strs[j][0][i] != previous:
                return letter
        letter+=previous
    return letter
strs = ["flower","flow","flight"]
print("\n",longestCommonPrefix(strs))