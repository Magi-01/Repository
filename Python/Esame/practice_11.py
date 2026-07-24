import numpy as np
class DP():
    def __init__(self):
        pass
    def levenshteinRecursive(self,word,word_to_replace,m,n,d={}):
        key = m, n
        if n == 0:
            return m
        if m == 0:
            return n
        if key in d:
            return d[key]
        
        if word[m - 1] == word_to_replace[n - 1]:
            return DP.levenshteinRecursive(self, word, word_to_replace, m - 1, n - 1)
        d[key] = 1 + min(
            # Insert
        DP.levenshteinRecursive(self, word, word_to_replace, m, n - 1),
            min(
                # Remove
                DP.levenshteinRecursive(self, word, word_to_replace, m - 1, n),
                # Replace
                DP.levenshteinRecursive(self, word, word_to_replace, m - 1, n - 1)
            )
        )

        return d[key]
    
    def knapSack(self,N,W,p,wt):
        if W==0 or N==0:
            return 0
        
        if t[N][W] != -1:
            return t[N][W]
        
        if (wt[N-1] > W):
            t[N][W] = DP.knapSack(self,N-1,W,p,wt)
            return t[N][W]
        
        else:
            t[N][W] = max(
                p[N-1] + 
                DP.knapSack(self,N-1,W-wt[N-1],p,wt),
                DP.knapSack(self,N-1,W,p,wt)
                )
            return t[N][W]
        
    def longest_common_subdivision(self,word):
        pass

# Levenshtein sum
trial = DP()
print(trial.levenshteinRecursive("casa","",len("casa"),len("")))

# KnapSack
profit = [60, 100, 120]
weight = [10, 20, 30]
W = 50
N = len(profit)
t = [[-1 for i in range(W + 1)] for j in range(N + 1)]
print(trial.knapSack(N, W, profit, weight))