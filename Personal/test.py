def stoneGameV(stoneValue) -> int:
    def rec(stoneValue):
        n = len(stoneValue)
        if n <=1:
            return 0
        
        mid = 0
        dis = float("inf")
        for i in range(1,n):
            splt_left = sum(stoneValue[:i])
            split_right = sum(stoneValue[i:])
            if abs(splt_left - split_right) < dis:
                mid = i
                dis = abs(splt_left - split_right)
        print(stoneValue[:mid])
        if sum(stoneValue[:mid]) > sum(stoneValue[mid:]):
            return rec(stoneValue[mid:]) + sum(stoneValue[mid:]) 
        elif sum(stoneValue[:mid]) <= sum(stoneValue[mid:]):
            return rec(stoneValue[:mid]) + sum(stoneValue[:mid])
    return rec(stoneValue)

stoneValue = [2,1,1]

print(stoneGameV(stoneValue))