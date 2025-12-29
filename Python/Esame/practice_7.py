def knapsack_problem(arr,W):
    arr_W = [i for i in range(W+1)]
    arr_w = [item[1] for item in arr]
    arr_and_W = {}
    tot_sum = 0
    for i in range(len(arr_W)):
        if arr[2][i] not in arr_and_W[1]:
            for j in range(arr_w):
                if arr_W[i] <= arr_w[j]:
                    tot_sum += arr
                    arr_and_W[arr[2][i]] = tot_sum
    return max(arr_and_W[2])
