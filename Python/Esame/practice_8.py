"""
A[-n,...,1,...,n] full of distinct  integer numbers
find the number of sums of two numbers whoose result is 0
that means find a number and it's oppsite
"""
def find_array_sum_zero(arr):
    count = {}
    summ = 0

    # Count occurrences of each element in the array
    for num in arr:
        count[num] = count.get(num, 0) + 1

    # Check for pairs with sum zero
    for num in arr:
        if -num in count:
            summ += count[-num]

    # Each pair is counted twice, so divide by 2 to get the total count
    return summ // 2

arr = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
print(find_array_sum_zero(arr))