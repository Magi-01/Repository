def power(base: int, exponent: int) -> int:
    if exponent == 0: # Base case
        return 1 # Return 1 when exponent is 0

    # At each recursion, multiply the base by the result of the 
    # smaller exponent
    # In this way, you go down to the base case where the exponent 
    # is 0 and go up multiplying the base each time
    return base * power(base, exponent - 1)


def annual_interest(principal: float, rate: float, years: int) -> float:
    if years == 0: # Base case
        return principal # Return the base ammount
    
    # Apply interest for one year and recurse for the remaining years
    # Note: the main value have to be multiplied by (1 + rate) 
    # and not by rate alone
    # Otherwise, you get only the intrest rate for that year
    return annual_interest(principal * (1 + rate), rate, years - 1)


def fibonacci(n: int) -> int:
    if n == 0: # Base case
        return 0 # Return 0 for the 0th Fibonacci number
    if n == 1: # Base case
        return 1 # Return 1 for the 1st Fibonacci number

    # The recursive step follow the definition of the Fibonacci sequence: 
    # F(n) = F(n-1) + F(n-2)
    return fibonacci(n - 1) + fibonacci(n - 2)


def subset_sum(numbers: list[int], target: int) -> bool:
    if target == 0: # Base case: target is 0
        # Return True because the empty subset sums to 0
        return True 
    if not numbers: # Base case: no numbers left
        # Return False because we can't form the target with no numbers
        return False 

    # The recursion tree have to be splitted at each step:
    # In the first branch, we include the first number in the subset and 
    # the target is reduced by that number
    first = numbers[0]
    # In the second branch, we exclude the first number (of that sequence) 
    # and keep looking for the target in the remaining set
    remaining = numbers[1:]
    # Since we want AT LEAST one subset to sum, we use OR to combine the results
    return subset_sum(remaining, target - first) or subset_sum(remaining, target)

