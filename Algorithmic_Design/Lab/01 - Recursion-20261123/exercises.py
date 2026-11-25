def power(base: int, exponent: int) -> int:
    return base**exponent

def annual_interest(principal: float, rate: float, years: int) -> float:
    if years == 0:
        return principal
    return annual_interest(principal, rate, years-1)*(1+rate)

def fibonacci(n: int) -> int:
    if n < 0:
        return
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n-1)+fibonacci(n-2)

def subset_sum(numbers: list[int], target: int) -> bool:
    n = len(numbers)
    if target == 0:
        return True
    if n < 1:
        return False
    take = subset_sum(numbers[:-1], target-numbers[-1])
    leave = subset_sum(numbers[:-1], target)
    if take | leave:
        return True
    return False