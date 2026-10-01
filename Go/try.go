package main

import "fmt"

func main() {
	name := "Alice"
	numbers := []int{1, 2, 3, 4, 5}

	sum := 0
	for _, num := range numbers {
		sum += num
	}

	fmt.Printf("Hello, %s! The sum of the numbers is: %d\n", name, sum)
}
