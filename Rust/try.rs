fn main() {
    let name = "Alice";
    let numbers = vec![1, 2, 3, 4, 5];

    let sum: i32 = numbers.iter().sum();
    println!("Hello, {}! The sum of the numbers is: {}", name, sum);
}