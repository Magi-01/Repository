from q_learning_agent import process_next_item

if __name__ == "__main__":
    from item_generator import generate_item

    # Fill database with a few items
    for _ in range(5):
        generate_item()

    # Process items
    while True:
        processed = process_next_item()
        if not processed:
            print("No pending items left.")
            break

