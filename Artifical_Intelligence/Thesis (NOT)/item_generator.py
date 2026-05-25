import time
import random
from db_utils import insert_item

ITEM_TYPES = ['A','B','C']
BIN_MAP = {'A':1, 'B':2, 'C':3}
MIN_DEADLINE = 10
MAX_DEADLINE = 30
INTERVAL = 5

def generate_item():
    item_type = random.choice(ITEM_TYPES)
    correct_bin = BIN_MAP[item_type]
    deadline = random.randint(MIN_DEADLINE, MAX_DEADLINE)
    drop_bin = correct_bin  # Assign drop-off bin explicitly

    # Insert with status "pending"
    insert_item(item_type, correct_bin, deadline, drop_bin, status="pending")
    print(f"[NEW ITEM] {item_type} -> Bin {correct_bin}, Deadline {deadline}s, drop_bin={drop_bin}")

if __name__ == "__main__":
    try:
        while True:
            generate_item()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Item generator stopped")
