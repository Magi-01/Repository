file_path = '/home/fadhla/Documents/numbers_1_to_1000.txt'

with open(file_path, 'w') as f:
    for i in range(1, 1001):
        f.write(f"{i}\n")

file_path