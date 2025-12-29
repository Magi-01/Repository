# Name: Fadhla Mohamed
# Sirname: Mutua
# Matricola: SM3201434

class Assembler():
    def assemble_manually(self, file_data):
        # Define the instruction set with corresponding opcodes
        instruction_set = {
            "ADD": 1,
            "SUB": 2,
            "STA": 3,
            "LDA": 5,
            "BRA": 6,
            "BRZ": 7,
            "BRP": 8,
            "INP": 9,
            "OUT": 10,
            "HLT": 0,
        }

        memory_address = 0  # Tracks the memory address for instructions
        label_addresses = {}  # Stores label names and their corresponding memory addresses

        # First pass: Identify labels and store their addresses
        for line in file_data:
            tokens = line.split()  # Tokenize the instruction line
            if not tokens:
                continue  # Skip empty lines

            first_token = tokens[0].upper()  # Convert to uppercase for consistency (assuming case-insensitivity)

            # If the first token is not a known instruction and not "DAT", treat it as a label
            if first_token not in instruction_set and first_token != "DAT":
                label_addresses[first_token] = memory_address  # Store the label's address
                tokens = tokens[1:]  # Remove the label from tokens

            # If there are remaining tokens and it's a valid instruction or "DAT", increment memory address
            if tokens and (tokens[0].upper() in instruction_set or tokens[0].upper() == "DAT"):
                memory_address += 1  # Move to the next memory location

        memory_address = 0  # Reset memory address for the second pass

        # Second pass: Translate instructions into machine code
        for line in file_data:
            tokens = line.split()
            if not tokens:
                continue  # Skip empty lines

            first_token = tokens[0].upper()

            # If the first token is a label, remove it from the tokens
            if first_token in label_addresses:
                tokens = tokens[1:]

            if not tokens:
                continue  # Skip lines that had only a label

            instruction = tokens[0].upper()  # Get the instruction

            if instruction == "DAT":
                # Store a numeric value in memory, default to 0 if no value is provided
                value = int(tokens[1]) if len(tokens) > 1 else 0
                self.memory[memory_address] = value

            elif instruction in ["INP", "OUT", "HLT"]:
                # Handle special cases where no operand is needed
                if instruction == "INP":
                    self.memory[memory_address] = 901  # INP has opcode 9 with address 01
                elif instruction == "OUT":
                    self.memory[memory_address] = 902  # OUT has opcode 9 with address 02
                else:
                    self.memory[memory_address] = 0  # HLT is just 0

            else:
                # General instruction processing
                opcode = instruction_set.get(instruction, -1)
                if opcode == -1:
                    raise ValueError(f"Unknown instruction: {instruction}")

                # Ensure an operand exists for instructions that require it
                operand = tokens[1] if len(tokens) > 1 else None
                if operand is None and tokens[0] not in ['OUT', 'HLT']:
                    raise ValueError(f"Missing operand for instruction: {instruction}")

                # Pass operand to a memory address
                if operand.isdigit():
                    address = int(operand)  # Direct numeric address
                elif operand.upper() in label_addresses:
                    # Pass label to address
                    address = label_addresses[operand.upper()]  
                else:
                    raise ValueError(f"Undefined label or address: {operand}")

                # Store the final machine code instruction in memory
                self.memory[memory_address] = opcode * 100 + address  
                # Format: OPCODE + Address
                # With OPCODE a hundredth number (so < 1000) and 0 < adress < 99

            memory_address += 1  # Move to the next memory slot
