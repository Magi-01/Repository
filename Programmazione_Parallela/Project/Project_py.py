from collections import deque

class Assembler():
    def assemble_manually(self, file_data):
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

        memory_address = 0
        label_addresses = {}

        for line in file_data:
            tokens = line.split()
            if not tokens:
                continue

            first_token = tokens[0].upper()

            if first_token not in instruction_set and first_token != "DAT":
                label_addresses[first_token] = memory_address
                tokens = tokens[1:]

            if tokens and tokens[0].upper() in instruction_set or tokens[0].upper() == "DAT":
                memory_address += 1

        memory_address = 0
        for line in file_data:
            tokens = line.split()
            if not tokens:
                continue

            first_token = tokens[0].upper()

            if first_token in label_addresses:
                tokens = tokens[1:]

            if not tokens:
                continue

            instruction = tokens[0].upper()

            if instruction == "DAT":
                value = int(tokens[1]) if len(tokens) > 1 else 0
                self.memory[memory_address] = value

            elif instruction in ["INP", "OUT", "HLT"]:
                if instruction == "INP":
                    self.memory[memory_address] = 901
                elif instruction == "OUT":
                    self.memory[memory_address] = 902
                else:
                    self.memory[memory_address] = 0

            else:
                opcode = instruction_set.get(instruction, -1)
                if opcode == -1:
                    raise ValueError(f"Unknown instruction: {instruction}")

                operand = tokens[1] if len(tokens) > 1 else None
                if operand is None and tokens[0] not in ['OUT', 'HLT']:
                    raise ValueError(f"Missing operand for instruction: {instruction}")

                if operand.isdigit():
                    address = int(operand)
                elif operand.upper() in label_addresses:
                    address = label_addresses[operand.upper()]
                else:
                    raise ValueError(f"Undefined label or address: {operand}")

                self.memory[memory_address] = opcode * 100 + address

            memory_address += 1

class LMC(Assembler):
    def __init__(self):
        self.program_counter = 0
        self.accumulator = 0
        self.memory = [0] * 100
        self.input_queue = deque()
        self.output_queue = deque()
        self.flag = False
        self.result_index = 0

    def load_instructions(self, file_data):
        self.assemble_manually(file_data)


    def run(self, file_data, input_data=None):
        self.file = file_data
        if input_data:
            for inp in input_data:
                self.input_queue.append(inp)
        else:
            self.input_queue = None
        
        self.load_instructions(self.file)

        def check_flag():
            self.flag = False
            if self.accumulator >= 1000:
                self.flag = True
                self.accumulator = self.accumulator % 1000
            elif self.accumulator < 0:
                self.flag = True
                self.accumulator = self.accumulator % 1000

        def lmcAdd():
            self.accumulator += self.memory[self.address_register]
            check_flag()
            return True

        def lmcSub():
            self.accumulator -= self.memory[self.address_register]
            check_flag()
            return True

        def lmcSTA():
            self.memory[self.address_register] = self.accumulator
            return True

        def lmcLDA():
            self.accumulator = self.memory[self.address_register]
            check_flag()
            return True

        def lmcBRA():
            self.program_counter = self.address_register
            return True

        def lmcBRZ():
            if self.accumulator == 0 and not self.flag:
                self.program_counter = self.address_register
            return True

        def lmcBRP():
            if self.accumulator >= 0 and not self.flag:
                self.program_counter = self.address_register
            return True

        def lmcINP():
            if self.input_queue is None:
                return True
            
            if len(self.input_queue) == 0:
                self.accumulator = 0
            else:
                self.accumulator = self.input_queue.popleft()
            check_flag()
            return True

        def lmcOUT():
            self.output_queue.append(self.accumulator)
            print(f"OUTPUT: {self.accumulator}")
            return True

        def lmcHLT():
            print("HALT: Execution Stopped")
            return False

        def lmcDAT(value=0):
            self.memory[self.address_register] = value

        def lmcError():
            raise ValueError(f"Invalid address for opcode 4: {self.address_register}")

        instruction_list = [
            lmcHLT,  # 0
            lmcAdd,  # 1
            lmcSub,  # 2
            lmcSTA,  # 3
            lmcError,  # 4
            lmcLDA,  # 5
            lmcBRA,  # 6
            lmcBRZ,  # 7
            lmcBRP,  # 8
            lmcINP,  # 9
            lmcOUT,  # 10
            lmcDAT  # 11
        ]

        while True:
            instr = str(self.memory[self.program_counter]).zfill(3)

            opcode = int(instr[0])
            address = int(instr[1:])

            self.program_counter += 1
            self.instruction_register = opcode

            print(f"Executing: PC={self.program_counter}, OpCode={opcode}, Address={address}, Accumulator:{self.accumulator}")

            if self.program_counter >= 100:
                self.program_counter = 0  # Wrap around to 0

            if self.instruction_register == 9:
                if address == 1:
                    instruction_list[opcode]()
                elif address == 2:
                    instruction_list[opcode+1]()
            else:
                self.address_register = address
                if not instruction_list[opcode]():
                    break  # Stop if HLT is reached

        self.exit_status()

    def reset(self):
        self.program_counter = 0
        self.accumulator = 0
        self.memory = [0] * 100
        self.flag = False

    def exit_status(self):
        print(f"PC={self.program_counter}")
        print(f"OpCode={self.instruction_register}")
        print(f"Address={self.address_register}")
        print(f"Memory={self.memory}")
        print(f"Output Queue: {list(self.output_queue)}")
        print(f"Len Output Queue: {len(list(self.output_queue))}")
        self.reset()


def unwrap(file):
    file_data = []
    
    with open(file, "r") as file_open:
        for row in file_open:
            row = row.strip()  # Remove newline and spaces around
            if "//" in row:
                row = row.split("//")[0].strip()  # Remove the comment part
            
            if row:  # Only add non-empty rows
                # Normalize the case of the instruction (e.g., convert to uppercase)
                row = row.upper()  # or row.lower() if you prefer lowercase
                file_data.append(row)  # Add the instruction to the list
    
    lmc_instance = LMC()
    lmc_instance.run(file_data, [1,2,3,4,5])

unwrap('C:\\Users\\mutua\\Documents\\Repository\\Repository\\Programmazione_Parallela\\Project\\Project_python\\quine.lmc')