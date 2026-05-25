# Name: Fadhla Mohamed
# Sirname: Mutua
# Matricola: SM3201434

from assembler import Assembler  # Import the Assembler class
from utils import Utility  # Import the Utility class
from collections import deque  # Import deque for input/output queues

class LMC(Assembler):
    def __init__(self):
        # Initialize LMC instance variables
        self.program_counter = 0
        self.accumulator = 0
        self.memory = [0] * 100
        self.input_queue = deque()
        self.output_queue = deque()
        self.flag = False
        self.result_index = 0
        self.Utility = Utility()  # Create an instance of Utility class

    def load_instructions(self, file_data):
        # Load instructions from file
        self.assemble_manually(file_data)

    def run(self, file_data, input_data=None):
        # Set file data and input queue
        self.file = file_data
        if input_data:
            for inp in input_data:
                self.input_queue.append(inp)
        else:
            self.input_queue = None
        
        self.load_instructions(self.file)

        # List of instruction functions
        instruction_list = [
            self.Utility.lmcHLT,  # 0
            self.Utility.lmcAdd,  # 1
            self.Utility.lmcSub,  # 2
            self.Utility.lmcSTA,  # 3
            self.Utility.lmcError,  # 4
            self.Utility.lmcLDA,  # 5
            self.Utility.lmcBRA,  # 6
            self.Utility.lmcBRZ,  # 7
            self.Utility.lmcBRP,  # 8
            self.Utility.lmcINP,  # 9
            self.Utility.lmcOUT,  # 10
            self.Utility.lmcDAT  # 11
        ]

        # Execution loop
        while True:
            # Ensures that the memory reference size is always 3 values (or 3 digits) by adding leading zeros (e.g. 5 becomes 005)
            instr = str(self.memory[self.program_counter]).zfill(3)
            opcode = int(instr[0])
            address = int(instr[1:])

            self.program_counter += 1
            self.instruction_register = opcode

            if self.program_counter >= 100:
                self.program_counter = 0  # Wrap around to 0

            # Execute the instruction based on opcode
            # To note: 
            # if operation code is 901 then it is input
            # if operation code is 902 then it is output
            # if operation code is 000 then it is halt
            if self.instruction_register == 9:
                if address == 1:
                    instruction_list[opcode](self)
                elif address == 2:
                    instruction_list[opcode+1](self)
            else:
                self.address_register = address
                if not instruction_list[opcode](self):
                    break  # Stop if HLT is reached

        self.exit_status()

    def reset(self):
        # Reset the LMC state, clearing all memory
        self.program_counter = 0
        self.accumulator = 0
        self.memory = [0] * 100  # Clear all memory slots
        self.flag = False


    def exit_status(self):
        # Print the current status
        print(f"PC={self.program_counter}")
        print(f"OpCode={self.instruction_register}")
        print(f"Address={self.address_register}")
        print(f"Memory={self.memory}")
        print(f"Output Queue: {list(self.output_queue)}")
        print(f"Len Output Queue: {len(self.output_queue)}")
        self.reset()
