# Name: Fadhla Mohamed
# Sirname: Mutua
# Matricola: SM3201434

class Utility:
    # Unwrap function to read and clean data from a file
    def unwrap(file):
        file_data = []
        
        with open(file, "r") as file_open:
            # Iterate over each line in the file
            for row in file_open:
                row = row.strip()  # Remove newline and spaces around the line
                if "//" in row:
                    row = row.split("//")[0].strip()  # Remove the comment part after "//"
                
                if row:  # Only add non-empty rows
                    row = row.upper()  # Normalize the case of the instruction to uppercase
                    file_data.append(row)  # Add the instruction to the list
        
        return file_data  # Return the cleaned list of instructions

    # Add the value at the address register to the accumulator
    def lmcAdd(self, lmc_instance):
        lmc_instance.accumulator += lmc_instance.memory[lmc_instance.address_register]
        self.check_flag(lmc_instance)  # Check and adjust the flag if needed
        return True

    # Subtract the value at the address register from the accumulator
    def lmcSub(self, lmc_instance):
        lmc_instance.accumulator -= lmc_instance.memory[lmc_instance.address_register]
        self.check_flag(lmc_instance)  # Check and adjust the flag if needed
        return True

    # Store the value of the accumulator in the memory at the address register
    def lmcSTA(self, lmc_instance):
        lmc_instance.memory[lmc_instance.address_register] = lmc_instance.accumulator
        return True

    # Load the value from memory at the address register into the accumulator
    def lmcLDA(self, lmc_instance):
        lmc_instance.accumulator = lmc_instance.memory[lmc_instance.address_register]
        self.check_flag(lmc_instance)  # Check and adjust the flag if needed
        return True

    # Branch to the address in the address register unconditionally
    def lmcBRA(self, lmc_instance):
        lmc_instance.program_counter = lmc_instance.address_register
        return True

    # Branch if the accumulator is zero and no flag is set
    def lmcBRZ(self, lmc_instance):
        if lmc_instance.accumulator == 0 and not lmc_instance.flag:
            lmc_instance.program_counter = lmc_instance.address_register
        return True

    # Branch if the accumulator is positive and no flag is set
    def lmcBRP(self, lmc_instance):
        if lmc_instance.accumulator >= 0 and not lmc_instance.flag:
            lmc_instance.program_counter = lmc_instance.address_register
        return True

    # Take input from the input queue and store it in the accumulator
    def lmcINP(self, lmc_instance):
        # if the queue is none we continue on
        if lmc_instance.input_queue is None:
            return True
        
        # If the queue is empty we carry on
        if len(lmc_instance.input_queue) == 0:
            lmc_instance.accumulator = 0
            return True
        
        # If the queue is empty but we still try to access it, raise an error
        if lmc_instance.input_queue:
            lmc_instance.accumulator = lmc_instance.input_queue.popleft()
        else: raise IndexError("Queue is empty")

        self.check_flag(lmc_instance)  # Check and adjust the flag if needed
        return True

    # Output the value of the accumulator to the output queue and print it
    def lmcOUT(self, lmc_instance):
        lmc_instance.output_queue.append(lmc_instance.accumulator)
        print(f"OUTPUT: {lmc_instance.accumulator}")
        return True

    # Halt the execution of the program
    def lmcHLT(self, lmc_instance):
        print("HALT: Execution Stopped")
        return False
    
    # Raise an error if the address register is invalid for opcode 4
    def lmcError(self, lmc_instance):
        raise ValueError(f"Invalid address for opcode 4: {lmc_instance.address_register}")

    # Store a value in memory at the address register
    def lmcDAT(self, lmc_instance, value=0):
        lmc_instance.memory[lmc_instance.address_register] = value

    # Check and handle the flag for overflow or underflow in the accumulator
    def check_flag(self, lmc_instance):
        lmc_instance.flag = False
        if lmc_instance.accumulator >= 1000 or lmc_instance.accumulator < 0:
            lmc_instance.flag = True  # Set the flag for overflow or underflow
            lmc_instance.accumulator = lmc_instance.accumulator % 1000  # Normalize the accumulator
