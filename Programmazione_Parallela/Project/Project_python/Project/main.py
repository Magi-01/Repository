# Name: Fadhla Mohamed
# Sirname: Mutua
# Matricola: SM3201434

import sys  # Import sys module to handle command-line arguments (I use Windows)
from cpu import LMC  # Import the LMC class from the cpu module
from utils import Utility as ut  # Import Utility class from the utils module

def main():
    # Check if the user has provided the required file path
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.lmc> input1 input2 ...")  
        # Print usage instructions
        raise Exception("You did not use the correct format or did not provide a File")

    # Retrieve the file path from the command-line arguments
    file_path = sys.argv[1]
    
    # Retrieve input data from command-line arguments, if any, and convert them to integers
    # If no input data is provided, input_data will be set to None
    input_data = [int(float(item)) for item in sys.argv[2:]] if sys.argv[2:] else None
    
    # Use the unwrap function from the Utility class to read and process the file
    file_data = ut.unwrap(file_path)

    # Create an instance of the LMC (Little Man Computer) class
    lmc_instance = LMC()
    
    # Run the LMC with the file data and input data (if any)
    lmc_instance.run(file_data, input_data)

# This block ensures that the main function is only executed if the script is run directly
if __name__ == "__main__":
    main()

