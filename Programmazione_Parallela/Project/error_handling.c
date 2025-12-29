#include <stdio.h>
#include <stdlib.h>
#include "error_handling.h"

// Function to handle errors and exit the program with a user-defined exit status
void error_encountered(int exit_status) {
    printf("Error occurred during processing. Exiting program with status %d.\n", exit_status);
    exit(exit_status); // Use the provided exit status
}