#
#    Name: Fadhla Mohamed
#    Sirname: Mutua
#    Matricola: SM3201434
#

# Compiler to use
CC = gcc

# Compiler flags
CFLAGS = -Wall -g -fopenmp

# Check if OpenMP is enabled
ifeq ($(USE_OPENMP), 1)
    CFLAGS += -fopenmp
endif

# Source files (automatically include all .c files in the current directory)
SRCS = $(wildcard *.c)

# Object directory
LIB_DIR = lib

# Object files (place .o files in lib/)
OBJS = $(patsubst %.c, $(LIB_DIR)/%.o, $(SRCS))

# Output executable
OUT = main

# OS-specific commands
ifeq ($(OS),Windows_NT)
    MKDIR = if not exist $(LIB_DIR) mkdir $(LIB_DIR)
    RM = cmd /c "del /F /Q $(OBJS) 2> nul"
    RMDIR = if exist $(LIB_DIR) rmdir /S /Q $(LIB_DIR)
else
    MKDIR = mkdir -p $(LIB_DIR)
    RM = rm -f $(OBJS)
    RMDIR = rm -rf $(LIB_DIR)
endif

# Default rule to compile and link the project
all: $(OUT)

# Rule to link object files into the final executable
$(OUT): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT) -lm  

# Rule to compile .c files into .o object files in lib/
$(LIB_DIR)/%.o: %.c
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# General clean rule to remove object files, executable, and other generated files
clean:
	-@$(RM)
	-@if exist $(OUT) del /F /Q $(OUT)
	-@$(RMDIR)

# Optional rule to clean only object files
clean-objects:
	-@$(RM)

# Optional rule to clean only the executable
clean-executable:
	-@if exist $(OUT) del /F /Q $(OUT)

# Optional rule to remove all generated files, including objects and executable
clean-all: clean

# Phony targets to prevent make from getting confused by files with the same names
.PHONY: all clean clean-objects clean-executable clean-all
