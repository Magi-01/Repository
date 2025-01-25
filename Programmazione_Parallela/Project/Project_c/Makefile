# Compiler to use
CC = gcc

# Compiler flags
CFLAGS = -Wall -g

# Source files
SRCS = Project_c.c scene.c calculations.c

# Object files (same names as source files, but with .o extension)
OBJS = $(SRCS:.c=.o)

# Output executable
OUT = Project

# Default rule to compile and link the project
all: $(OUT)

# Rule to link object files into the final executable
$(OUT): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(OUT)

# Rule to compile .c files into .o object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove object files and executable
clean:
	rm -f $(OBJS) $(OUT)
