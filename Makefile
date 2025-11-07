MPICC = mpicc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = strassen_mpi

SOURCES = main.c strassen_mpi.c matrix_utils.c
HEADERS = strassen_mpi.h matrix_utils.h

all: $(TARGET)

# Compile and link directly
$(TARGET): $(SOURCES) $(HEADERS)
	$(MPICC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run with default parameters (4x4 matrix, 2 processes)
run: $(TARGET)
	mpirun -np 2 ./$(TARGET) 4

# Run with custom parameters
# Usage: make run-custom NP=4 SIZE=8
run-custom: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(SIZE)

# Run tests with different matrix sizes
test: $(TARGET)
	@echo "Testing with 2x2 matrix, 2 processes:"
	mpirun -np 2 ./$(TARGET) 2
	@echo "\nTesting with 4x4 matrix, 2 processes:"
	mpirun -np 2 ./$(TARGET) 4
	@echo "\nTesting with 8x8 matrix, 4 processes:"
	mpirun -np 4 ./$(TARGET) 8

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Performance test with larger matrices
performance: $(TARGET)
	@echo "Performance test with 16x16 matrix, 4 processes:"
	mpirun -np 4 ./$(TARGET) 16
	@echo "\nPerformance test with 32x32 matrix, 8 processes:"
	mpirun -np 8 ./$(TARGET) 32

# Check MPI installation
check-mpi:
	@echo "Checking MPI installation..."
	@which mpicc || (echo "Error: mpicc not found. Please install MPI." && exit 1)
	@mpicc --version
	@echo "MPI is properly installed."

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the executable (default)"
	@echo "  clean       - Remove build artifacts"
	@echo "  run         - Run with default parameters (4x4, 2 processes)"
	@echo "  run-custom  - Run with custom parameters (use NP=n SIZE=n)"
	@echo "  test        - Run tests with different configurations"
	@echo "  debug       - Build with debug symbols"
	@echo "  performance - Run performance tests"
	@echo "  check-mpi   - Check MPI installation"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run-custom NP=4 SIZE=8"
	@echo "  make debug"
	@echo "  make performance"

.PHONY: all clean run run-custom test debug performance check-mpi help
