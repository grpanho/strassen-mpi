# MPI Strassen Matrix Multiplication

Parallel implementation of Strassen's matrix multiplication algorithm using MPI with a 7-ary process tree for work distribution.

## Overview

This implementation distributes the 7 Strassen products (P1-P7) across MPI processes using a simple message-passing protocol. The algorithm recursively divides matrices and distributes computations when conditions are met, falling back to sequential computation otherwise.

**Key Design Principles:**
- Simple MPI communication (only `MPI_Send` and `MPI_Recv`)
- Minimal message overhead
- Dynamic work distribution based on available processes
- Automatic fallback to sequential computation

## Algorithm

### Strassen's Method

Instead of 8 multiplications in standard algorithm, Strassen uses 7:
- **P1** = (A11 + A22) × (B11 + B22)
- **P2** = (A21 + A22) × B11
- **P3** = A11 × (B12 - B22)
- **P4** = A22 × (B21 - B11)
- **P5** = (A11 + A12) × B22
- **P6** = (A21 - A11) × (B11 + B12)
- **P7** = (A12 - A22) × (B21 + B22)

Results combined as:
- **C11** = P1 + P4 - P5 + P7
- **C12** = P3 + P5
- **C21** = P2 + P4
- **C22** = P1 - P2 + P3 + P6

### MPI Distribution Strategy

The implementation uses a **7-ary process tree** where each process can have up to 7 children:

```
Process 0 (root)
├── Process 1 (computes P1)
├── Process 2 (computes P2)
├── Process 3 (computes P3)
├── Process 4 (computes P4)
├── Process 5 (computes P5)
├── Process 6 (computes P6)
└── Process 7 (computes P7)
```

**Child Rank Formula:** For process `rank`, children are at ranks `rank * 7 + 1` through `rank * 7 + 7`.

### Distribution Decision

Work is distributed to child processes if **all three conditions** are met:

1. **Matrix size** > `MIN_SIZE_THRESHOLD` (default: 64)
2. **Tree level** < `MAX_TREE_HEIGHT` (default: 5)
3. **At least one child process exists** (rank × 7 + 1 < num_procs)

If any condition fails, all 7 products are computed locally.

### Communication Protocol

Each parent-to-child message sequence consists of 5 messages:

1. **Matrix size** `n` (int) - also serves as termination signal when 0
2. **Product index** `i` (int) - which product to compute (0-6 for P1-P7)
3. **Tree level** (int) - depth in process tree for recursive distribution
4. **Matrix A** (n×n integers, flattened)
5. **Matrix B** (n×n integers, flattened)

Child responds with:
- **Result matrix** (k×k integers, flattened) where k = n/2

### Worker Process Behavior

Worker processes (rank ≠ 0):
1. Loop waiting for work assignments
2. Receive matrices and parameters from parent
3. Divide matrices into quadrants
4. Compute assigned Strassen product recursively
5. Can further distribute to own children if conditions permit
6. Send result back to parent
7. Exit when receiving termination signal (n = 0)

## Files

- `strassen_mpi.h/c` - Core MPI Strassen implementation
- `matrix_utils.h/c` - Matrix operations (add, subtract, split, combine, flatten/unflatten)
- `main.c` - Master/worker coordination and verification
- `Makefile` - Build and run configurations

## Building and Running

### Requirements
- MPI implementation (OpenMPI or MPICH)
- C99 compiler (gcc)

### Build
```bash
make              # Compile
make clean        # Remove build artifacts
make check-mpi    # Verify MPI installation
```

### Run
```bash
# Basic usage
mpirun -np <num_processes> ./strassen_mpi <matrix_size>

# Examples
mpirun -np 8 ./strassen_mpi 128   # 128×128 with 8 processes
mpirun -np 4 ./strassen_mpi 64    # 64×64 with 4 processes

# Using Makefile shortcuts
make run                          # Default: 2 processes, 4×4 matrix
make run-custom NP=8 SIZE=64      # Custom parameters
make test                         # Run test suite
```

**Note:** Matrix size must be a power of 2 (2, 4, 8, 16, 32, 64, 128, ...)

## Configuration

Edit `strassen_mpi.h`:

```c
#define MAX_TREE_HEIGHT 5      // Maximum recursion depth for distribution
#define MIN_SIZE_THRESHOLD 64  // Minimum matrix size for parallel processing
#define TAG_WORK 100           // MPI message tag
```

**Tuning Guidelines:**
- **Increase `MIN_SIZE_THRESHOLD`** to reduce communication overhead (less messages, more local computation)
- **Decrease `MAX_TREE_HEIGHT`** to limit process tree depth (prevents over-parallelization)
- **Optimal settings** depend on matrix size, network latency, and process count

## Process Tree Example

With 8 processes computing a 128×128 matrix:

```
Level 0: Process 0 (128×128)
         ├─ Distributes P1-P7 to processes 1-7
         └─ Each child computes with 64×64 submatrices

Level 1: Processes 1-7 (64×64 each)
         └─ If level < MAX_TREE_HEIGHT and size > threshold:
            Could distribute further, but likely compute locally
```

## Verification

The program includes automatic correctness verification:
- Computes result using parallel Strassen
- Computes same multiplication using sequential standard algorithm
- Compares results element-by-element
- Reports timing and speedup

Example output:
```
=== MPI Strassen Matrix Multiplication ===
Matrix size: 64x64
Number of processes: 8
Tree height limit: 5
Sequential threshold: 64
==========================================

Starting MPI Strassen multiplication...
MPI Strassen multiplication completed!
CPU Time: 0.012340 seconds
Wall Time: 0.008765 seconds

Verifying result with standard multiplication...
Standard multiplication time: 0.045678 seconds
Verification PASSED - Results match!
Speedup: 5.21x
```

## Performance Considerations

**When Parallelization Helps:**
- Large matrices (≥ 256×256)
- Sufficient processes available
- Low communication latency

**When to Use Sequential:**
- Small matrices (< 64×64)
- Very few processes
- High network overhead

**Complexity:**
- Sequential Strassen: O(n^2.807)
- Standard multiplication: O(n^3)
- Communication cost: O(log P × n²) for P processes

## Troubleshooting

**Error: Matrix size must be a power of 2**
- Solution: Use 2, 4, 8, 16, 32, 64, 128, 256, etc.

**Poor speedup or slowdown**
- Increase `MIN_SIZE_THRESHOLD` to reduce communication
- Use fewer processes
- Test with larger matrices

**Process hangs or deadlocks**
- Ensure `MAX_TREE_HEIGHT` is reasonable (≤ 5)
- Check process count: need rank × 7 + 7 < num_procs for full distribution

## Implementation Notes

1. **Matrix flattening**: 2D matrices are flattened to 1D arrays for MPI communication
2. **Memory management**: All allocated matrices are properly freed after use
3. **Hybrid approach**: Combines distributed and sequential computation seamlessly
4. **No complex structures**: Uses only basic MPI send/receive with integer arrays
