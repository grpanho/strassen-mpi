#ifndef STRASSEN_MPI_H
#define STRASSEN_MPI_H

#include "matrix_utils.h"
#include <mpi.h>
#include <math.h>

#define MAX_TREE_HEIGHT 5    // Maximum height of the process tree (reduced to avoid deadlock)
#define MIN_SIZE_THRESHOLD 64 // Minimum size for parallel processing

int** strassenMultiplyMPI(int** A, int** B, int n, int rank, int num_procs, int level);

// Strassen computation functions for MPI
int** computeStrassenProductMPI(int** A11, int** A12, int** A21, int** A22,
                               int** B11, int** B12, int** B21, int** B22,
                               int k, int rank, int num_procs, int level, int product_index);

// Helper function to determine if work should be distributed
int shouldDistribute(int n, int level, int num_procs, int rank);

#endif // STRASSEN_MPI_H
