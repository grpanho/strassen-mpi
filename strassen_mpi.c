#include "strassen_mpi.h"

int** strassenMultiplyMPI(int** A, int** B, int n, int rank, int num_procs, int level) {
    // Base case
    if (n == 1) {
        int** C = initializeMatrix(1);
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    // Use standard multiplication for small matrices
    if (n <= MIN_SIZE_THRESHOLD) {
        return standardMultiply(A, B, n);
    }

    int k = n / 2;

    // Divide matrices into quadrants
    int** A11 = initializeMatrix(k);
    int** A12 = initializeMatrix(k);
    int** A21 = initializeMatrix(k);
    int** A22 = initializeMatrix(k);

    int** B11 = initializeMatrix(k);
    int** B12 = initializeMatrix(k);
    int** B21 = initializeMatrix(k);
    int** B22 = initializeMatrix(k);

    splitMatrix(A, A11, A12, A21, A22, k);
    splitMatrix(B, B11, B12, B21, B22, k);

    int** P[7];

    // Check if we should distribute work to child processes
    if (shouldDistribute(n, level, num_procs, rank)) {
        int num_children = 0;
        for (int i = 0; i < 7; i++) {
            int child_rank = rank * 7 + (i + 1);
            if (child_rank < num_procs) {
                num_children++;

                // Flatten and send matrices to child
                int* flatA = flattenMatrix(A, n);
                int* flatB = flattenMatrix(B, n);

                MPI_Send(&n, 1, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD);
                MPI_Send(&i, 1, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD);
                MPI_Send(&level, 1, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD);
                MPI_Send(flatA, n * n, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD);
                MPI_Send(flatB, n * n, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD);

                free(flatA);
                free(flatB);
            }
        }

        // Compute products: receive from children or compute locally
        for (int i = 0; i < 7; i++) {
            int child_rank = rank * 7 + (i + 1);
            if (child_rank < num_procs) {
                // Receive result from child
                int* flatResult = (int*)malloc(k * k * sizeof(int));
                MPI_Recv(flatResult, k * k, MPI_INT, child_rank, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                P[i] = unflattenMatrix(flatResult, k);
                free(flatResult);
            } else {
                // Compute locally (no more children available)
                P[i] = computeStrassenProductMPI(A11, A12, A21, A22, B11, B12, B21, B22, k, rank, num_procs, level, i);
            }
        }
    } else {
        // No distribution - compute all products locally
        for (int i = 0; i < 7; i++) {
            P[i] = computeStrassenProductMPI(A11, A12, A21, A22, B11, B12, B21, B22, k, rank, num_procs, level, i);
        }
    }

    // Calculate result quadrants using Strassen's formulas
    // C11 = P1 + P4 - P5 + P7
    int** temp1 = addMatrices(P[0], P[3], k);
    int** temp2 = subtractMatrices(temp1, P[4], k);
    int** C11 = addMatrices(temp2, P[6], k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // C12 = P3 + P5
    int** C12 = addMatrices(P[2], P[4], k);

    // C21 = P2 + P4
    int** C21 = addMatrices(P[1], P[3], k);

    // C22 = P1 - P2 + P3 + P6
    temp1 = subtractMatrices(P[0], P[1], k);
    temp2 = addMatrices(temp1, P[2], k);
    int** C22 = addMatrices(temp2, P[5], k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // Combine result quadrants
    int** C = initializeMatrix(n);
    combineBlocks(C, C11, C12, C21, C22, k);

    // Free memory
    freeMatrix(A11, k); freeMatrix(A12, k); freeMatrix(A21, k); freeMatrix(A22, k);
    freeMatrix(B11, k); freeMatrix(B12, k); freeMatrix(B21, k); freeMatrix(B22, k);
    for (int i = 0; i < 7; i++) {
        freeMatrix(P[i], k);
    }
    freeMatrix(C11, k); freeMatrix(C12, k); freeMatrix(C21, k); freeMatrix(C22, k);

    return C;
}

int** computeStrassenProductMPI(int** A11, int** A12, int** A21, int** A22,
                               int** B11, int** B12, int** B21, int** B22,
                               int k, int rank, int num_procs, int level, int product_index) {
    int** tempA = NULL;
    int** tempB = NULL;
    int** result = NULL;

    switch (product_index) {
        case 0: // P1 = (A11 + A22) * (B11 + B22)
            tempA = addMatrices(A11, A22, k);
            tempB = addMatrices(B11, B22, k);
            break;
        case 1: // P2 = (A21 + A22) * B11
            tempA = addMatrices(A21, A22, k);
            tempB = initializeMatrix(k);
            copyMatrix(B11, tempB, k);
            break;
        case 2: // P3 = A11 * (B12 - B22)
            tempA = initializeMatrix(k);
            copyMatrix(A11, tempA, k);
            tempB = subtractMatrices(B12, B22, k);
            break;
        case 3: // P4 = A22 * (B21 - B11)
            tempA = initializeMatrix(k);
            copyMatrix(A22, tempA, k);
            tempB = subtractMatrices(B21, B11, k);
            break;
        case 4: // P5 = (A11 + A12) * B22
            tempA = addMatrices(A11, A12, k);
            tempB = initializeMatrix(k);
            copyMatrix(B22, tempB, k);
            break;
        case 5: // P6 = (A21 - A11) * (B11 + B12)
            tempA = subtractMatrices(A21, A11, k);
            tempB = addMatrices(B11, B12, k);
            break;
        case 6: // P7 = (A12 - A22) * (B21 + B22)
            tempA = subtractMatrices(A12, A22, k);
            tempB = addMatrices(B21, B22, k);
            break;
    }

    result = strassenMultiplyMPI(tempA, tempB, k, rank, num_procs, level + 1);

    freeMatrix(tempA, k);
    freeMatrix(tempB, k);

    return result;
}


int shouldDistribute(int n, int level, int num_procs, int rank) {
    // Condition 1: Matrix must be bigger than minimum threshold
    if (n <= MIN_SIZE_THRESHOLD) {
        return 0;
    }

    // Condition 2: Maximum tree depth must not be reached
    if (level >= MAX_TREE_HEIGHT) {
        return 0;
    }

    // Condition 3: At least one child process must be available
    // Using tree structure: rank uses children at positions rank*7+1 to rank*7+7
    int first_child = rank * 7 + 1;
    if (first_child >= num_procs) {
        return 0;
    }

    // All conditions met - distribute work
    return 1;
}
