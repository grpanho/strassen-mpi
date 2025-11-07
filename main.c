#include "strassen_mpi.h"
#include <time.h>

void initializeRandomMatrix(int** matrix, int n, int seed);
void verifyResult(int** A, int** B, int** C, int n);
int** sequentialStandardMultiply(int** A, int** B, int n);
void workerProcess(int rank, int num_procs);


int main(int argc, char* argv[]) {
    int rank, num_procs;
    int n = 4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc > 1) {
        n = atoi(argv[1]);
        if (!isPowerOfTwo(n) || n < 2) {
            if (rank == 0) {
                printf("Error: Matrix size must be a power of 2 and >= 2\n");
                printf("Usage: %s [matrix_size]\n", argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        printf("=== MPI Strassen Matrix Multiplication ===\n");
        printf("Matrix size: %dx%d\n", n, n);
        printf("Number of processes: %d\n", num_procs);
        printf("Tree height limit: %d\n", MAX_TREE_HEIGHT);
        printf("Sequential threshold: %d\n", MIN_SIZE_THRESHOLD);
        printf("==========================================\n\n");

        int** A = initializeMatrix(n);
        int** B = initializeMatrix(n);

        initializeRandomMatrix(A, n, 123);
        initializeRandomMatrix(B, n, 456);

        if (n <= 8) {
            printMatrix(A, n, "A");
            printMatrix(B, n, "B");
        }

        clock_t start_time = clock();
        double mpi_start_time = MPI_Wtime();

        printf("Starting MPI Strassen multiplication...\n");
        int** C = strassenMultiplyMPI(A, B, n, rank, num_procs, 0);

        double mpi_end_time = MPI_Wtime();
        clock_t end_time = clock();

        double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        double wall_time = mpi_end_time - mpi_start_time;

        printf("MPI Strassen multiplication completed!\n");
        printf("CPU Time: %.6f seconds\n", cpu_time);
        printf("Wall Time: %.6f seconds\n", wall_time);

        if (n <= 8) {
            printMatrix(C, n, "Result C");
        }

        if (n <= 2048) {
            printf("\nVerifying result with Strassen sequential multiplication...\n");
            clock_t verify_start = clock();
            int** C_verify = strassenMultiply(A, B, n);
            clock_t verify_end = clock();

            double verify_time = ((double)(verify_end - verify_start)) / CLOCKS_PER_SEC;
            printf("Strassen sequential multiplication time: %.6f seconds\n", verify_time);

            int correct = 1;
            for (int i = 0; i < n && correct; i++) {
                for (int j = 0; j < n && correct; j++) {
                    if (C[i][j] != C_verify[i][j]) {
                        correct = 0;
                        printf("Mismatch at [%d][%d]: Strassen MPI=%d, Strassen Seq=%d\n",
                               i, j, C[i][j], C_verify[i][j]);
                        break;
                    }
                }
            }

            if (correct) {
                printf("Verification PASSED - Results match!\n");
                printf("Speedup: %.2fx\n", verify_time / wall_time);
            } else {
                printf("Verification FAILED - Results do not match!\n");
            }

            freeMatrix(C_verify, n);
        }

        freeMatrix(A, n);
        freeMatrix(B, n);
        freeMatrix(C, n);

        int terminate = 0;
        for (int i = 1; i < num_procs; i++) {
            MPI_Send(&terminate, 1, MPI_INT, i, TAG_WORK, MPI_COMM_WORLD);
        }
    } else {
        workerProcess(rank, num_procs);
    }

    MPI_Finalize();
    return 0;
}


void workerProcess(int rank, int num_procs) {
    while (1) {
        MPI_Status status;
        int n, product_index, level;

        // Try to receive matrix size
        MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK, MPI_COMM_WORLD, &status);

        // Check if this is a termination signal (n = 0)
        if (n == 0) {
            break;
        }

        int parent_rank = status.MPI_SOURCE;

        // Receive product index and level
        MPI_Recv(&product_index, 1, MPI_INT, parent_rank, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&level, 1, MPI_INT, parent_rank, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive matrices A and B
        int* flatA = (int*)malloc(n * n * sizeof(int));
        int* flatB = (int*)malloc(n * n * sizeof(int));

        MPI_Recv(flatA, n * n, MPI_INT, parent_rank, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(flatB, n * n, MPI_INT, parent_rank, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Unflatten matrices
        int** A = unflattenMatrix(flatA, n);
        int** B = unflattenMatrix(flatB, n);
        free(flatA);
        free(flatB);

        // Divide into quadrants
        int k = n / 2;
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

        // Compute the specific product based on product_index
        int** result = computeStrassenProductMPI(A11, A12, A21, A22, B11, B12, B21, B22,
                                                  k, rank, num_procs, level + 1, product_index);

        // Send result back to parent
        int* flatResult = flattenMatrix(result, k);
        MPI_Send(flatResult, k * k, MPI_INT, parent_rank, TAG_WORK, MPI_COMM_WORLD);

        free(flatResult);
        freeMatrix(A, n);
        freeMatrix(B, n);
        freeMatrix(A11, k); freeMatrix(A12, k); freeMatrix(A21, k); freeMatrix(A22, k);
        freeMatrix(B11, k); freeMatrix(B12, k); freeMatrix(B21, k); freeMatrix(B22, k);
        freeMatrix(result, k);
    }
}

void initializeRandomMatrix(int** matrix, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10; // Values 0-9 for easy verification
        }
    }
}

int** sequentialStandardMultiply(int** A, int** B, int n) {
    int** C = initializeMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
