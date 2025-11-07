#include "matrix_utils.h"


int** initializeMatrix(int n) {
    int** matrix = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (int*)calloc(n, sizeof(int));
    }
    return matrix;
}


void freeMatrix(int** matrix, int n) {
    if (matrix) {
        for (int i = 0; i < n; i++) {
            free(matrix[i]);
        }
        free(matrix);
    }
}


void printMatrix(int** matrix, int n, const char* name) {
    printf("\nMatrix %s (%dx%d):\n", name, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


int** addMatrices(int** A, int** B, int n) {
    int** result = initializeMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}


int** subtractMatrices(int** A, int** B, int n) {
    int** result = initializeMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// Split a matrix into 4 quadrants
void splitMatrix(int** parent, int** A11, int** A12, int** A21, int** A22, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i][j] = parent[i][j];              // Top-left
            A12[i][j] = parent[i][j + k];          // Top-right
            A21[i][j] = parent[i + k][j];          // Bottom-left
            A22[i][j] = parent[i + k][j + k];      // Bottom-right
        }
    }
}

// Combine 4 quadrants into a single matrix
void combineBlocks(int** C, int** C11, int** C12, int** C21, int** C22, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i][j] = C11[i][j];              // Top-left
            C[i][j + k] = C12[i][j];          // Top-right
            C[i + k][j] = C21[i][j];          // Bottom-left
            C[i + k][j + k] = C22[i][j];      // Bottom-right
        }
    }
}


int* flattenMatrix(int** matrix, int n) {
    int* flat = (int*)malloc(n * n * sizeof(int));
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat[index++] = matrix[i][j];
        }
    }
    return flat;
}


int** unflattenMatrix(int* flat, int n) {
    int** matrix = initializeMatrix(n);
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = flat[index++];
        }
    }
    return matrix;
}


void copyMatrix(int** source, int** dest, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dest[i][j] = source[i][j];
        }
    }
}


int isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Sequential Strassen multiplication (for local computation)
int** strassenMultiply(int** A, int** B, int n) {
    if (n == 1) {
        int** C = initializeMatrix(1);
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    if (n <= 32) {
        return standardMultiply(A, B, n);
    }

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

    int** temp1, **temp2;

    // P1 = (A11 + A22) * (B11 + B22)
    temp1 = addMatrices(A11, A22, k);
    temp2 = addMatrices(B11, B22, k);
    int** P1 = strassenMultiply(temp1, temp2, k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // P2 = (A21 + A22) * B11
    temp1 = addMatrices(A21, A22, k);
    int** P2 = strassenMultiply(temp1, B11, k);
    freeMatrix(temp1, k);

    // P3 = A11 * (B12 - B22)
    temp1 = subtractMatrices(B12, B22, k);
    int** P3 = strassenMultiply(A11, temp1, k);
    freeMatrix(temp1, k);

    // P4 = A22 * (B21 - B11)
    temp1 = subtractMatrices(B21, B11, k);
    int** P4 = strassenMultiply(A22, temp1, k);
    freeMatrix(temp1, k);

    // P5 = (A11 + A12) * B22
    temp1 = addMatrices(A11, A12, k);
    int** P5 = strassenMultiply(temp1, B22, k);
    freeMatrix(temp1, k);

    // P6 = (A21 - A11) * (B11 + B12)
    temp1 = subtractMatrices(A21, A11, k);
    temp2 = addMatrices(B11, B12, k);
    int** P6 = strassenMultiply(temp1, temp2, k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // P7 = (A12 - A22) * (B21 + B22)
    temp1 = subtractMatrices(A12, A22, k);
    temp2 = addMatrices(B21, B22, k);
    int** P7 = strassenMultiply(temp1, temp2, k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // Calculate result quadrants
    // C11 = P1 + P4 - P5 + P7
    temp1 = addMatrices(P1, P4, k);
    temp2 = subtractMatrices(temp1, P5, k);
    int** C11 = addMatrices(temp2, P7, k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // C12 = P3 + P5
    int** C12 = addMatrices(P3, P5, k);

    // C21 = P2 + P4
    int** C21 = addMatrices(P2, P4, k);

    // C22 = P1 - P2 + P3 + P6
    temp1 = subtractMatrices(P1, P2, k);
    temp2 = addMatrices(temp1, P3, k);
    int** C22 = addMatrices(temp2, P6, k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);

    // Combine result
    int** C = initializeMatrix(n);
    combineBlocks(C, C11, C12, C21, C22, k);

    freeMatrix(A11, k); freeMatrix(A12, k); freeMatrix(A21, k); freeMatrix(A22, k);
    freeMatrix(B11, k); freeMatrix(B12, k); freeMatrix(B21, k); freeMatrix(B22, k);
    freeMatrix(P1, k); freeMatrix(P2, k); freeMatrix(P3, k); freeMatrix(P4, k);
    freeMatrix(P5, k); freeMatrix(P6, k); freeMatrix(P7, k);
    freeMatrix(C11, k); freeMatrix(C12, k); freeMatrix(C21, k); freeMatrix(C22, k);

    return C;
}


int** standardMultiply(int** A, int** B, int n) {
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
