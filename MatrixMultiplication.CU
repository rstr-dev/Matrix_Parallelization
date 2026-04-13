#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int VALUEMAX = 10;

char output_A_file[] = "outputs/A.txt";
char output_B_file[] = "outputs/B.txt";
char output_C_serial_file[] = "outputs/C_serial.txt";
char output_C_parallel_file[] = "outputs/C_parallel.txt";

// UTILITY ------------------------------------------------------------------------
struct RowCol {
    int row;
    int col;
};

void print_to_file(int* matrix, struct RowCol row_col, char* filename) {
    FILE *fptr;
    fptr = fopen(filename, "w");
    fprintf(fptr, "\n\n");

    for (int i = 0; i < (row_col.row * row_col.col); i++) {
        if ((i % row_col.col) == 0 && (i != 0)) {
            fprintf(fptr, "\n");
        }

        fprintf(fptr, "%d\t", matrix[i]);
    }

    fprintf(fptr, "\n\n\n");
    fclose(fptr);
}

void print_to_command(int* matrix, struct RowCol row_col) {
    for (int i = 0; i < (row_col.row * row_col.col); i++) {
        if ((i % row_col.col) == 0 && (i != 0)) {
            printf("\n");
        }

        printf("%d\t", matrix[i]);
    }

    printf("\n\n");
}

// create the arrays and return the row col structure that fits the given array
struct RowCol initalize_matrix_values(int *matrix, int type, int row, int col) {
    int N = row * col;

    for (int i = 0; i < N; i++) {
        if (type == 0) { // matrix only has 0s
            matrix[i] = 0;
        } else { // matrix has random values
            matrix[i] = rand() % (VALUEMAX + 1); // value [0, VALUEMAX]
        }
    }

    struct RowCol row_col = {row, col};

    return row_col;
}

int confirm_input(int default_value, char user_provided[]) {
    int n;
    char ch;
    int status;

    // checking if the input is an int
    status = sscanf(user_provided, "%d%c", &n, &ch);
    if (status != 1) {
        printf("The provided user inputer was not an int, the program will use the default value.\n");

        return default_value;
    }

    return atoi(user_provided);
}

// converts a row and column to an index
__host__ __device__ int row_col_to_index(int row, int col, int row_width) {
    return (row * row_width) + col;
}

// converts an index into a row and column
__host__ __device__ struct RowCol index_to_row_col(int index, int row_width) {
    struct RowCol values;

    values.row = (index / row_width);
    values.col = (index % row_width);

    return values;
}

// MULTIPLICATION -----------------------------------------------------------------

int dot_product(int* A, int* B, struct RowCol row_col_C, int index, int m) {
    int sum = 0;

    // convert the current index to row col values
    struct RowCol current_row_col = index_to_row_col(index, row_col_C.col);

    for (int i = 0; i < m; i++) {
        // get the index for A and B that corresponds to the current row and column
        int A_index = row_col_to_index(current_row_col.row, i, m); // m is the rowWidth of A
        int B_index = row_col_to_index(i, current_row_col.col, row_col_C.col); // row_col_C.col is the rowWidth of B

        sum += (A[A_index] * B[B_index]);
    }

    return sum;
}

void serial_multiplication(int* A, int* B, int* C, struct RowCol row_col_C, int m) {
    for (int i = 0; i < (row_col_C.row * row_col_C.col); i++) {
        C[i] = dot_product(A, B, row_col_C, i, m);
    }
}

__global__ void kernel_dot_product(int* A, int* B, int* C, struct RowCol row_col_C, int m) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;

    // convert the current index to row col values
    struct RowCol current_row_col = index_to_row_col(index, row_col_C.col);

    for (int i = 0; i < m; i++) {
        // get the index for A and B that corresponds to the current row and column
        int A_index = row_col_to_index(current_row_col.row, i, m); // m is the rowWidth of A
        int B_index = row_col_to_index(i, current_row_col.col, row_col_C.col); // col_amount_C is the rowWidth of B

        sum += (A[A_index] * B[B_index]);
    }

    C[index] = sum;
}

void parallel_multiplication(int *A, int *B, int *C, struct RowCol row_col_C, int m) {
    int N = row_col_C.row * row_col_C.col;
    int *dev_A, *dev_B, *dev_C;

    int size_A = (row_col_C.row * m) * sizeof(int);
    int size_B = (m * row_col_C.col) * sizeof(int);
    int size_C = N * sizeof(int);

    cudaMalloc((void**)&dev_A, size_A);
    cudaMalloc((void**)&dev_B, size_B);
    cudaMalloc((void**)&dev_C, size_C);

    cudaMemcpy(dev_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size_B, cudaMemcpyHostToDevice);

    // Mathematical Relation for Grid Sizing 
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_dot_product << <blocksPerGrid, threadsPerBlock >> > (dev_A, dev_B, dev_C, row_col_C, m);

    cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost);
}

// MAIN ---------------------------------------------------------------------------

// user allowed inputs:
//  1: the number of rows in the first matrix
//  2: the number of columns in the first matrix and the number of rows in the second matrix
//  3: the number of columns in the second matrix

int main(int argc, char *argv[]) {
    srand(time(NULL));

    int A_row_count = 2;
    int m = 3;
    int B_col_count = 1;

    if (3 < argc) { // only use user provided values if all three were provided
        A_row_count = confirm_input(A_row_count, argv[1]);
        m = confirm_input(m, argv[2]);
        B_col_count = confirm_input(B_col_count, argv[3]);
    }

    // "matrices" are stored as 1D arrays
    // RowCol structure is used to keep track of the row col values for each 1D array
    int *A = (int*)malloc((A_row_count * m) * sizeof(int));
    struct RowCol row_col_A = initalize_matrix_values(A, 1, A_row_count, m);

    int *B = (int*)malloc((m * B_col_count) * sizeof(int));
    struct RowCol row_col_B = initalize_matrix_values(B, 1, m, B_col_count);

    int *C = (int*)malloc((A_row_count * B_col_count) * sizeof(int));
    struct RowCol row_col_C = initalize_matrix_values(C, 0, A_row_count, B_col_count);

    int *C_parallel = (int*)malloc((A_row_count * B_col_count) * sizeof(int));
    struct RowCol row_col_C_parallel = initalize_matrix_values(C_parallel, 0, A_row_count, B_col_count);

    clock_t start_serial = clock();
    serial_multiplication(A, B, C, row_col_C, m);
    clock_t end_serial = clock();

    print_to_file(A, row_col_A, output_A_file);
    print_to_file(B, row_col_B, output_B_file);
    print_to_file(C, row_col_C, output_C_serial_file);

    clock_t start_parallel = clock();
    parallel_multiplication(A, B, C_parallel, row_col_C_parallel, m);
    clock_t end_parallel = clock();
    print_to_file(C_parallel, row_col_C_parallel, output_C_parallel_file);

    printf("\nTime for (serial) multiplication: %f\n", ((double) (end_serial - start_serial)) / CLOCKS_PER_SEC);
    printf("\nTime for (parallel) multiplication: %f\n", ((double) (end_parallel - start_parallel)) / CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);
    free(C_parallel);
}