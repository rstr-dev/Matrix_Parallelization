#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int VALUEMAX = 10;

char output_A_file[] = "outputs/A.txt";
char output_B_file[] = "outputs/B.txt";
char output_C_serial_file[] = "outputs/C_serial.txt";
char output_C_parallel_outer_file[] = "outputs/C_parallel_outer.txt";
char output_C_parallel_inner_file[] = "outputs/C_parallel_inner.txt";

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

struct RowCol initalize_matrix_values(int *matrix, int type, int row, int col) {
    int N = row * col;

    for (int i = 0; i < N; i++) {
        if (type == 0) {
            matrix[i] = 0;
        } else {
            matrix[i] = rand() % (VALUEMAX + 1);
        }
    }

    struct RowCol row_col = {row, col};

    return row_col;
}

int confirm_input(int default_value, char user_provided[]) {
    int n;
    char ch;
    int status;

    status = sscanf(user_provided, "%d%c", &n, &ch);
    if (status != 1) {
        printf("The provided user inputer was not an int, the program will use the default value.\n");

        return default_value;
    }

    return atoi(user_provided);
}

__host__ __device__ int row_col_to_index(int row, int col, int row_width) {
    return (row * row_width) + col;
}

__host__ __device__ struct RowCol index_to_row_col(int index, int row_width) {
    struct RowCol values;

    values.row = (index / row_width);
    values.col = (index % row_width);

    return values;
}

int compare_matrices(int* X, int* Y, int N) {
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (X[i] != Y[i]) {
            mismatches++;
        }
    }
    return mismatches;
}

int dot_product(int* A, int* B, struct RowCol row_col_C, int index, int m) {
    int sum = 0;

    struct RowCol current_row_col = index_to_row_col(index, row_col_C.col);

    for (int i = 0; i < m; i++) {
        int A_index = row_col_to_index(current_row_col.row, i, m);
        int B_index = row_col_to_index(i, current_row_col.col, row_col_C.col);

        sum += (A[A_index] * B[B_index]);
    }

    return sum;
}

void serial_multiplication(int* A, int* B, int* C, struct RowCol row_col_C, int m) {
    for (int i = 0; i < (row_col_C.row * row_col_C.col); i++) {
        C[i] = dot_product(A, B, row_col_C, i, m);
    }
}

__global__ void kernel_dot_product_outer(int* A, int* B, int* C, struct RowCol row_col_C, int m) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int N = row_col_C.row * row_col_C.col;
    if (index >= N) return;

    int sum = 0;

    struct RowCol current_row_col = index_to_row_col(index, row_col_C.col);

    for (int i = 0; i < m; i++) {
        int A_index = row_col_to_index(current_row_col.row, i, m);
        int B_index = row_col_to_index(i, current_row_col.col, row_col_C.col);

        sum += (A[A_index] * B[B_index]);
    }

    C[index] = sum;
}

void parallel_multiplication_outer(int *A, int *B, int *C, struct RowCol row_col_C, int m, float *kernel_ms_out) {
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel-only GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_dot_product_outer<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, row_col_C, m);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_ms_out, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Keep host timing accurate
    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

// Inner-loop kernel: one thread per dot-product term
__global__ void kernel_dot_product_inner(int* A, int* B, int* C, struct RowCol row_col_C, struct RowCol current_row_col, int m, int current_C_index) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= m) return;

    int A_index = row_col_to_index(current_row_col.row, index, m);
    int B_index = row_col_to_index(index, current_row_col.col, row_col_C.col);

    atomicAdd(&C[current_C_index], A[A_index] * B[B_index]);
}

void parallel_multiplication_inner(int *A, int *B, int *C, struct RowCol row_col_C, int m, float *kernel_ms_out) {
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
    // atomicAdd needs a zeroed output buffer
    cudaMemset(dev_C, 0, size_C);

    int threadsPerBlock = 256;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int current_C_index = 0; current_C_index < N; current_C_index++) {
        struct RowCol current_row_col = index_to_row_col(current_C_index, row_col_C.col);
        kernel_dot_product_inner<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, row_col_C, current_row_col, m, current_C_index);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_ms_out, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    int A_row_count = 2;
    int m = 3;
    int B_col_count = 1;

    if (3 < argc) {
        A_row_count = confirm_input(A_row_count, argv[1]);
        m = confirm_input(m, argv[2]);
        B_col_count = confirm_input(B_col_count, argv[3]);
    }

    // Matrices are stored as 1D arrays
    int *A = (int*)malloc((A_row_count * m) * sizeof(int));
    struct RowCol row_col_A = initalize_matrix_values(A, 1, A_row_count, m);

    int *B = (int*)malloc((m * B_col_count) * sizeof(int));
    struct RowCol row_col_B = initalize_matrix_values(B, 1, m, B_col_count);

    int *C = (int*)malloc((A_row_count * B_col_count) * sizeof(int));
    struct RowCol row_col_C = initalize_matrix_values(C, 0, A_row_count, B_col_count);

    int *C_parallel_outer = (int*)malloc((A_row_count * B_col_count) * sizeof(int));
    struct RowCol row_col_C_parallel_outer = initalize_matrix_values(C_parallel_outer, 0, A_row_count, B_col_count);

    int *C_parallel_inner = (int*)malloc((A_row_count * B_col_count) * sizeof(int));
    struct RowCol row_col_C_parallel_inner = initalize_matrix_values(C_parallel_inner, 0, A_row_count, B_col_count);

    clock_t start_serial = clock();
    serial_multiplication(A, B, C, row_col_C, m);
    clock_t end_serial = clock();

    print_to_file(A, row_col_A, output_A_file);
    print_to_file(B, row_col_B, output_B_file);
    print_to_file(C, row_col_C, output_C_serial_file);

    float kernel_outer_ms = 0.0f;
    clock_t start_parallel_outer = clock();
    parallel_multiplication_outer(A, B, C_parallel_outer, row_col_C_parallel_outer, m, &kernel_outer_ms);
    clock_t end_parallel_outer = clock();
    print_to_file(C_parallel_outer, row_col_C_parallel_outer, output_C_parallel_outer_file);

    float kernel_inner_ms = 0.0f;
    clock_t start_parallel_inner = clock();
    parallel_multiplication_inner(A, B, C_parallel_inner, row_col_C_parallel_inner, m, &kernel_inner_ms);
    clock_t end_parallel_inner = clock();
    print_to_file(C_parallel_inner, row_col_C_parallel_inner, output_C_parallel_inner_file);

    int N = A_row_count * B_col_count;
    double serial_s   = ((double)(end_serial   - start_serial))   / CLOCKS_PER_SEC;
    double parallel_outer_s = ((double)(end_parallel_outer - start_parallel_outer)) / CLOCKS_PER_SEC;
    double parallel_inner_s = ((double)(end_parallel_inner - start_parallel_inner)) / CLOCKS_PER_SEC;

    printf("\nTime for (serial) multiplication: %.5f s\n", serial_s);
    printf("Time for (parallel inner) kernel only: %.5f s\n", kernel_inner_ms/1000);
    printf("Time for (parallel inner) total (compute + memcpy): %.5f s\n", parallel_inner_s);
    printf("Time for (parallel outer) kernel only: %.5f s\n", kernel_outer_ms/1000);
    printf("Time for (parallel outer) total (compute + memcpy): %.5f s\n", parallel_outer_s);

    int mismatches_outer = compare_matrices(C, C_parallel_outer, N);
    int mismatches_inner = compare_matrices(C, C_parallel_inner, N);

    if (mismatches_outer == 0) {
        printf("\nCORRECTNESS (outer parallel vs serial): PASS\n");
    } else {
        printf("\nCORRECTNESS (outer parallel vs serial): FAIL (%d mismatches)\n", mismatches_outer);
    }

    if (mismatches_inner == 0) {
        printf("CORRECTNESS (inner parallel vs serial): PASS\n");
    } else {
        printf("CORRECTNESS (inner parallel vs serial): FAIL (%d mismatches)\n", mismatches_inner);
    }

    free(A);
    free(B);
    free(C);
    free(C_parallel_outer);
    free(C_parallel_inner);
}
