# Matrix Multiplication Parallelization

CUDA project comparing serial vs parallel matrix multiplication for CSI-4650 (Dr. Debnath, Oakland University).

## Team
Austin Toma, Simon Griemert, Anton Shemshur, Rory Strachan, Damjan Dukoski, Kole Micakaj

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (provides `nvcc`)
- Windows: Visual Studio Build Tools with "Desktop development with C++" (for `cl.exe`). Run commands from the **x64 Native Tools Command Prompt**.
- Linux: `gcc`/`g++`

## Build & Run
```
nvcc MatrixMultiplication.CU -o matmul
matmul.exe 1000 1000 1000
```
On Linux: `./matmul 1000 1000 1000`

Make sure an `outputs/` folder exists next to the binary, the program writes result files there.

## Arguments
`matmul <A_rows> <M> <B_cols>`
- `A_rows` — rows of matrix A
- `M` — columns of A / rows of B (shared dim)
- `B_cols` — columns of B

No args defaults to a 2x3 * 3x1 example.

## Variants Implemented
- **Serial** — standard triple-loop CPU multiplication.
- **Parallel (outer)** — one CUDA thread per output element of C.
- **Parallel (inner)** — each thread computes one term of a dot product, accumulated via `atomicAdd`. Launches one kernel per output cell.

## Output Files (`outputs/`)
- `A.txt`, `B.txt` — input matrices
- `C_serial.txt`, `C_parallel.txt`, `C_parallel_inner.txt` — results per variant
