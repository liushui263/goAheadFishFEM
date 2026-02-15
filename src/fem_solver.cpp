#include "fem_solver.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

// Eigen includes
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// CUDA includes
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cudss.h>
#endif

// --- CPU Solver Implementation ---

void CPU_Solver::solve(const SparseMatrix& K, const Vector& F, Vector& E) {
    std::cout << "Solving on CPU using Eigen::SparseLU..." << std::endl;
    
    // Maxwell system is typically complex symmetric (non-Hermitian).
    // SparseLU is a robust direct solver for general square matrices.
    Eigen::SparseLU<SparseMatrix> solver;
    solver.analyzePattern(K);
    solver.factorize(K);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "CPU Solver factorization failed: " << solver.lastErrorMessage() << std::endl;
        return;
    }
    
    E = solver.solve(F);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "CPU Solver solve failed: " << solver.lastErrorMessage() << std::endl;
    }
}

// --- GPU Solver Implementation ---

#ifdef USE_GPU
void checkCudaErrors(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << std::endl;
    }
}

void checkCudssErrors(cudssStatus_t status, const char* func, const char* file, int line) {
    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "cuDSS error at " << file << ":" << line << " status=" << status << " \"" << func << "\" \n";
    }
}

#define CUDA_CHECK(val) checkCudaErrors((val), #val, __FILE__, __LINE__)
#define CUDSS_CHECK(val) checkCudssErrors((val), #val, __FILE__, __LINE__)

void GPU_Solver::solve(const SparseMatrix& K, const Vector& F, Vector& E) {
    std::cout << "Solving on GPU using cuDSS..." << std::endl;

    // 1. Convert Eigen SparseMatrix (CSC) to CSR format for cuDSS
    // Eigen defaults to Column Major (CSC). cuDSS supports CSR.
    // We create a temporary RowMajor copy to get CSR layout.
    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> K_csr = K;
    
    int64_t rows = K_csr.rows();
    int64_t cols = K_csr.cols();
    int64_t nnz = K_csr.nonZeros();
    
    // 2. Allocate Device Memory
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
    cuDoubleComplex* d_csr_val = nullptr;
    cuDoubleComplex* d_b = nullptr;
    cuDoubleComplex* d_x = nullptr;

    CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_val, nnz * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_b, rows * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_x, rows * sizeof(cuDoubleComplex)));

    // 3. Copy data to Device
    // Note: Eigen uses int for indices by default. cuDSS expects 32-bit indices here.
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, K_csr.outerIndexPtr(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_ind, K_csr.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_val, K_csr.valuePtr(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, F.data(), rows * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // 4. cuDSS Setup
    cudssHandle_t handle;
    CUDSS_CHECK(cudssCreate(&handle));

    cudssConfig_t config;
    CUDSS_CHECK(cudssConfigCreate(&config));

    cudssData_t solver_data;
    CUDSS_CHECK(cudssDataCreate(handle, &solver_data));

    cudssMatrix_t matrix_A, vector_b, vector_x;
    
       // Matrix A: General non-Hermitian (Complex Symmetric)
    // Note: cuDSS 0.7.1 uses standard CUDA_R_32I for indices and CUDA_C_64F for values
    CUDSS_CHECK(cudssMatrixCreateCsr(&matrix_A, rows, cols, nnz, 
                                     d_csr_row_ptr, nullptr, d_csr_col_ind, d_csr_val, 
                                     CUDA_R_32I, CUDA_C_64F, CUDSS_MTYPE_GENERAL, 
                                     CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    // Vectors b and x
    CUDSS_CHECK(cudssMatrixCreateDn(&vector_b, rows, 1, rows, d_b, 
                                    CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&vector_x, rows, 1, rows, d_x, 
                                    CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));


    // 5. Analysis, Factorization, Solve
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solver_data, matrix_A, vector_x, vector_b));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solver_data, matrix_A, vector_x, vector_b));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solver_data, matrix_A, vector_x, vector_b));

    // 6. Retrieve Result
    E.resize(rows);
    CUDA_CHECK(cudaMemcpy(E.data(), d_x, rows * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // 7. Cleanup
    CUDSS_CHECK(cudssMatrixDestroy(matrix_A));
    CUDSS_CHECK(cudssMatrixDestroy(vector_b));
    CUDSS_CHECK(cudssMatrixDestroy(vector_x));
    CUDSS_CHECK(cudssDataDestroy(handle, solver_data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));

    CUDA_CHECK(cudaFree(d_csr_row_ptr));
    CUDA_CHECK(cudaFree(d_csr_col_ind));
    CUDA_CHECK(cudaFree(d_csr_val));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
}
#endif

// --- FEM_Solver Implementation ---

FEM_Solver::FEM_Solver(Mesh* mesh, Material* material, Sources* sources)
    : mesh_(mesh), material_(material), sources_(sources) {
}

void FEM_Solver::assemble() {
    std::cout << "Assembling system matrix K and load vector F..." << std::endl;
    
    // Placeholder assembly logic based on FEM3D_for_LWD_P03.tex
    // K_ij = integral( (curl Ni).mu^-1.(curl Nj) - w^2 Ni.eps.Nj )
    // F_i  = -j*w*Ni(r0).p + (curl Ni)(r0).m
    
    // Use actual mesh size if available, otherwise fallback to dummy for testing without mesh
    size_t num_unknowns = (mesh_ && !mesh_->edges.empty()) ? mesh_->edges.size() : 100;
    
    K_.resize(num_unknowns, num_unknowns);
    F_.resize(num_unknowns);
    
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(num_unknowns * 7); // Reserve for ~7-point stencil

    // Construct a synthetic 3D-like sparse structure for benchmarking
    // (Diagonal + Neighbors + Far Neighbors to simulate 3D bandwidth)
    int bandwidth = static_cast<int>(std::pow(num_unknowns, 2.0/3.0)); // Approx N^2 stride for N^3 grid
    if (bandwidth < 1) bandwidth = 1;

    for(size_t i=0; i<num_unknowns; ++i) {
        // Diagonal (ensure diagonal dominance for stability)
        triplets.push_back({(int)i, (int)i, std::complex<double>(6.0, 0.5)});
        
        // Immediate neighbors (1D-like)
        if(i > 0) triplets.push_back({(int)i, (int)i-1, std::complex<double>(-1.0, 0.0)});
        if(i < num_unknowns-1) triplets.push_back({(int)i, (int)i+1, std::complex<double>(-1.0, 0.0)});
        
        // Far neighbors (Simulate 3D connectivity)
        if(i >= (size_t)bandwidth) triplets.push_back({(int)i, (int)i-bandwidth, std::complex<double>(-1.0, 0.0)});
        if(i + bandwidth < num_unknowns) triplets.push_back({(int)i, (int)i+bandwidth, std::complex<double>(-1.0, 0.0)});
        
        F_(i) = std::complex<double>(1.0, 0.0);
    }
    K_.setFromTriplets(triplets.begin(), triplets.end());
    
    std::cout << "Assembly complete. System size: " << K_.rows() << ", NNZ: " << K_.nonZeros() << std::endl;
}

void FEM_Solver::solve(Vector& E) {
#ifdef USE_GPU
    gpu_solver_.solve(K_, F_, E);
#else
    cpu_solver_.solve(K_, F_, E);
#endif
}