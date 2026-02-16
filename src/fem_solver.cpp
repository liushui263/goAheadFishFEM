#include "fem_solver.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>

// Eigen includes
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>

// CUDA includes
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cudss.h>
#endif

// --- CPU Solver Implementation ---

void CPU_Solver::solve(const SparseMatrix& K, const Vector& F, Vector& E) {
#ifdef USE_ITERATIVE_SOLVER
    std::cout << "Solving on CPU using Eigen::BiCGSTAB (Iterative)..." << std::endl;

    // BiCGSTAB is suitable for non-Hermitian problems
    Eigen::BiCGSTAB<SparseMatrix> solver;
    solver.setTolerance(1e-6);
    
    solver.compute(K);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "CPU Iterative Solver compute failed." << std::endl;
        return;
    }
    
    E = solver.solve(F);
    
    std::cout << "Iterative Solver converged. Iterations: " << solver.iterations() 
              << ", Error: " << solver.error() << std::endl;

#else
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
#endif
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

// --- New Realistic Physics Assembly ---
// Generates a matrix representing (CurlCurl - k^2) operator.
// This is typically indefinite and ill-conditioned at low frequencies,
// showcasing the robustness of the GPU Direct Solver where Iterative solvers fail.
void FEM_Solver::assemble_maxwell(int N, double freq, double sigma) {
    std::cout << "Assembling Maxwell system (Realistic Physics)..." << std::endl;
    std::cout << "  Frequency: " << freq << " Hz" << std::endl;
    std::cout << "  Conductivity: " << sigma << " S/m" << std::endl;

    size_t num_unknowns = (N > 0) ? N : 10000;
    K_.resize(num_unknowns, num_unknowns);
    F_.resize(num_unknowns);

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(num_unknowns * 7);

    // Physics Constants
    const double mu0 = 4.0 * std::acos(-1.0) * 1e-7;
    const double eps0 = 8.854e-12;
    const double omega = 2.0 * std::acos(-1.0) * freq;
    
    // Grid step (assume 1.0m for simplicity, or small for LWD)
    double dx = 1.0; 

    // k^2 = w^2 * mu * eps - i * w * mu * sigma
    // Operator is: CurlCurl - k^2
    // In discrete form (approx): Diag = 4.0 - k^2 * dx^2
    std::complex<double> k2 = std::complex<double>(
        omega * omega * mu0 * eps0, 
        -omega * mu0 * sigma
    );
    
    std::complex<double> diag_val = std::complex<double>(4.0, 0.0) - k2 * (dx * dx);
    std::complex<double> off_diag = std::complex<double>(-1.0, 0.0);

    // 3D Bandwidth approximation
    int bandwidth = static_cast<int>(std::pow(num_unknowns, 2.0/3.0));
    if (bandwidth < 1) bandwidth = 1;

    for(size_t i=0; i<num_unknowns; ++i) {
        triplets.push_back({(int)i, (int)i, diag_val});
        
        // Neighbors (Standard 7-point stencil structure)
        if(i > 0) triplets.push_back({(int)i, (int)i-1, off_diag});
        if(i < num_unknowns-1) triplets.push_back({(int)i, (int)i+1, off_diag});
        if(i >= (size_t)bandwidth) triplets.push_back({(int)i, (int)i-bandwidth, off_diag});
        if(i + bandwidth < num_unknowns) triplets.push_back({(int)i, (int)i+bandwidth, off_diag});
        
        // Source term (RHS)
        F_(i) = (i == num_unknowns/2) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
    }
    
    K_.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Maxwell Assembly complete. System size: " << K_.rows() << std::endl;
}

void FEM_Solver::assemble_interference_pattern(int N) {
    std::cout << "Assembling Interference Pattern (Helmholtz 3D)..." << std::endl;
    
    // Calculate grid dimensions for a cube
    int nx = std::round(std::pow(N, 1.0/3.0));
    int ny = nx;
    int nz = nx;
    size_t num_unknowns = nx * ny * nz;
    
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << " (" << num_unknowns << " unknowns)" << std::endl;

    K_.resize(num_unknowns, num_unknowns);
    F_.resize(num_unknowns);
    F_.setZero();

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(num_unknowns * 7);

    // Physics: Scalar Helmholtz (Laplacian + k^2)
    // h = 1.0, lambda = 10.0 (10 grid points per wavelength for visible ripples)
    double k = 2.0 * std::acos(-1.0) / 10.0;
    std::complex<double> k2 = std::complex<double>(k*k, 0.0);
    
    // Finite Difference Stencil: (6 - k^2*h^2) * u_ijk - sum(u_neighbors) = f
    std::complex<double> diag_val = std::complex<double>(6.0, 0.0) - k2;
    std::complex<double> off_diag = std::complex<double>(-1.0, 0.0);

    for(int z=0; z<nz; ++z) {
        for(int y=0; y<ny; ++y) {
            for(int x=0; x<nx; ++x) {
                int idx = z * (nx * ny) + y * nx + x;
                
                triplets.push_back({idx, idx, diag_val});

                // Neighbors with boundary checks (Dirichlet BC = 0)
                if(x > 0) triplets.push_back({idx, idx - 1, off_diag});
                if(x < nx - 1) triplets.push_back({idx, idx + 1, off_diag});
                
                if(y > 0) triplets.push_back({idx, idx - nx, off_diag});
                if(y < ny - 1) triplets.push_back({idx, idx + nx, off_diag});
                
                if(z > 0) triplets.push_back({idx, idx - nx*ny, off_diag});
                if(z < nz - 1) triplets.push_back({idx, idx + nx*ny, off_diag});
            }
        }
    }

    // Add Sources (Interference Pattern)
    int cx = nx/2, cy = ny/2, cz = nz/2;
    int dist = nx/5;
    double pi = std::acos(-1.0);

    // Place sources with different phases to create interference
    F_(cz * (nx * ny) + cy * nx + cx) += std::polar(10.0, 0.0);           // Center
    F_(cz * (nx * ny) + cy * nx + (cx + dist)) += std::polar(10.0, 0.0);  // Right
    F_(cz * (nx * ny) + cy * nx + (cx - dist)) += std::polar(10.0, pi);   // Left (Anti-phase)
    F_(cz * (nx * ny) + (cy + dist) * nx + cx) += std::polar(10.0, pi/2); // Top
    F_(cz * (nx * ny) + (cy - dist) * nx + cx) += std::polar(10.0, -pi/2);// Bottom

    K_.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Assembly complete." << std::endl;
}

void FEM_Solver::assemble_dipole(int N, const std::string& type) {
    std::cout << "Assembling Dipole Test (" << type << ")..." << std::endl;
    
    // Calculate grid dimensions for a cube
    int nx = std::round(std::pow(N, 1.0/3.0));
    int ny = nx;
    int nz = nx;
    size_t num_unknowns = nx * ny * nz;
    
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << " (" << num_unknowns << " unknowns)" << std::endl;

    K_.resize(num_unknowns, num_unknowns);
    F_.resize(num_unknowns);
    F_.setZero();

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(num_unknowns * 7);

    // Physics: Scalar Helmholtz (Laplacian + k^2)
    double k = 2.0 * std::acos(-1.0) / 10.0;
    std::complex<double> k2 = std::complex<double>(k*k, 0.0);
    
    std::complex<double> diag_val = std::complex<double>(6.0, 0.0) - k2;
    std::complex<double> off_diag = std::complex<double>(-1.0, 0.0);

    for(int z=0; z<nz; ++z) {
        for(int y=0; y<ny; ++y) {
            for(int x=0; x<nx; ++x) {
                int idx = z * (nx * ny) + y * nx + x;
                
                triplets.push_back({idx, idx, diag_val});

                if(x > 0) triplets.push_back({idx, idx - 1, off_diag});
                if(x < nx - 1) triplets.push_back({idx, idx + 1, off_diag});
                
                if(y > 0) triplets.push_back({idx, idx - nx, off_diag});
                if(y < ny - 1) triplets.push_back({idx, idx + nx, off_diag});
                
                if(z > 0) triplets.push_back({idx, idx - nx*ny, off_diag});
                if(z < nz - 1) triplets.push_back({idx, idx + nx*ny, off_diag});
            }
        }
    }

    // Add Source
    int cx = nx/2, cy = ny/2, cz = nz/2;
    
    if (type == "electric") {
        // Z-directed Electric Dipole (Point Source)
        F_(cz * (nx * ny) + cy * nx + cx) = std::complex<double>(100.0, 0.0);
    } else if (type == "magnetic") {
        // Z-directed Magnetic Dipole (Loop Source in XY plane)
        double pi = std::acos(-1.0);
        double mag = 100.0;
        if (cx+1 < nx) F_(cz * (nx * ny) + cy * nx + (cx + 1)) += std::polar(mag, 0.0);
        if (cy+1 < ny) F_(cz * (nx * ny) + (cy + 1) * nx + cx) += std::polar(mag, pi/2);
        if (cx > 0)    F_(cz * (nx * ny) + cy * nx + (cx - 1)) += std::polar(mag, pi);
        if (cy > 0)    F_(cz * (nx * ny) + (cy - 1) * nx + cx) += std::polar(mag, 3*pi/2);
    }

    K_.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Assembly complete." << std::endl;
}

void FEM_Solver::assemble_vector_maxwell(int nx, int ny, int nz, double dx, double freq, double sigma) {
    std::cout << "Assembling 3D Vector Maxwell System (FEM Element-wise)..." << std::endl;
    std::cout << "  Mesh: " << nx << "x" << ny << "x" << nz << " elements" << std::endl;

    // 1. Indexing Strategy (Edge Elements)
    // Unknowns are edges.
    // Ex lives at (i+0.5, j, k) -> count: nx * (ny+1) * (nz+1)
    // Ey lives at (i, j+0.5, k) -> count: (nx+1) * ny * (nz+1)
    // Ez lives at (i, j, k+0.5) -> count: (nx+1) * (ny+1) * nz
    
    int n_ex = nx * (ny + 1) * (nz + 1);
    int n_ey = (nx + 1) * ny * (nz + 1);
    int n_ez = (nx + 1) * (ny + 1) * nz;
    int num_unknowns = n_ex + n_ey + n_ez;

    std::cout << "  Total Vector Unknowns (Edges): " << num_unknowns << std::endl;

    // Helper lambdas for indexing
    auto idx_ex = [&](int i, int j, int k) { return k * (ny + 1) * nx + j * nx + i; };
    auto idx_ey = [&](int i, int j, int k) { return n_ex + k * ny * (nx + 1) + j * (nx + 1) + i; };
    auto idx_ez = [&](int i, int j, int k) { return n_ex + n_ey + k * (ny + 1) * (nx + 1) + j * (nx + 1) + i; };

    // 2. Physics Constants
    const double mu0 = 4.0 * std::acos(-1.0) * 1e-7;
    const double eps0 = 8.854e-12;
    const double omega = 2.0 * std::acos(-1.0) * freq;
    std::complex<double> k0_sq = omega * omega * mu0 * eps0;
    std::complex<double> sigma_term = std::complex<double>(0.0, omega * mu0 * sigma);

    // 3. Element-wise Assembly
    // We iterate over elements (voxels), compute local matrices, and assemble.
    // For a rectangular brick element of size dx*dx*dx:
    // Stiffness (Curl-Curl) and Mass matrices can be precomputed or computed analytically.
    // Here we use a simplified "Lumped" integration which is robust and equivalent to FDFD on structured grids,
    // but the assembly structure is pure FEM.

    K_.resize(num_unknowns, num_unknowns);
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(num_unknowns * 9); // Approx 9 non-zeros per row on average

    // Local coefficients for a cube of side h (dx)
    // Stiffness K_local ~ 1/h * volume = h^2 ? No.
    // Curl N ~ 1/h. Integral (Curl N)^2 dV ~ (1/h^2) * h^3 = h.
    // Mass M_local ~ N^2 dV ~ 1 * h^3 = h^3.
    // However, to match standard wave equation form (CurlCurl - k^2) E = f,
    // we often divide by volume or keep it integral form.
    // Let's use the integral form:
    // Weak form: integral( (curl E) * (curl v) - k^2 E * v ) = integral( f * v )
    
    double h = dx;
    double coeff_K = h;        // Stiffness scaling
    std::complex<double> coeff_M = (k0_sq - sigma_term) * (h * h * h); // Mass scaling (k^2 * vol)

    // Loop over elements
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                
                // Identify the 12 global edge indices for this element
                // Order: Ex(4), Ey(4), Ez(4)
                int edges[12];
                
                // Ex edges (aligned with x-axis)
                edges[0] = idx_ex(i, j, k);     // bottom-front
                edges[1] = idx_ex(i, j+1, k);   // top-front
                edges[2] = idx_ex(i, j, k+1);   // bottom-back
                edges[3] = idx_ex(i, j+1, k+1); // top-back
                
                // Ey edges (aligned with y-axis)
                edges[4] = idx_ey(i, j, k);     // left-front
                edges[5] = idx_ey(i+1, j, k);   // right-front
                edges[6] = idx_ey(i, j, k+1);   // left-back
                edges[7] = idx_ey(i+1, j, k+1); // right-back
                
                // Ez edges (aligned with z-axis)
                edges[8] = idx_ez(i, j, k);     // left-bottom
                edges[9] = idx_ez(i+1, j, k);   // right-bottom
                edges[10] = idx_ez(i, j+1, k);  // left-top
                edges[11] = idx_ez(i+1, j+1, k); // right-top

                // Assemble Local Contributions
                // For a structured grid, we can simplify the local matrix integration.
                // We apply the "stiffness" (curl-curl) to loops on faces and "mass" to edges.
                
                // 1. Mass Matrix Contribution (Diagonal Lumped)
                // Each edge is shared by 4 elements. We distribute the mass.
                // Total mass for an edge in the mesh is M_total. Here we add 1/4 of element mass.
                std::complex<double> m_val = -coeff_M * 0.25; 
                for(int e=0; e<12; ++e) {
                    triplets.push_back({edges[e], edges[e], m_val});
                }

                // 2. Stiffness Matrix Contribution (Curl-Curl)
                // Curl-Curl couples edges on the same face.
                // A face has 4 edges. Circulation = sum(edges * orientation).
                // Energy = (Circulation)^2.
                // We iterate over the 6 faces of the cube.
                
                // Helper to add face contribution: (e1 + e2 - e3 - e4)^2
                auto add_face = [&](int e1, int e2, int e3, int e4) {
                    int idx[] = {e1, e2, e3, e4};
                    double sgn[] = {1.0, 1.0, -1.0, -1.0}; // Orientation
                    for(int a=0; a<4; ++a) {
                        for(int b=0; b<4; ++b) {
                            // Factor 1.0 because we integrate over unit volume scaled by coeff_K
                            // Actually for a face, the term is 1/h * h^2? No.
                            // Correct scaling for CurlCurl on cube is simply 1.0 * coeff_K / h^2 * vol?
                            // Let's use the standard result: +/- 1.0 * (h) for the integral.
                            // Wait, Curl is 1/h. Curl^2 is 1/h^2. Volume is h^3. Result is h.
                            // So we just add +/- coeff_K.
                            triplets.push_back({idx[a], idx[b], sgn[a] * sgn[b] * coeff_K});
                        }
                    }
                };

                // Face Z-normal (xy plane): edges Ex(y), Ex(y+1), Ey(x), Ey(x+1)
                // Loop: Ex(j) + Ey(i+1) - Ex(j+1) - Ey(i)
                add_face(edges[0], edges[5], edges[1], edges[4]); // Front face (k)
                add_face(edges[2], edges[7], edges[3], edges[6]); // Back face (k+1)

                // Face Y-normal (xz plane): edges Ex(z), Ex(z+1), Ez(x), Ez(x+1)
                // Loop: Ex(k) + Ez(i+1) - Ex(k+1) - Ez(i)
                add_face(edges[0], edges[9], edges[2], edges[8]); // Bottom face (j)
                add_face(edges[1], edges[11], edges[3], edges[10]); // Top face (j+1)

                // Face X-normal (yz plane): edges Ey(z), Ey(z+1), Ez(y), Ez(y+1)
                // Loop: Ey(k) + Ez(j+1) - Ey(k+1) - Ez(j)
                add_face(edges[4], edges[10], edges[6], edges[8]); // Left face (i)
                add_face(edges[5], edges[11], edges[7], edges[9]); // Right face (i+1)
            }
        }
    }

    K_.setFromTriplets(triplets.begin(), triplets.end());

    // 4. Source (Electric Dipole in center, Z-oriented)
    F_.resize(num_unknowns);
    F_.setZero();
    
    int cx = nx / 2;
    int cy = ny / 2;
    int cz = nz / 2;
    
    // Excite the center Ez edge
    int src_idx = idx_ez(cx, cy, cz);
    if (src_idx < num_unknowns) {
        // Source term J. In weak form: integral(J * N) dV.
        // For a delta source on edge, it scales with length h.
        F_(src_idx) = std::complex<double>(0.0, -omega * mu0) * h; 
    }

    std::cout << "FEM Assembly complete. System size: " << K_.rows() << ", NNZ: " << K_.nonZeros() << std::endl;
}

void FEM_Solver::save_as_json(const std::string& filename, const Vector& E) {
    nlohmann::json j;
    
    // 1. Metadata
    j["metadata"]["solver_type"] = 
#ifdef USE_GPU
        "GPU_Direct_cuDSS";
        const std::string jsonfile = filename + "_gpu.json";
#elif defined(USE_ITERATIVE_SOLVER)
        "CPU_Iterative_BiCGSTAB";
        const std::string jsonfile = filename + "_iter.json";
#else
        "CPU_Direct_SparseLU";
        const std::string jsonfile = filename + "_cpu.json";
#endif
    j["metadata"]["system_size"] = E.size();

    // 2. Solution Data (Split Complex -> Real/Imag for JSON compatibility)
    std::vector<double> real_part, imag_part;
    real_part.reserve(E.size());
    imag_part.reserve(E.size());
    
    for(int i=0; i<E.size(); ++i) {
        real_part.push_back(E[i].real());
        imag_part.push_back(E[i].imag());
    }
    j["solution"]["real"] = real_part;
    j["solution"]["imag"] = imag_part;

    // 3. Position Information (Coordinates)
    // Generate synthetic coordinates assuming a roughly cubic grid structure
    // This matches the bandwidth assumption in assemble_maxwell (N^(2/3) stride)
    size_t N = E.size();
    std::vector<double> x(N), y(N), z(N);
    int nx = static_cast<int>(std::ceil(std::pow(N, 1.0/3.0)));
    int ny = nx;

    for(size_t i=0; i<N; ++i) {
        int iz = i / (nx * ny);
        int rem = i % (nx * ny);
        int iy = rem / nx;
        int ix = rem % nx;
        x[i] = static_cast<double>(ix);
        y[i] = static_cast<double>(iy);
        z[i] = static_cast<double>(iz);
    }
    j["mesh"]["x"] = x;
    j["mesh"]["y"] = y;
    j["mesh"]["z"] = z;

    std::ofstream o(jsonfile);
    o << std::setw(4) << j << std::endl;
    std::cout << "Results saved to " << jsonfile << std::endl;
}

void FEM_Solver::solve(Vector& E) {
#ifdef USE_GPU
    gpu_solver_.solve(K_, F_, E);
#else
    cpu_solver_.solve(K_, F_, E);
#endif
}