//====================================================
// FEM Maxwell Solver - src/ files
// Supports both CPU (Eigen/CSparse) and GPU (cuDSS) solvers
// Compile-time selection via CMake option USE_GPU
//====================================================

//================= fem_solver.h =====================
#pragma once
#include "mesh.h"
#include "material.h"
#include "sources.h"
#include "utils.h"
#include <vector>

class SparseMatrix;
class Vector;

class Solver {
public:
    virtual void solve(const SparseMatrix& K, const Vector& F, Vector& E) = 0;
    virtual ~Solver() {}
};

class CPU_Solver : public Solver {
public:
    void solve(const SparseMatrix& K, const Vector& F, Vector& E) override;
};

#ifdef USE_GPU
#include <cudss.h>
class GPU_Solver : public Solver {
public:
    void solve(const SparseMatrix& K, const Vector& F, Vector& E) override;
};
#endif

class FEM_Solver {
public:
    FEM_Solver(Mesh* mesh, Material* material, Sources* sources);
    void assemble();
    void solve(Vector& E);

private:
    Mesh* mesh_;
    Material* material_;
    Sources* sources_;
    SparseMatrix K_;
    Vector F_;
#ifdef USE_GPU
    GPU_Solver gpu_solver_;
#else
    CPU_Solver cpu_solver_;
#endif
};


//================= fem_solver.cpp =====================
#include "fem_solver.h"
#include <iostream>

FEM_Solver::FEM_Solver(Mesh* mesh, Material* material, Sources* sources)
    : mesh_(mesh), material_(material), sources_(sources) {}

void FEM_Solver::assemble() {
    // Loop over elements, assemble K_ and F_
    // Include anisotropic material tensor handling
    // Include electric dipole p and magnetic dipole m
    std::cout << "Assembling FEM matrices and source vectors..." << std::endl;
    // Pseudo-code: element loops -> integrate curl(N_i) * mu^{-1} * curl(N_j) - omega^2 N_i*epsilon*N_j
    // Add source contributions at r0
}

void FEM_Solver::solve(Vector& E) {
#ifdef USE_GPU
    std::cout << "Solving using GPU/cuDSS..." << std::endl;
    gpu_solver_.solve(K_, F_, E);
#else
    std::cout << "Solving using CPU sparse solver..." << std::endl;
    cpu_solver_.solve(K_, F_, E);
#endif
}

void CPU_Solver::solve(const SparseMatrix& K, const Vector& F, Vector& E) {
    // CPU sparse solver, e.g., Eigen, CSparse
    std::cout << "[CPU Solver] Solving sparse linear system" << std::endl;
    // E = K^{-1} F
}

#ifdef USE_GPU
void GPU_Solver::solve(const SparseMatrix& K, const Vector& F, Vector& E) {
    // GPU/cuDSS solver code
    std::cout << "[GPU Solver] Solving sparse linear system on GPU" << std::endl;
    // Transfer K/F to GPU, call cuDSS, copy back E
}
#endif


//================= material.h =====================
#pragma once
#include <vector>
#include <Eigen/Dense>

struct Material {
    std::vector<Eigen::Matrix3cd> mu;      // per-element 3x3 complex magnetic tensor
    std::vector<Eigen::Matrix3cd> epsilon; // per-element 3x3 complex permittivity tensor
};


//================= mesh.h =====================
#pragma once
#include <vector>
#include <Eigen/Dense>
struct Mesh {
    std::vector<Eigen::Vector3d> nodes;
    std::vector<std::vector<int>> elements; // tetrahedral connectivity
};

//================= sources.h =====================
#pragma once
#include <Eigen/Dense>
struct Sources {
    Eigen::Vector3cd p;     // Electric dipole
    Eigen::Vector3cd m;     // Magnetic dipole
    Eigen::Vector3d r0;     // Dipole location
};

//================= utils.h =====================
#pragma once
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<std::complex<double>> SparseMatrix;
typedef Eigen::VectorXcd Vector;

//================= main.cpp =====================
#include "fem_solver.h"
#include <iostream>

int main() {
    Mesh mesh;
    Material material;
    Sources sources;

    // Initialize mesh, material, sources (test example)
    std::cout << "Initializing mesh, material, sources..." << std::endl;

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();

    Vector E; // solution
    fem.solve(E);

    std::cout << "Simulation complete." << std::endl;
    return 0;
}
