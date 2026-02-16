#pragma once
#include "mesh.h"
#include "material.h"
#include "sources.h"
#include "utils.h"
#include <vector>

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
    void assemble_maxwell(int N, double freq, double sigma);
    void assemble_interference_pattern(int N);
    void assemble_vector_maxwell(int nx, int ny, int nz, double dx, double freq, double sigma);
    void assemble_dipole(int N, const std::string& type);
    void save_as_json(const std::string& filename, const Vector& E);
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
