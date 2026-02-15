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
