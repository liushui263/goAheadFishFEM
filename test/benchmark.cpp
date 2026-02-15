#include "fem_solver.h"
#include <chrono>
#include <iostream>
int main() {
    Mesh mesh; // large mesh
    Material material;
    Sources sources;

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();

    Vector E;
    auto start = std::chrono::high_resolution_clock::now();
    fem.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Benchmark solve time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    return 0;
}