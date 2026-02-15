#include "fem_solver.h"
#include <iostream>
int main() {
    Mesh mesh; // small mesh
    Material material;
    Sources sources;

    // Initialize anisotropic material tensors per element

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();

    Vector E;
    fem.solve(E);

    std::cout << "Anisotropic test complete." << std::endl;
    return 0;
}