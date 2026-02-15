
#include "fem_solver.h"
#include <iostream>
int main() {
    Mesh mesh; // small uniform mesh
    Material material;
    Sources sources;

    // Simple test: isotropic material, single dipole
    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();

    Vector E;
    fem.solve(E);

    std::cout << "Simple test complete. E size: " << E.size() << std::endl;
    return 0;
}