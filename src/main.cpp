#include "fem_solver.h"
#include <iostream>

int main() {
    Mesh mesh;
    Material material;
    Sources sources;

    // Initialize mesh, material, sources (test example)
    std::cout << "Initializing mesh, material, sources..." << std::endl;

    // Generate edges for Nédélec elements
    mesh.generate_edges();

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();

    Vector E; // solution
    fem.solve(E);

    std::cout << "Simulation complete." << std::endl;
    return 0;
}
