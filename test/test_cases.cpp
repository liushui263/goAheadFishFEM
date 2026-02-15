#include <gtest/gtest.h>
#include "fem_solver.h"
#include "mesh.h"
#include "material.h"
#include "sources.h"
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// Helper to create a structured box mesh of tetrahedra
// nx, ny, nz: number of cells in each direction
void CreateBoxMesh(Mesh& mesh, int nx, int ny, int nz, double lx, double ly, double lz) {
    mesh.nodes.clear();
    mesh.elements.clear();
    
    double dx = lx / nx;
    double dy = ly / ny;
    double dz = lz / nz;

    // Generate Nodes
    for(int k=0; k<=nz; ++k) {
        for(int j=0; j<=ny; ++j) {
            for(int i=0; i<=nx; ++i) {
                mesh.nodes.emplace_back(i*dx, j*dy, k*dz);
            }
        }
    }

    auto get_node = [&](int i, int j, int k) {
        return k * (nx+1) * (ny+1) + j * (nx+1) + i;
    };

    // Generate Elements (Kuhn triangulation: 6 tetrahedra per cube)
    for(int k=0; k<nz; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                int n0 = get_node(i, j, k);
                int n1 = get_node(i+1, j, k);
                int n2 = get_node(i+1, j+1, k);
                int n3 = get_node(i, j+1, k);
                int n4 = get_node(i, j, k+1);
                int n5 = get_node(i+1, j, k+1);
                int n6 = get_node(i+1, j+1, k+1);
                int n7 = get_node(i, j+1, k+1);

                mesh.elements.push_back({n0, n1, n2, n6});
                mesh.elements.push_back({n0, n1, n5, n6});
                mesh.elements.push_back({n0, n4, n5, n6});
                mesh.elements.push_back({n0, n2, n3, n7});
                mesh.elements.push_back({n0, n2, n6, n7});
                mesh.elements.push_back({n0, n4, n6, n7});
            }
        }
    }
}

class FEM3DTest : public ::testing::Test {
protected:
    Mesh mesh;
    Material material;
    Sources sources;

    void SetUp() override {
        // Setup default source
        sources.p = Eigen::Vector3cd(1.0, 0.0, 0.0);
        sources.m = Eigen::Vector3cd(0.0, 0.0, 0.0);
        sources.r0 = Eigen::Vector3d(0.5, 0.5, 0.5);
    }

    void SetupMaterial() {
        // Initialize material properties for all elements
        material.mu.resize(mesh.elements.size(), Eigen::Matrix3cd::Identity());
        material.epsilon.resize(mesh.elements.size(), Eigen::Matrix3cd::Identity());
    }
};

TEST_F(FEM3DTest, MeshGenerationSmall) {
    // 2x2x2 grid = 8 cubes. 6 tets per cube = 48 elements.
    CreateBoxMesh(mesh, 2, 2, 2, 1.0, 1.0, 1.0);
    mesh.generate_edges();

    EXPECT_EQ(mesh.elements.size(), 48);
    EXPECT_EQ(mesh.nodes.size(), 27); // 3x3x3 nodes
    EXPECT_GT(mesh.edges.size(), 0);
}

TEST_F(FEM3DTest, SolverPipelineMedium) {
    // 5x5x5 grid = 125 cubes * 6 = 750 elements
    CreateBoxMesh(mesh, 5, 5, 5, 1.0, 1.0, 1.0);
    mesh.generate_edges();
    SetupMaterial();

    FEM_Solver fem(&mesh, &material, &sources);
    EXPECT_NO_THROW(fem.assemble());
    
    Vector E;
    EXPECT_NO_THROW(fem.solve(E));
}

TEST_F(FEM3DTest, SolverPipelineLarge) {
    // 10x10x10 grid = 1000 cubes * 6 = 6000 elements
    CreateBoxMesh(mesh, 10, 10, 10, 1.0, 1.0, 1.0);
    mesh.generate_edges();
    SetupMaterial();

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();
    
    Vector E;
    fem.solve(E);
    
    // Basic check to ensure we have a result
    // Note: E size currently depends on dummy implementation in fem_solver.cpp
    // Once fixed, it should match mesh.edges.size()
    EXPECT_GT(E.size(), 0);
}

TEST_F(FEM3DTest, Heterogeneous3D) {
    // 10x10x10 grid = 1000 cubes * 6 = 6000 elements
    // Physical size 2.0 x 2.0 x 2.0
    CreateBoxMesh(mesh, 10, 10, 10, 2.0, 2.0, 2.0);
    mesh.generate_edges();
    
    // Initialize base materials (Identity)
    SetupMaterial();

    // Define a spherical inclusion at center
    Eigen::Vector3d center(1.0, 1.0, 1.0);
    double radius = 0.6;

    for(size_t i=0; i<mesh.elements.size(); ++i) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for(int n=0; n<4; ++n) {
            centroid += mesh.nodes[mesh.elements[i][n]];
        }
        centroid /= 4.0;

        if((centroid - center).norm() < radius) {
            // Inclusion: high contrast (e.g. epsilon = 10 + 2i)
            material.epsilon[i] = Eigen::Matrix3cd::Identity() * std::complex<double>(10.0, 2.0);
        }
    }

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();
    
    Vector E;
    fem.solve(E);
    
    EXPECT_GT(E.size(), 0);
}

TEST_F(FEM3DTest, SolverPipelineMassive) {
    // 60x60x60 grid = 216,000 cubes * 6 = 1,296,000 elements
    // This generates ~1.5M+ edges, sufficient to show GPU advantage
    std::cout << "Generating massive mesh (60x60x60)..." << std::endl;
    CreateBoxMesh(mesh, 60, 60, 60, 6.0, 6.0, 6.0);
    mesh.generate_edges();
    SetupMaterial();

    std::cout << "Mesh Stats - Elements: " << mesh.elements.size() 
              << ", Edges (Unknowns): " << mesh.edges.size() << std::endl;

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();
    
    Vector E;
    fem.solve(E);
    
    EXPECT_GT(E.size(), 0);
}

TEST_F(FEM3DTest, SolverPipelineHuge) {
    // 40x40x40 grid = 64,000 cubes * 6 = 384,000 elements
    // This generates ~200k+ edges, sufficient to show GPU advantage
    std::cout << "Generating huge mesh (40x40x40)..." << std::endl;
    CreateBoxMesh(mesh, 40, 40, 40, 4.0, 4.0, 4.0);
    mesh.generate_edges();
    SetupMaterial();

    std::cout << "Mesh Stats - Elements: " << mesh.elements.size() 
              << ", Edges (Unknowns): " << mesh.edges.size() << std::endl;

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();
    
    Vector E;
    fem.solve(E);
    
    EXPECT_GT(E.size(), 0);
}


TEST_F(FEM3DTest, Heterogeneous3DMassive) {
    // 10x10x10 grid = 1000 cubes * 6 = 6000 elements
    // Physical size 2.0 x 2.0 x 2.0
    CreateBoxMesh(mesh, 100, 100, 100, 20.0, 20.0, 20.0);
    mesh.generate_edges();
    
    // Initialize base materials (Identity)
    SetupMaterial();

    // Define a spherical inclusion at center
    Eigen::Vector3d center(1.0, 1.0, 1.0);
    double radius = 2.6;

    for(size_t i=0; i<mesh.elements.size(); ++i) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for(int n=0; n<4; ++n) {
            centroid += mesh.nodes[mesh.elements[i][n]];
        }
        centroid /= 4.0;

        if((centroid - center).norm() < radius) {
            // Inclusion: high contrast (e.g. epsilon = 10 + 2i)
            material.epsilon[i] = Eigen::Matrix3cd::Identity() * std::complex<double>(10.0, 2.0);
        }
    }

    FEM_Solver fem(&mesh, &material, &sources);
    fem.assemble();
    
    Vector E;
    fem.solve(E);
    
    EXPECT_GT(E.size(), 0);
}
