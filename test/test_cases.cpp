#include <gtest/gtest.h>
#include "fem_solver.h"
#include <Eigen/Core>
#include <complex>
#include <iostream>
#include <chrono>

// Helper to print separator for readability in logs
void PrintSeparator(const std::string& name) {
    std::cout << "\n=============================================\n";
    std::cout << "   Running Test: " << name << "\n";
    std::cout << "=============================================\n";
}

// 1. Basic Synthetic Test (Replaces test_simple)
// Uses the default assemble() which creates a diagonally dominant matrix
TEST(FEM3DTest, SyntheticAssembly_Small) {
    PrintSeparator("SyntheticAssembly_Small");
    
    // Initialize with null pointers (uses default internal logic)
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    // Default assemble() creates a 100x100 diagonal dominant system if no mesh is provided
    solver.assemble();
    
    Eigen::VectorXcd E;
    solver.solve(E);
    
    // Validation
    EXPECT_EQ(E.size(), 100);
    // Ensure we have a non-zero solution
    EXPECT_GT(E.norm(), 1e-9);
    // Ensure solution is finite (no NaNs)
    EXPECT_TRUE(E.allFinite());
    
    std::cout << "Synthetic Solution Norm: " << E.norm() << std::endl;
}

// 2. Realistic Physics Test (The new requirement)
// Low Frequency (2kHz), High Conductivity (1 S/m) -> Ill-conditioned matrix
TEST(FEM3DTest, RealisticPhysics_LWD_Scenario) {
    PrintSeparator("RealisticPhysics_LWD_Scenario");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    int N = 1000000;        // Moderate size for unit testing
    double freq = 2000.0; // 2 kHz
    double sigma = 1.0;   // 1 S/m
    
    solver.assemble_maxwell(N, freq, sigma);
    
    Eigen::VectorXcd E;
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Solve Time (N=" << N << "): " << elapsed.count() << " ms" << std::endl;
    
    EXPECT_EQ(E.size(), N);
    EXPECT_TRUE(E.allFinite());
    std::cout << "Physics Solution Norm: " << E.norm() << std::endl;

    // Export results to JSON for Python comparison
    solver.save_as_json("results_physics", E);
}

// 3. Benchmark / Stress Test (Replaces benchmark)
// Larger system to stress test the solver (CPU vs GPU)
TEST(FEM3DTest, Benchmark_LargeSystem) {
    PrintSeparator("Benchmark_LargeSystem");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    // 50,000 unknowns is large enough to measure time, but small enough for CI
    int N = 500000; 
    double freq = 2000.0;
    double sigma = 1.0;
    
    std::cout << "Assembling system with N=" << N << "..." << std::endl;
    solver.assemble_maxwell(N, freq, sigma);
    
    Eigen::VectorXcd E;
    
    std::cout << "Solving..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Benchmark Solve Time: " << elapsed.count() << " ms" << std::endl;
    
    // Export results to JSON for Python comparison
    solver.save_as_json("results_LargeSystem", E);

    EXPECT_EQ(E.size(), N);
    EXPECT_TRUE(E.allFinite());
}

// 4. Beautiful Pattern Test
// Generates a 3D interference pattern from multiple sources
TEST(FEM3DTest, BeautifulPattern) {
    PrintSeparator("BeautifulPattern");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    // ~260k unknowns (64^3 grid) provides good resolution for visualization
    int N = 262144; 
    
    solver.assemble_interference_pattern(N);
    
    Eigen::VectorXcd E;
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Pattern Solve Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    
    solver.save_as_json("results_pattern", E);
}

// 5. Electric Dipole Test
// Z-directed electric dipole in the center
TEST(FEM3DTest, ElectricDipoleZ) {
    PrintSeparator("ElectricDipoleZ");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    int N = 262144; // 64^3 grid
    
    solver.assemble_dipole(N, "electric");
    
    Eigen::VectorXcd E;
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Electric Dipole Solve Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    
    solver.save_as_json("results_electric_dipole", E);
}

// 6. Magnetic Dipole Test
// Z-directed magnetic dipole in the center (simulated loop)
TEST(FEM3DTest, MagneticDipoleZ) {
    PrintSeparator("MagneticDipoleZ");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    int N = 262144; // 64^3 grid
    
    solver.assemble_dipole(N, "magnetic");
    
    Eigen::VectorXcd E;
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Magnetic Dipole Solve Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    
    solver.save_as_json("results_magnetic_dipole", E);
}

// 7. True 3D Vector Maxwell Test
// Solves the full vector wave equation on a Yee grid
TEST(FEM3DTest, TrueVectorMaxwell) {
    PrintSeparator("TrueVectorMaxwell");
    
    FEM_Solver solver(nullptr, nullptr, nullptr);
    
    // Grid parameters
    int nx = 60, ny = 60, nz = 60;
    double dx = 0.25;
    double freq = 2e6; 
    double sigma = 0.1;
    
    solver.assemble_vector_maxwell(nx, ny, nz, dx, freq, sigma);
    
    Eigen::VectorXcd E;
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(E);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Vector Maxwell Solve Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    solver.save_as_json("results_vector_maxwell", E);
}