#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>

struct Edge {
    int n1, n2; // Global node indices, sorted such that n1 < n2
};

struct Mesh {
    std::vector<Eigen::Vector3d> nodes;
    std::vector<std::vector<int>> elements; // tetrahedral connectivity

    // Nédélec edge support
    std::vector<Edge> edges;
    std::vector<std::array<int, 6>> element_edges; // Indices of the 6 edges for each element
    std::vector<std::array<int, 6>> element_edge_orientations; // +1 if local edge matches global edge direction, -1 otherwise

    void generate_edges();
};
