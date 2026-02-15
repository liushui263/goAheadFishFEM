#include "mesh.h"
#include <map>
#include <algorithm>
#include <iostream>

void Mesh::generate_edges() {
    edges.clear();
    element_edges.resize(elements.size());
    element_edge_orientations.resize(elements.size());

    // Map to keep track of unique edges: (min_node, max_node) -> global_edge_index
    // Using a map ensures we identify shared edges between elements
    std::map<std::pair<int, int>, int> edge_map;

    // Local edge definitions for a tetrahedron with local nodes 0, 1, 2, 3
    // Standard lexicographical numbering:
    // Edge 0: (0,1), Edge 1: (0,2), Edge 2: (0,3)
    // Edge 3: (1,2), Edge 4: (1,3), Edge 5: (2,3)
    const int local_edges[6][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };

    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& elem = elements[i];
        if (elem.size() != 4) {
            std::cerr << "Warning: Element " << i << " is not a tetrahedron (size " << elem.size() << ")." << std::endl;
            continue;
        }

        for (int j = 0; j < 6; ++j) {
            int n1 = elem[local_edges[j][0]];
            int n2 = elem[local_edges[j][1]];

            // Determine global edge key (sorted nodes to ensure uniqueness)
            int u = std::min(n1, n2);
            int v = std::max(n1, n2);
            std::pair<int, int> key = {u, v};

            int global_edge_idx = -1;
            if (edge_map.find(key) == edge_map.end()) {
                // New edge found
                global_edge_idx = static_cast<int>(edges.size());
                edges.push_back({u, v});
                edge_map[key] = global_edge_idx;
            } else {
                // Existing edge found
                global_edge_idx = edge_map[key];
            }

            element_edges[i][j] = global_edge_idx;
            
            // Orientation: +1 if local direction (n1->n2) matches global (u->v), else -1
            // Since u is min and v is max, u->v implies smaller->larger index.
            element_edge_orientations[i][j] = (n1 < n2) ? 1 : -1;
        }
    }
    std::cout << "Mesh edges generated. Total unique edges: " << edges.size() << std::endl;
}