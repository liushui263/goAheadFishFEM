#pragma once
#include <vector>
#include <Eigen/Dense>

struct Material {
    std::vector<Eigen::Matrix3cd> mu;      // per-element 3x3 complex magnetic tensor
    std::vector<Eigen::Matrix3cd> epsilon; // per-element 3x3 complex permittivity tensor
};
