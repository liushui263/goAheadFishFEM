#pragma once
#include <Eigen/Dense>
struct Sources {
    Eigen::Vector3cd p;     // Electric dipole
    Eigen::Vector3cd m;     // Magnetic dipole
    Eigen::Vector3d r0;     // Dipole location
};
