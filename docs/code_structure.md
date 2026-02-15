FEM_Maxwell_Solver/
├── src/
│   ├── fem_solver.h           # FEM solver header
│   ├── fem_solver.cpp         # FEM solver implementation
│   ├── material.h             # 各向异性材料管理
│   ├── mesh.h                 # 网格生成与管理
│   ├── sources.h              # 电偶极子和磁偶极子源
│   ├── main.cpp               # 主程序入口
│   └── utils.h                # 工具函数 (线性代数、GPU封装)
├── test/
│   ├── test_simple.cpp        # 简单均匀介质测试例
│   ├── test_anisotropic.cpp   # 各向异性测试例
│   └── benchmark.cpp          # 大规模性能测试
├── docs/
│   ├── FEM3D_for_CSEM_Review.md
│   ├── FEM3D_for_CSEM_rewrite.md
│   ├── images/FEM3D_01.png
│   ├── images/FEM3D_02.png
│   └── images/FEM3D_03.png
├── include/
│   └── third_party/           # cuDSS 或 PETSc头文件
├── CMakeLists.txt             # 构建脚本
└── README.md                  # 使用说明和测试方法

# FEM_Maxwell_Solver Test and Documentation

## 项目结构 (包括测试)

```
FEM_Maxwell_Solver/
├── src/
│   └── ...
├── test/
│   ├── test_simple.cpp
│   ├── test_anisotropic.cpp
│   └── benchmark.cpp
├── images/
│   └── FEM3D_*.png
├── CMakeLists.txt
└── README.md
```

---

## 1. test_simple.cpp

```cpp
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
```

## 2. test_anisotropic.cpp

```cpp
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
```

## 3. benchmark.cpp

```cpp
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
```
