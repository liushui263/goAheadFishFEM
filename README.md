# FEM_Maxwell_Solver

## 功能
- 3D FEM Maxwell Solver
- 支持任意各向异性材料
- 电偶极子 & 磁偶极子源
- CPU稀疏求解和GPU/cuDSS切换

## 编译
```bash
mkdir build && cd build
cmake .. -DUSE_GPU=ON  # 或 OFF 使用CPU
make
```

## 运行示例
```bash
./test_simple
./test_anisotropic
./benchmark
```

## 文件说明
- src/ 源码
- test/ 测试和benchmark
- docs/ 文档
- docs/images/ FEM示意图
- CMakeLists.txt 构建配置


## build options 
- CMAKE_BUILD_TYPE=Release by default unless overridden.
- CMAKE_INSTALL_PREFIX defaults to $HOME/FEM3D_CPU_build_install or $HOME/FEM3D_GPU_build_install depending on USE_GPU, but can be overridden.
- GPU headers/libraries (cuDSS) are installed only if USE_GPU=ON.
- CPU-only build avoids all NVPL/CUDA hints.
- Executables go to bin/, library to lib/, headers to include/.

## Build & Install CPU build:
```bash
cmake -S . -B build_cpu -DUSE_GPU=OFF
cmake --build build_cpu -j$(nproc)
cmake --install build_cpu
```
- Binaries: $HOME/FEM3D_CPU_build_install/bin/
- Headers: $HOME/FEM3D_CPU_build_install/include/
- Library: $HOME/FEM3D_CPU_build_install/lib/


##  Build & Install GPU build:
```bash
cmake -S . -B build_gpu -DUSE_GPU=ON
cmake --build build_gpu -j$(nproc)
cmake --install build_gpu
```
- Binaries: $HOME/FEM3D_GPU_build_install/bin/
- Library + cuDSS headers installed only for GPU.