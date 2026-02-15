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