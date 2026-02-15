Geophysical Journal International 187, 63–74
《Three-dimensional adaptive higher order finite element simulation for geo-electromagnetics—a marine CSEM example》


1. 研究背景与目标

本文提出了一种三维自适应高阶向量有限元（FEM）数值模拟框架，用于地球电磁学问题，重点应用于海洋可控源电磁法（marine CSEM）。

核心动机：

海底地形（bathymetry）会显著影响CSEM响应

传统有限差分/体积方法在复杂几何建模上存在局限

需要高精度 + 几何灵活 + 自适应能力的三维模拟工具

作者目标是构建一个：

使用 Nédélec 边元

支持 高阶多项式近似

支持 自适应网格加密

基于 primary/secondary field 分解

适用于复杂海底地形

的三维电磁模拟平台。

2. 数学模型

作者求解频域 Maxwell 方程（时间依赖 $e^{-i\omega t}$），采用secondary electric field formulation。

控制方程（向量Helmholtz型）
curl
⁡
(
𝜇
−
1
curl
⁡
𝐸
𝑠
)
−
𝑖
𝜔
(
𝜎
−
𝑖
𝜔
𝜀
)
𝐸
𝑠
=
curl
⁡
(
[
𝜇
𝑝
−
1
−
𝜇
−
1
]
curl
⁡
𝐸
𝑝
)
−
𝑖
𝜔
(
[
𝜎
𝑝
−
𝜎
]
−
𝑖
𝜔
[
𝜀
𝑝
−
𝜀
]
)
𝐸
𝑝
curl(μ
−1
curlE
s
	​

)−iω(σ−iωε)E
s
	​

=curl([μ
p
−1
	​

−μ
−1
]curlE
p
	​

)−iω([σ
p
	​

−σ]−iω[ε
p
	​

−ε])E
p
	​


边界条件：

𝑛
×
𝐸
𝑠
=
0
n×E
s
	​

=0

函数空间：

𝐸
∈
𝐻
(
curl
⁡
,
Ω
)
E∈H(curl,Ω)
3. 数值方法
3.1 有限元离散

使用 tetrahedral mesh

使用 Nédélec edge elements

多项式阶数 $p = 1,2,3$

近似展开：

𝐸
~
𝑠
=
∑
𝑗
=
1
𝑛
𝑒
𝑗
𝜙
𝑗
E
~
s
	​

=
j=1
∑
n
	​

e
j
	​

ϕ
j
	​


离散线性系统：

𝐴
𝑒
=
𝑓
Ae=f

矩阵形式：

𝐴
𝑖
𝑗
=
∫
Ω
(
curl
⁡
𝜙
𝑖
)
⋅
𝜇
−
1
(
curl
⁡
𝜙
𝑗
)
 
𝑑
3
𝑟
−
𝑖
𝜔
∫
Ω
𝜙
𝑖
⋅
(
𝜎
−
𝑖
𝜔
𝜀
)
𝜙
𝑗
 
𝑑
3
𝑟
A
ij
	​

=∫
Ω
	​

(curlϕ
i
	​

)⋅μ
−1
(curlϕ
j
	​

)d
3
r−iω∫
Ω
	​

ϕ
i
	​

⋅(σ−iωε)ϕ
j
	​

d
3
r
4. 自适应误差估计

磁场两种近似：

方法1（直接）：
𝐻
~
𝑠
=
(
𝑖
𝜔
𝜇
)
−
1
curl
⁡
𝐸
~
𝑠
H
~
s
	​

=(iωμ)
−1
curl
E
~
s
	​

方法2（L² 投影）：
𝐻
^
𝑠
=
𝐿
2
-projection of 
(
𝑖
𝜔
𝜇
)
−
1
curl
⁡
𝐸
~
𝑠
H
^
s
	​

=L
2
-projection of (iωμ)
−1
curl
E
~
s
	​


局部误差指标：

𝜂
𝐾
𝑖
=
∫
𝐾
𝑖
(
𝐻
^
𝑠
−
𝐻
~
𝑠
)
⋅
𝜇
(
𝐻
^
𝑠
−
𝐻
~
𝑠
)
 
𝑑
3
𝑟
η
K
i
	​

	​

=∫
K
i
	​

	​

(
H
^
s
	​

−
H
~
s
	​

)⋅μ(
H
^
s
	​

−
H
~
s
	​

)d
3
r
5. 数值实验结论
(1) 收敛性研究

高阶元显著提高精度

$p=2$ 性价比最佳

自适应 refinement 优于单纯 h-refinement

(2) 海底地形影响

海底起伏引入明显三维效应

破坏对称性

会影响反演解释

(3) Disc模型

模拟油气储层

对比有限体积法 FDM3D

FEM 在源附近精度更优

但内存消耗更高

6. 优点

几何灵活性强

高阶多项式显著提高精度

自适应 refinement 高效

primary/secondary 分解提升数值稳定性

7. 局限性

直接求解器内存占用大

复杂模型时 $p=3$ 计算量过高

计算域大小对 air layer 敏感

8. 总体评价

这是一篇方法学扎实、数值验证充分的高质量工作。其贡献主要在：

将高阶 edge FEM + AMR 系统性引入 marine CSEM

证明海底地形必须被精确建模

为后续高精度三维EM模拟奠定基础



1. Objective and Motivation

This paper presents a 3-D vector finite element framework for frequency-domain geo-electromagnetic simulations, with a specific focus on marine Controlled-Source Electromagnetics (CSEM).

The work addresses three core challenges:

Accurate modeling of complex seafloor bathymetry

High numerical accuracy in 3-D electromagnetic diffusion problems

Efficient error control via adaptive mesh refinement (AMR)

The authors combine:

Nédélec edge elements

Higher-order polynomial approximation

Adaptive tetrahedral meshes

Primary/secondary field decomposition

into a unified computational framework.

2. Mathematical Formulation

The study solves the time-harmonic Maxwell system assuming an $e^{-i\omega t}$ dependence.

Secondary Electric Field Equation
curl
⁡
(
𝜇
−
1
curl
⁡
𝐸
𝑠
)
−
𝑖
𝜔
(
𝜎
−
𝑖
𝜔
𝜀
)
𝐸
𝑠
=
curl
⁡
(
[
𝜇
𝑝
−
1
−
𝜇
−
1
]
curl
⁡
𝐸
𝑝
)
−
𝑖
𝜔
(
[
𝜎
𝑝
−
𝜎
]
−
𝑖
𝜔
[
𝜀
𝑝
−
𝜀
]
)
𝐸
𝑝
curl(μ
−1
curlE
s
	​

)−iω(σ−iωε)E
s
	​

=curl([μ
p
−1
	​

−μ
−1
]curlE
p
	​

)−iω([σ
p
	​

−σ]−iω[ε
p
	​

−ε])E
p
	​


Boundary condition:

𝑛
×
𝐸
𝑠
=
0
n×E
s
	​

=0

Function space:

𝐸
∈
𝐻
(
c
u
r
l
,
Ω
)
E∈H(curl,Ω)

The primary/secondary decomposition restricts numerical approximation to the scattered field, improving accuracy.

3. Finite Element Discretization

The approximation is expressed as:

𝐸
~
𝑠
=
∑
𝑗
=
1
𝑛
𝑒
𝑗
𝜙
𝑗
E
~
s
	​

=
j=1
∑
n
	​

e
j
	​

ϕ
j
	​


Resulting in a linear system:

𝐴
𝑒
=
𝑓
Ae=f

with stiffness matrix:

𝐴
𝑖
𝑗
=
∫
Ω
(
c
u
r
l
 
𝜙
𝑖
)
⋅
𝜇
−
1
(
c
u
r
l
 
𝜙
𝑗
)
 
𝑑
3
𝑟
−
𝑖
𝜔
∫
Ω
𝜙
𝑖
⋅
(
𝜎
−
𝑖
𝜔
𝜀
)
𝜙
𝑗
 
𝑑
3
𝑟
A
ij
	​

=∫
Ω
	​

(curlϕ
i
	​

)⋅μ
−1
(curlϕ
j
	​

)d
3
r−iω∫
Ω
	​

ϕ
i
	​

⋅(σ−iωε)ϕ
j
	​

d
3
r

Properties:

Complex symmetric

Sparse

Indefinite

4. Adaptive Error Estimation

Two magnetic field approximations are constructed:

Direct:

𝐻
~
𝑠
=
(
𝑖
𝜔
𝜇
)
−
1
c
u
r
l
𝐸
~
𝑠
H
~
s
	​

=(iωμ)
−1
curl
E
~
s
	​


L² Projection:

𝐻
^
𝑠
=
projection of 
(
𝑖
𝜔
𝜇
)
−
1
c
u
r
l
𝐸
~
𝑠
H
^
s
	​

=projection of (iωμ)
−1
curl
E
~
s
	​


Local refinement indicator:

𝜂
𝐾
=
∫
𝐾
(
𝐻
^
𝑠
−
𝐻
~
𝑠
)
⋅
𝜇
(
𝐻
^
𝑠
−
𝐻
~
𝑠
)
 
𝑑
3
𝑟
η
K
	​

=∫
K
	​

(
H
^
s
	​

−
H
~
s
	​

)⋅μ(
H
^
s
	​

−
H
~
s
	​

)d
3
r
5. Numerical Experiments
(1) Convergence Study

Higher-order elements ($p=2,3$) significantly reduce error

$p=2$ provides best cost–accuracy tradeoff

AMR outperforms uniform refinement

(2) Bathymetry Model

Seafloor topography breaks symmetry

Generates strong 3-D effects

Must be modeled to avoid misinterpretation

(3) Canonical Disc Model

Hydrocarbon reservoir scenario

Compared against finite-volume code FDM3D

FEM more accurate near source

Higher memory consumption due to direct solver

6. Strengths

Geometric flexibility (unstructured tetrahedra)

Rigorous curl-conforming formulation

Effective AMR strategy

Demonstrated benefit of higher-order basis

7. Limitations

High memory demand (direct solver)

Conditioning issues when air layer included

$p=3$ often computationally prohibitive

8. Overall Assessment

This work convincingly demonstrates that:

Adaptive higher-order Nédélec finite elements provide a robust and accurate framework for 3-D marine CSEM simulations, particularly when complex bathymetry must be incorporated.

The paper represents a solid contribution to computational geo-electromagnetics.


