# GeoSlope

2D 边坡稳定数值分析工具，基于 Python 实现 FEM（有限元强度折减法）+ LEM（极限平衡法）交叉验证。

## 功能

- **FEM 强度折减法 (SRF)**：弹塑性 Mohr-Coulomb 模型，自动搜索安全系数
- **LEM 极限平衡法**：Fellenius、Bishop 简化、Janbu 简化、Morgenstern-Price
- **交叉验证**：FEM 与 LEM 结果对比，多方法一致性检验
- **DXF 导入**：支持从 AutoCAD DXF 导入几何模型
- **可视化**：位移云图、塑性区分布、滑弧展示

## 安装

```bash
git clone git@github.com:DaiYH-creator/GeoSlope.git
cd GeoSlope
pip install -r requirements.txt
```

## 快速开始

```python
from core.geometry import SlopeGeometry
from core.material import MohrCoulomb
from core.mesh import TriMesh
from core.fem.srf import srf_fos

# 定义边坡
geo = SlopeGeometry(height=10.0, angle=45.0, crest_width=5.0, base_depth=5.0)
mat = MohrCoulomb(gamma=20.0, c=15.0, phi=25.0, E=10000.0, nu=0.35)
mesh = TriMesh(geo, mesh_size=1.0)

# FEM 强度折减
fos, result = srf_fos(mesh, mat, fos_min=0.5, fos_max=3.0)
print(f"FEM FoS = {fos:.3f}")
```

## 验证结果

测试工况：H=10m, β=45°, c=15kPa, φ=25°, γ=20kN/m³

| 方法 | 安全系数 |
|------|----------|
| Fellenius | 1.136 |
| Bishop 简化 | 1.219 |
| Janbu 简化 | 1.253 |
| Morgenstern-Price | 1.180 |
| **FEM SRF** | **1.256** |

FEM/LEM 偏差 < 5%，METHODS AGREE.

## 项目结构

```
GeoSlope/
├── core/
│   ├── geometry.py          # 边坡几何定义
│   ├── material.py          # Mohr-Coulomb 材料模型
│   ├── mesh.py              # 三角形网格生成
│   ├── fem/                 # 有限元模块
│   │   ├── element.py       # 三角形单元刚度矩阵
│   │   ├── assembly.py      # 总体刚度组装与边界条件
│   │   ├── solver.py        # 线性方程组求解
│   │   ├── eplastic.py      # 弹塑性迭代求解
│   │   └── srf.py           # 强度折减 FoS 搜索
│   ├── lem/                 # 极限平衡模块
│   │   ├── slip_surface.py  # 圆弧滑面生成与条分
│   │   ├── fellenius.py     # Fellenius 法
│   │   ├── bishop.py        # Bishop 简化法
│   │   ├── janbu.py         # Janbu 简化法
│   │   ├── morgenstern_price.py  # Morgenstern-Price 法
│   │   └── cross_validate.py     # FEM/LEM 交叉验证
│   └── io/
│       └── dxf_reader.py    # DXF 几何导入
├── visualization/
│   └── plot_2d.py           # 2D 可视化
├── tests/
│   ├── test_srf.py
│   └── test_cross_validate.py
├── examples/
│   ├── demo_vis.py
│   └── demo_dxf.py
└── requirements.txt
```

## 技术要点

- **平面应变假定**：出平面应力 σz = ν(σx+σy)（岩土力学符号惯例）
- **Mohr-Coulomb 屈服准则**：以 σ1≥σ2≥σ3 排序判定剪切破坏
- **渐进损伤**：弹塑性迭代采用渐进刚度折减（rate=0.5），避免数值振荡
- **位移失效判据**：最大位移超过坡高 2% 判定失稳
- **增量扫描 + 二分法**：SRF 搜索先用粗扫确定上下界，再二分精化

## 许可

MIT License
