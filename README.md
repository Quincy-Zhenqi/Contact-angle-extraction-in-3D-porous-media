# Contact angle extraction with geometrical and morphological constraints in 3D porous media

**V4.2 · English / 中文**
This repository contains the Python source accompanying the revised manuscript of the same title. It measures local contact angles along the solid–liquid–gas contact line in segmented CT data. Comparing the geometrically computable and quality-controlled distributions shows how the choice of measurement sites affects the inferred wettability.

本仓库提供同名修订稿及补充材料（SI）中的 V4.2 Python 源码，用于测量已分割 CT 中固–液–气三相接触线上的局部接触角。比较几何可计算集合与质量筛选子集的角度分布，可分析测点选择对润湿性表征的影响。

## Measurement / 测量方法

1. Read a 3D integer-labelled TIFF stack in **(z, y, x)** order. Solid = 1, liquid = 2, gas = 3, exterior background = 4 (optional). All three scientific phases must occur. Input labels are read-only. The study uses **9.14 µm isotropic voxels**. The implementation requires isotropic spacing.

   读取 **(z, y, x)** 顺序的三维整数标签 TIFF。固体、液体、气体分别为 1、2、3，均须出现，外部背景可标为 4。程序只读原始标签。研究数据的各向同性体素边长为 **9.14 µm**，实现要求三轴间距相同。

2. Represent both interfaces by liquid voxel centres. A raw candidate touches both solid and gas through voxel faces. In the 9³ fitting cube, select the candidate-connected sheet of each interface separately (18-connectivity). The surrounding 11³ geometry envelope must lie inside the array and contain only scientific labels. Fit each plane by centred SVD using all selected points. At least three non-collinear points and a resolvable second singular value are required.

   两个界面均用液相体素中心表示，同时与固、气两相面邻接的液相体素构成原始候选点。在 9³ 拟合立方体内分别取与候选点 18 邻接连通的界面片。外围 11³ 几何支撑区须完整且仅含三相标签。对全部入选点做中心化 SVD，至少需要三个非共线点且第二奇异值须可分辨。

3. Use local 6-connected phase centroids to orient unit normals **nLG** (liquid → gas) and **nSL** (solid → liquid). Where the centre lies outside a phase, use the largest face-adjacent local component. **G** contains candidates with a finite **θ = arccos(nLG · nSL)** in degrees (0–180°), including obtuse angles. Normal orientation requires a defined, nonzero projection. **Q** takes rows of G that pass all 23 checks below. The stored angle is copied from G.

   用局部 6 邻接相组分的质心将法向量分别定向为液→气、固→液。中心位于该相之外时，取与中心面邻接的最大局部组分。**G** 包含定向投影有定义且非零、能计算有限夹角的候选点，角度范围为 0–180°，保留钝角。**Q** 从 G 中选出以下 23 项全部通过的点，每个点沿用 G 中的角度。

## Acceptance criteria / 接受条件

Each raw candidate receives an independent `pass`, `fail` or `not_evaluated` for each criterion. Unavailable measurements can coexist with failures on the same candidate. Each paired `wg` / `sw` entry below counts twice (liquid–gas / solid–liquid).
每个候选点的每项条件均独立记录通过、失败或不可评价。同一点可同时存在失败项和不可评价项。下表中每对 `wg` / `sw` 分别对应液–气和固–液界面，计作两项。

| Criterion / 条件 | V4.2 default / 默认判据 |
|---|---|
| `numerical_geometry` | Finite oriented angle (membership in G) / 有限定向夹角，定义 G |
| `topology` | Six face neighbours include all three phases, excluding the centre / 中心之外的六个面邻居含三相 |
| `legacy_margin` | Distance to each array face ≥15 voxels / 距每一数组边界至少 15 体素 |
| `sphere` | Complete radius-15 composition sphere containing labels 1–3 only / 半径 15 的组分球完整且仅含三相 |
| `component_size` | Liquid and largest face-adjacent gas components each ≥50 voxels, global 6-connectivity / 液相及面邻接最大气相组分均 ≥50 体素，全局 6 邻接 |
| `candidate_density` | ≤3 face-adjacent raw candidates / 面邻接原始候选点数 ≤3 |
| `dense_guard` | Dense = ≥4 raw-candidate face neighbours. Core = dense with ≥4 dense face neighbours. Chebyshev distance to core >2 voxels / 稠密点至少有 4 个候选面邻居，核心点还须有至少 4 个稠密面邻居，距核心的切比雪夫距离须 >2 |
| `solid_count`, `liquid_count`, `gas_count` | Each phase ≥20 voxels in the complete 11³ count cube / 完整 11³ 计数立方体中每相 ≥20 体素 |
| `solid_fraction` | 0.30 ≤ solid fraction ≤0.70 in the radius-15 sphere / 组分球中固相比例含上下限 |
| `liquid_fraction`, `gas_fraction` | Each finite and ≥0, upper bound disabled / 各自有限且 ≥0，未启用上限 |
| `wg_interface_size`, `sw_interface_size` | Each global 18-connected interface ≥15 voxels / 两个全局 18 邻接界面组分各 ≥15 体素 |
| `wg_plane_points`, `sw_plane_points` | Each local sheet ≥15 fitting points in the 9³ cube / 9³ 窗内两个局部界面片各 ≥15 个拟合点 |
| `wg_plane_rms`, `sw_plane_rms` | Orthogonal plane RMS ≤1.0 reference voxel / 正交平面残差均方根 ≤1.0 个参考体素 |
| `wg_singular_value_ratio`, `sw_singular_value_ratio` | s3/s2 ≤0.30 for each fit / 两个拟合的奇异值比均 ≤0.30 |
| `wg_orientation_alignment`, `sw_orientation_alignment` | Absolute projection of each unit normal onto its unit centroid direction ≥0.05 / 单位法向量在对应质心连线单位方向上的投影绝对值 ≥0.05 |

## Run one image / 运行单个图像

The pinned dependencies were tested with 64-bit Python 3.8.2. From the repository directory, use a virtual environment and a new output folder. Replace the example TIFF path with your segmented image.
固定依赖已在 64 位 Python 3.8.2 上验证。在仓库目录的虚拟环境中执行下列命令，将示例输入替换为实际 TIFF，结果目录须为新目录。

```sh
python -m pip install -r requirements.txt
python -c "from contact_angle_extraction import run_image; run_image('data/sample.tif', 'results/run01')"
python -m unittest discover -s tests -v
```

The unit tests generate two perpendicular interfaces and check a true angle of 90° (43 raw candidates, 33 in G, 13 in Q). To compare the independently extracted SI source and the repository on this model, run the command below with your copy of the SI. Add `--tiff data/sample.tif` to compare your own stack.
测试现场构建两个垂直界面，验证 90° 已知角度，以及 43 个候选、33 个 G 点和 13 个 Q 点。下面的命令从本地 SI 独立提取源码并比较该模型的逐点结果。附加 `--tiff data/sample.tif` 可比较自己的图像。

```sh
python tests/compare_si.py "Supplementary Material_2026.11.20.docx"
```

`example_config.json` records every SI default, including `max_plane_points: null` (use all local points). `run_image` always uses `Parameters()`. For a sensitivity analysis, load an edited JSON into `Parameters` and call `calculate` directly. Save your configuration, input checksum, Python/package versions and Git commit alongside the outputs.
`example_config.json` 列出 SI 全部默认值，`max_plane_points: null` 表示使用全部局部拟合点。`run_image` 固定使用默认参数。参数敏感性分析可按下例调用，并随结果保存实际配置、输入校验值、Python 及依赖版本和 Git 提交号。

```python
import json
import tifffile
from contact_angle_extraction import Parameters, calculate

with open("example_config.json", encoding="utf-8") as f:
    cfg = Parameters(**json.load(f))
features, accepted, states, summary = calculate(tifffile.imread("data/sample.tif"), cfg)
```

## Outputs / 输出

| File | Contents / 内容 |
|---|---|
| `candidate_features.csv.gz` | Every raw candidate, stable within-volume ID, original zero-based z/y/x voxel indices, phase counts/fractions, plane and orientation diagnostics, angle, Q flag and reasons / 全部原始候选及其卷内 ID、从零开始的原图坐标、相计数及比例、拟合与定向诊断、角度、Q 标记及原因 |
| `criterion_states.csv.gz` | 23 states per raw candidate in original feature-row order / 按特征表原始行序保存各候选点的 23 项状态 |
| `G.csv.gz`, `Q.csv.gz` | Feature rows in G and Q with the same coordinates and angles / 分别收录 G、Q 对应的特征行、坐标与角度 |
| `summary.json` | G and Q counts, fractions of raw candidates, mean, median, population SD (ddof=0), quartiles, IQR, P05/P95 and fractions below 10° or above 170° / 两集合的数量、占原始候选比例、均值、中位数、总体标准差、四分位数及四分位距、P05/P95 和两端比例 |

Normal components and centroid fields use **x, y, z** suffixes. Centroids and plane RMS use reference voxels, angles are in degrees, and phase fractions are dimensionless. For the SI state export, attach keys from the original, unsorted feature file **from the same run** before filtering or sorting. For Table 2, compute bias-corrected Fisher–Pearson skewness on Q as below. It is undefined for constant data or fewer than three points.
法向量分量与质心字段后缀为 **x、y、z**，质心坐标与平面 RMS 使用参考体素，角度单位为度，相比例无量纲。状态表缺少 ID，请先从**同次运行、未经重排的完整特征表**附加键，再筛选或排序。Table 2 的校正 Fisher–Pearson 偏度可从 Q 计算，少于三个角或角度恒定时无定义。

```python
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew

out = Path("results/run01")
keys = pd.read_csv(out / "candidate_features.csv.gz", usecols=["candidate_id", "z", "y", "x"])
states = pd.read_csv(out / "criterion_states.csv.gz")
assert len(keys) == len(states) and keys.candidate_id.is_unique
keyed_states = pd.concat([keys, states], axis=1).set_index(["z", "y", "x"], verify_integrity=True)
angles = pd.read_csv(out / "Q.csv.gz", usecols=["contact_angle_degrees"]).iloc[:, 0]
adjusted_skew = skew(angles, bias=False) if len(angles) >= 3 and angles.nunique() > 1 else np.nan
```

## Interpretation, data and citation / 适用范围、数据与引用

Voxelisation, segmentation and curvature at the chosen fitting scale affect the inferred angle. Q describes the distribution among accepted sites. A candidate count is a count of voxels along contact lines. The support scales and cutoffs are the study settings. Transfer to a different resolution or segmentation requires checking their physical scale and the fraction of candidates retained. See manuscript §§2.3, 2.4 and 3.2 and SI Tables S1–S2 for rationale, limitations and related methods.

体素化、分割质量及拟合尺度上的曲率都会影响角度。Q 描述通过筛选的测点分布，候选数量统计的是接触线上的体素数。将本研究的支撑尺度与阈值用于其他分辨率或分割方法时，需核对其物理尺度及候选保留比例。方法依据、局限和相关文献见正文 §§2.3、2.4、3.2 及 SI 表 S1–S2。

If you use this code, please cite Guo and colleagues’ revised manuscript, *Contact angle extraction with geometrical and morphological constraints in 3D porous media*, and identify the software commit used. The earlier SSRN preprint is titled *Physics-constrained contact angle extraction in 3D porous media*. The legacy implementation is archived at commit `888a63d`.
感谢使用本代码。引用时请注明 Guo 等人的上述修订稿及实际使用的提交号。SSRN 预印本沿用历史题名，历史实现可按提交号查阅。[Research data / 研究数据](https://figshare.com/articles/figure/3D_contact_angle_distribution_/30885293) · [Earlier SSRN preprint / 历史预印本](https://doi.org/10.2139/ssrn.6641861) · [Legacy software / 历史代码](https://github.com/Quincy-Zhenqi/Contact-angle-extraction-in-3D-porous-media/tree/888a63da642392db6efbe09a40b7f645633bfef5)
