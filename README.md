# Contact angle extraction with geometrical and morphological constraints in 3D porous media

This Python code measures local contact angles along solid–liquid–gas contact lines in segmented CT volumes. It exports both the geometrically computable angles and a quality-controlled subset for comparing wettability across samples.

本程序沿分割 CT 体数据中的固–液–气三相接触线测量局部接触角，输出几何可计算的全部角度及其质量筛选子集，用于比较样品间的润湿性。

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

| Criterion / 条件 | Default / 默认判据 |
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

In a virtual environment (tested with 64-bit Python 3.8.2), run the following from the repository directory. The requirements file pins NumPy, SciPy, pandas and tifffile. Supply your segmented TIFF and choose a new output directory for each run.

在虚拟环境中从仓库目录运行以下命令。依赖文件固定了 NumPy、SciPy、pandas 和 tifffile 的版本，已用 64 位 Python 3.8.2 验证。将输入路径替换为自己的分割 TIFF，每次使用新的输出目录。

```sh
python -m pip install -r requirements.txt
python -c "from contact_angle_extraction import run_image; run_image('data/sample.tif', 'results/run01')"
```

`run_image` uses the defaults listed in `example_config.json`, including all local fitting points (`max_plane_points: null`). To study a parameter change, load your configuration and call `calculate` directly.

`run_image` 使用 `example_config.json` 所列的默认参数，其中 `max_plane_points: null` 表示使用全部局部拟合点。研究参数变化时，可按下例载入配置并调用 `calculate`。

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

Normals and centroids use **x, y, z** component suffixes. Centroid coordinates and plane RMS are in reference voxels. Before filtering or sorting the state table, attach `candidate_id, z, y, x` from the same run’s unsorted feature table in the original row order. For Table 2, calculate adjusted skewness with `scipy.stats.skew(angles, bias=False)` on Q’s `contact_angle_degrees` when at least three values are present and the angles vary.

法向量和质心分量的后缀为 **x、y、z**，质心坐标及平面 RMS 以参考体素计。状态表按同次运行的原始特征表行序排列；筛选或排序前，先逐行附上后者的 `candidate_id, z, y, x`。复算 Table 2 时，若 Q 中有至少三个角且角度不全相同，可对 `contact_angle_degrees` 列调用 `scipy.stats.skew(angles, bias=False)` 获得校正偏度。

## Tests / 测试

The tests generate perpendicular interfaces with a known 90° angle (43 candidates, 33 in G and 13 in Q). The optional second command compares the repository and Supplementary Material (SI) on this input, candidate by candidate. Supply your local SI document. Add `--tiff data/sample.tif` to compare another volume.

测试生成夹角为 90° 的双界面，预期有 43 个候选点，其中 G 有 33 个、Q 有 13 个。可选的第二条命令从本地补充材料（SI）独立提取源码，逐点比较两份程序的输出。附加 `--tiff data/sample.tif` 可核对其他体数据。

```sh
python -m unittest discover -s tests -v
python tests/compare_si.py "Supplementary Material_2026.11.20.docx"
```

## Research data and citation / 数据与引用

For the measurement assumptions and choice of thresholds, see manuscript §§2.3–2.4 and SI Tables S1–S2. The spacing, fitting window and segmentation affect the inferred angles. Please cite the manuscript below and record the Git commit used for your analysis.

测量假设与阈值选择见正文第 2.3–2.4 节和 SI 表 S1–S2。体素间距、拟合窗口和分割结果均会影响角度估计。欢迎引用以下稿件，并注明分析所用的 Git 提交号。[Research data on Figshare / 研究数据](https://figshare.com/articles/figure/3D_contact_angle_distribution_/30885293)

Zhenqi Guo, Feiyan Jin, Kun Wang, Yajuan Zhuang, Si Suo, Sergio Andres Galindo Torres, and Liang Lei. *Contact angle extraction with geometrical and morphological constraints in 3D porous media*. Unpublished manuscript.
