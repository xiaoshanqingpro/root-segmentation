# 根系扫描图像分割项目

用语义分割把扫描根系图逐像素分类（根 / 阴影 / 背景），输出与原图同分辨率的干净根 mask，
供后续根长、根面积测量使用。

**当前进度：阶段一（数据勘察）已完成 ✅，已完成预处理与 GPU 环境搭建，等待确认标注标准后进入阶段二。**

---

## 目录结构

```
根系分割项目/
├─ code/            按序号执行的脚本（00 → 09）
├─ docs/            阶段报告、标注判定标准
├─ processed/       预处理后的训练/推理输入（统一 600 dpi）
│   ├─ 600dpi/          主数据集 106 张
│   └─ 600dpi_excluded/ 排除的 2 张（透射背光件、叶片）
└─ outputs/         所有产出按阶段分目录
   └─ 阶段一/
```

原图**不被修改、不被复制**，脚本一律从 `..\图片\` 只读读取。

## 环境（已就绪 ✅）

| | |
|---|---|
| Python | 3.11.8 |
| GPU | RTX 5060 Laptop 8G（Blackwell，sm_120）—— **已验证可用** |
| torch | **2.11.0+cu128**（CUDA 12.8，`cuda_available=True`，前向+反向冒烟测试通过） |
| torchvision | 0.26.0+cu128 |
| 其他 | numpy / Pillow / opencv / scikit-image / scipy / matplotlib / pandas / tqdm / albumentations 1.4.6 / segmentation-models-pytorch 0.5.0 / timm |

> `albumentations` 2.x 需要 MSVC 编译 `stringzilla`，Windows 上装不了，已固定用 **1.4.6**。
> cu128 源上 torch 最高只到 2.11.0，所以是"降版本换 GPU"。

## 脚本

```powershell
cd 根系分割项目\code
python 00_env_probe.py            # GPU / 依赖探针（含小卷积前向+反向冒烟测试）
python 01_data_survey.py          # 逐张勘察 -> survey.csv
python 02_shadow_proposal.py      # 抽样 9 张 + 阴影判定可视化
python 03_classic_baseline.py     # 经典流水线基线对比
python 04_contact_sheet.py        # 108 张缩略图总表
python 05_shadow_root_overlap.py  # 阴影↔根 距离分布
python 06_survey_stats.py         # 报告汇总 + mm/px 换算表
python 07_crop_check.py           # 裁剪线可视化
python 08_edge_band.py            # 四边伪影带逐张测量（含粉色刻度尺专用判据）
python 09_preprocess.py [--dry-run]  # 按图裁边 + 统一重采样到 600 dpi
```

## 关键约定（后续阶段必须遵守）

- **随机种子固定**（阶段一抽样 `seed=42`）。
- **输出命名**：`<原文件名>_mask.png`，与原图一一对应、同分辨率、逐像素对齐。
- **数据集按整图划分** train/val/test，同一张图的 patch 不跨集合。
- **标注打版本号**：`mask_v1`、`mask_v2`…… 旧版本全保留；badcase 回炉后**从头重训**，不做增量微调。
- **每阶段先给你看结果，确认后才进下一阶段**。

## dpi 换算（测量要用，容易踩坑）

**已由 `09_preprocess.py` 统一解决**：`processed/600dpi/` 里全部是 600 dpi，
统一按 **1 px = 0.042333 mm，1 mm = 23.62 px** 换算即可。

原始文件的差异（仅作溯源参考）：

| 批次 | 像素尺寸 | dpi | mm/px |
|---|---|---|---|
| 绝大多数 | 4962 × 7019 | 600 | 0.042333 |
| `0814` 的 10 张 | 5954 × 8423 | **720** | **0.035278** |
| `0814\G218-002` | 5100 × 7019 | 600 | 0.042333（幅面异常，待确认） |

**全库不能用同一个换算系数**，逐张数值见 `outputs/阶段一/measure_lookup.csv`。
