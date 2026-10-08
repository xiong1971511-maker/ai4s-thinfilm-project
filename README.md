# AI4S Thin-Film Research Project

《薄膜技术》课程大作业：TMM 数据生成、MLP 光谱预测、训练规模实验与候选膜系物理验证。

## 个人参数与统一设置

- 参数编号：N = 56。
- **λtarget = 700 nm；seed = 270256；design_seed = 270257。**
- Air/H/L/H/L/Glass；nH=2.30，nL=1.45，ns=1.52；正入射、无吸收、忽略色散。
- 四层膜厚各 40–180 nm；400–800 nm，每 10 nm 一点，共 41 点。
- 5000 组数据，固定 4000/500/500 划分；训练规模使用同一训练池前 500/1000/2000/4000 组。
- 主模型 4–128–128–64–41，隐藏层 ReLU、线性输出，Adam/MSE，批大小 128，最多 2500 轮，早停耐心 100。
- 候选数 10000，按 MLP 的 R(700 nm)分别形成最小化/最大化 Top 10，再用 TMM 验证得到各自 Top 5；额外复核全候选池。

## 提交材料

1. Research Article：[Word](paper/AI4S_ThinFilm_Research_Article_Submission.docx) + [PDF](paper/AI4S_ThinFilm_Research_Article_Submission.pdf)，公开版 6 页；中文正文，标题、关键词、图注、表头及图内说明中英双语，省略姓名与完整学号；课程提交版保留模板要求的身份信息。
2. GitHub 仓库链接：https://github.com/xiong1971511-maker/ai4s-thinfilm-project
3. [作业要求逐项对应与最终验收记录](docs/FINAL_SUBMISSION_CHECKLIST.md)。课程平台的额外提交要求以平台通知为准。

## 环境配置

在仓库根目录运行。原主结果记录于 Windows/Python 3.14.7；最终检查在 Linux/Python 3.12.14 上使用相同的锁定 NumPy、PyTorch、Matplotlib 版本通过。推荐 Python 3.12 或 3.14、CPU 环境。

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Linux/macOS shell：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

## 直接验证论文已发布结果

```bash
python scripts/check_tmm.py
python -m unittest discover -s tests -v
python scripts/evaluate_mlp.py --checkpoint models_convergence_linear/mlp_train_4000.pt
python scripts/verify_submission.py
python scripts/make_submission_figures.py
```

`verify_submission.py` 从 seed 重建全部 5000 组数据及固定划分，加载已发布主模型独立计算验证/测试误差，重算全部 10000 个候选的 41 点 TMM 光谱，并核对短名单、全池极值、训练历史和失败案例。检查结果写入 `verification/submission_verification.json`。`make_submission_figures.py` 重建论文三个组合图，输出双语 PNG、SVG 和数据来源清单至 `paper/figures/`。脚本直接加载仓库附带的 OFL 授权中文字体子集 `assets/fonts/`；SVG 文字保存为矢量路径，无需额外安装字体。

主要结果：测试 RMSE **0.0113750**；500/1000/2000/4000 样本测试 RMSE 依次为 0.027351/0.019207/0.012742/0.011375。全池最低 R(700 nm)=0.00020416604（ID 3405），最高=0.65699974（ID 2256）。Table 1 的排名限于各 MLP Top-10 短名单，不能作为全池排名。

## 数据生成与从头训练

以下将新实验写到 `reproduced/`，保留论文原始结果。

```bash
python scripts/generate_dataset.py --output reproduced/thinfilm_dataset.npz
python scripts/train_mlp.py --dataset reproduced/thinfilm_dataset.npz --output-activation linear --max-epochs 2500 --patience 100 --models-dir reproduced/models_linear --outputs-dir reproduced/outputs_linear
python scripts/evaluate_mlp.py --dataset reproduced/thinfilm_dataset.npz --checkpoint reproduced/models_linear/mlp_train_4000.pt
python scripts/screen_designs.py --dataset reproduced/thinfilm_dataset.npz --checkpoint reproduced/models_linear/mlp_train_4000.pt --candidates 10000 --output-dir reproduced/design_linear
```

`train_mlp.py` 默认依次训练四个规定规模，各自使用固定训练池前缀；验证与测试集不变。随机数由显式设种的 NumPy Generator、`torch.manual_seed` 和 `torch.Generator` 控制。跨平台浮点运算可能改变训练轨迹，因此重新训练的检查点数值无需逐位等于原 Windows 记录；已发布检查点的独立推断在最终验证容差内一致。

## 筛选与全池 TMM 复核

使用论文已发布检查点重新筛选：

```bash
python scripts/screen_designs.py --checkpoint models_convergence_linear/mlp_train_4000.pt --candidates 10000 --output-dir reproduced/published_model_screening
python scripts/audit_linear_candidate_pool.py --output-dir reproduced/published_pool_tmm_audit
```

全池审计脚本读取论文保存的候选池与检查点；`--output-dir` 必须是尚不存在的新目录。若只是检查已发布结果，使用 `verify_submission.py`。最终膜系性能使用 TMM 值，线性代理的负反射率预测只作为排序诊断。

## 仓库材料索引

| 内容 | 路径 |
|---|---|
| 物理、数据、MLP、训练、筛选实现 | `src/ai4s_thinfilm/` |
| 生成、训练、评估、筛选、验证与画图脚本 | `scripts/` |
| 数据与固定划分 | `data/thinfilm_dataset.npz` |
| 已发布主模型 | `models_convergence_linear/mlp_train_4000.pt` |
| 四规模指标与完整损失历史 | `outputs_convergence_linear/mlp_metrics.json`、`loss_history_train_*.csv` |
| MLP Top 10 与 TMM 复核结果 | `outputs_convergence_linear/design_top10_{min,max}.csv`、相关 NPZ |
| 全池光谱、排名与极值 | `outputs_convergence_linear/global_tmm_audit/` |
| 高误差测试案例 | `paper/heldout_test_case_row_3477_spectra.csv` |
| 最终论文、图与图来源清单 | `paper/`、`paper/figures/figure_manifest.json` |
| 数值验收记录 | `verification/submission_verification.json` |

## 可选实验脚本

`compare_model_variants.py` 和旧版 `make_figures.py` 用于 Linear/Sigmoid 对照，需要先用 `train_mlp.py` 分别生成两种激活函数的全部模型、指标及历史，再显式指定输入路径；这些对照不是最终论文图表。`train_size_ablation.py`、`run_training_size_ablation.py` 是 250-epoch 可选实验，不能与论文 2500-epoch 主实验合并解释。`check_dataset.py` 验证 CSV 版本，需要先运行 `build_dataset_csv.py`；主要 NPZ 数据直接用最终验收脚本检查。

## 研究范围

本研究不包含吸收、色散、斜入射、偏振、制造误差和实验测量。仅采用一次固定划分；小样本组达到训练上限。随机候选池的极值不代表连续空间全局最优，也未测量加速倍数。请在本学期内保持仓库公开可访问。
