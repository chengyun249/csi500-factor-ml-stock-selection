# GitHub 上传检查清单

## 1. 建议上传的内容

```text
README.md
requirements.txt
config.yaml（如果有）
src/ 或根目录下的核心 py 脚本
reports/final/
reports/tables/ 里的关键结果表
reports/figures/ 里的关键图片
```

## 2. 建议不要上传的内容

```text
Tushare Token
任何包含个人账号、路径、密钥的文件
过大的原始 parquet / csv 数据
可从 API 重新下载的完整 raw 数据
__pycache__/
.ipynb_checkpoints/
临时日志文件
```

## 3. 建议上传的核心结果文件

```text
reports/final/final_core_top50_15bp_comparison.csv
reports/final/final_core_top100_15bp_comparison.csv
reports/final/final_model_ic_comparison.csv
reports/final/final_factor_ic_selected.csv
reports/final/final_project_summary.md
reports/final/project_report_draft.md
reports/figures/fig_16_final_top50_15bp_excess_return.png
```

## 4. README 中必须说明

```text
1. 数据来自 Tushare，用户需自行配置 TUSHARE_TOKEN；
2. 项目使用历史中证500成分股，避免简单幸存者偏差；
3. 财务数据使用 ann_date 做 point-in-time 对齐；
4. 最终基准为中证500全收益指数 h00905.CSI；
5. 当前结果为研究型回测，不构成投资建议。
```

## 5. 推荐 .gitignore

```text
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env
.venv/
venv/

# Data
data/raw/
data/processed/
data/model_outputs/
data/backtest_results/
*.parquet

# Personal / temporary
*.log
*.tmp
.DS_Store
```

## 6. 最终项目定位

```text
这是一个中证500截面多因子选股研究项目。
核心价值不是展示单一高收益曲线，而是展示从数据构建、因子检验、机器学习建模、组合回测到稳健性检验的完整研究流程。
```
