# compare_all_errors.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- 1. 配置实验信息 ---
# 定义每个实验的名称和日志文件路径
experiments = [
    {
        "label": "单卡",
        "path": "pics/单卡sft-pad/training.log"
    },
    {
        "label": "双卡 TP+SP",
        "path": "pics/双卡 tp+sp-sft/training.log"
    },
    {
        "label": "双卡 TP",
        "path": "pics/双卡sft-pad/training.log"
    }
]

# --- 2. 日志解析与绘图辅助函数 ---

def parse_log_file(filepath):
    """从指定的日志文件中解析出 global_step 和 loss。"""
    if not os.path.exists(filepath):
        print(f"❌ 警告: 找不到日志文件 {filepath}，将跳过此实验。")
        return None
    train_pattern = re.compile(r"loss: (?P<loss>[\d\.]+),.*?global_step: (?P<step>\d+)")
    steps, losses = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = train_pattern.search(line)
            if match:
                steps.append(int(match.group('step')))
                losses.append(float(match.group('loss')))
    return steps, losses

def plot_loss_comparison(ax, df):
    """在给定的 axes 上绘制三条 Loss 对比曲线。"""
    labels = df.columns
    styles = [
        {'color': 'deepskyblue', 'linestyle': '-', 'marker': 'o', 'label': labels[0]},
        {'color': 'darkorange', 'linestyle': '--', 'marker': 's', 'label': labels[1]},
        {'color': 'limegreen', 'linestyle': ':', 'marker': '^', 'label': labels[2]}
    ]
    for i, col in enumerate(df.columns):
        df[col].plot(ax=ax, color=styles[i]['color'], linestyle=styles[i]['linestyle'],
                     marker=styles[i]['marker'], linewidth=2, markersize=5,
                     alpha=0.8, label=styles[i]['label'])
    ax.set_title('Loss Comparison (Enhanced Visibility)', fontsize=16)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

# --- 3. 主逻辑：数据处理 ---

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

all_series = []
for exp in experiments:
    print(f"⚙️ 正在处理: {exp['label']}...")
    result = parse_log_file(exp['path'])
    if result:
        steps, losses = result
        series = pd.Series(losses, index=steps, name=exp['label'])
        all_series.append(series)

df_loss = pd.concat(all_series, axis=1).ffill()
print("\n✅ 数据解析和对齐完成。")

label1, label2, label3 = df_loss.columns

# --- 4. 计算绝对差异和相对误差 ---

# 绝对差异
diff_abs_1_vs_2 = (df_loss[label1] - df_loss[label2]).abs()
diff_abs_1_vs_3 = (df_loss[label1] - df_loss[label3]).abs()
diff_abs_2_vs_3 = (df_loss[label2] - df_loss[label3]).abs()

# 相对误差
rel_err_1_vs_2 = diff_abs_1_vs_2 / (df_loss[label1] + 1e-9)
rel_err_1_vs_3 = diff_abs_1_vs_3 / (df_loss[label1] + 1e-9)
rel_err_2_vs_3 = diff_abs_2_vs_3 / (df_loss[label2] + 1e-9)

# <<< 主要修改点 (1/2): 计算统一的 Y 轴范围 >>>
# 将三组差异数据合并，找到全局最大值，并增加 5% 的边距
max_abs_diff = pd.concat([diff_abs_1_vs_2, diff_abs_1_vs_3, diff_abs_2_vs_3]).max()
max_rel_err = pd.concat([rel_err_1_vs_2, rel_err_1_vs_3, rel_err_2_vs_3]).replace([np.inf, -np.inf], np.nan).max()

# --- 5. 绘制第一张图：绝对差异 ---

print("\n🎨 正在生成第一张图 (绝对差异)...")
fig1, axes1 = plt.subplots(2, 2, figsize=(20, 16))
fig1.suptitle('Analysis of Absolute Loss Difference', fontsize=20)

plot_loss_comparison(axes1[0, 0], df_loss)

# 绘制绝对差异图
diff_abs_1_vs_2.dropna().plot(ax=axes1[0, 1], color='purple', linewidth=2)
axes1[0, 1].set_title(f'Absolute Loss Difference\n({label1} vs {label2})', fontsize=16)
diff_abs_1_vs_3.dropna().plot(ax=axes1[1, 0], color='brown', linewidth=2)
axes1[1, 0].set_title(f'Absolute Loss Difference\n({label1} vs {label3})', fontsize=16)
diff_abs_2_vs_3.dropna().plot(ax=axes1[1, 1], color='teal', linewidth=2)
axes1[1, 1].set_title(f'Absolute Loss Difference\n({label2} vs {label3})', fontsize=16)

# <<< 主要修改点 (2/2): 为后三个子图应用统一的 Y 轴范围 >>>
for ax in [axes1[0, 1], axes1[1, 0], axes1[1, 1]]:
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Absolute Difference')
    ax.grid(True, linestyle='--', alpha=0.6)
    if max_abs_diff > 0:
        ax.set_ylim(0, max_abs_diff * 1.05) # 设置统一的Y轴上限

fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path1 = 'loss_comparison_ABSOLUTE.png'
fig1.savefig(save_path1)
print(f"🎉 第一张图已成功保存至: {save_path1}")

# --- 6. 绘制第二张图：相对误差 ---

print("\n🎨 正在生成第二张图 (相对误差)...")
fig2, axes2 = plt.subplots(2, 2, figsize=(20, 16))
fig2.suptitle('Analysis of Relative Loss Error', fontsize=20)

plot_loss_comparison(axes2[0, 0], df_loss)

# 绘制相对误差图
rel_err_1_vs_2.dropna().plot(ax=axes2[0, 1], color='purple', linewidth=2)
axes2[0, 1].set_title(f'Relative Loss Error\n(|{label1} - {label2}| / {label1})', fontsize=16)
rel_err_1_vs_3.dropna().plot(ax=axes2[1, 0], color='brown', linewidth=2)
axes2[1, 0].set_title(f'Relative Loss Error\n(|{label1} - {label3}| / {label1})', fontsize=16)
rel_err_2_vs_3.dropna().plot(ax=axes2[1, 1], color='teal', linewidth=2)
axes2[1, 1].set_title(f'Relative Loss Error\n(|{label2} - {label3}| / {label2})', fontsize=16)

for ax in [axes2[0, 1], axes2[1, 0], axes2[1, 1]]:
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Relative Error')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle='--', alpha=0.6)
    if max_rel_err > 0:
        ax.set_ylim(0, max_rel_err * 1.05) # 设置统一的Y轴上限

fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path2 = 'loss_comparison_RELATIVE.png'
fig2.savefig(save_path2)
print(f"🎉 第二张图已成功保存至: {save_path2}")