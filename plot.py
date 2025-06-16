import matplotlib.pyplot as plt

# 数据准备
relations_num = [30, 60, 100, 234]
models = {
    'ICL-13b': [0.426, 0.246, 0.208, 0.182],
    'ICL-70b': [0.418, 0.398, 0.352, 0.24],
    'ToolkenGPT-13b': [0.518, 0.332, 0.326, 0.216],
    'ToolkenGPT-70b': [0.532, 0.28, 0.364, 0.326],
    'TokenLearning-13b': [0.538, 0.488, 0.354, 0.238],
    'TokenLearning-70b': [0.632, 0.38, 0.358, 0.33]
}

# 颜色和标记样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'p', '*']

# 创建图形
plt.figure(figsize=(12, 7), dpi=100)

# 绘制每条折线
for i, (model, accuracies) in enumerate(models.items()):
    line = plt.plot(relations_num, accuracies,
                    label=model,
                    color=colors[i],
                    marker=markers[i],
                    linestyle='-',
                    linewidth=2.5,
                    markersize=10,
                    markeredgecolor='black',
                    markeredgewidth=0.5)

    # 添加数据标签
    # for x, y in zip(relations_num, accuracies):
    #     plt.text(x, y + 0.01, f'{y:.3f}',
    #              ha='center',
    #              va='bottom',
    #              fontsize=9,
    #              color=colors[i])

plt.xlabel("Number of Relations", fontsize=15, labelpad=10)
plt.ylabel('Accuracy', fontsize=15, labelpad=10)

# 调整坐标轴范围
plt.ylim(0.15, 0.68)

# 网格和边框设置
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 添加图例（移动到右上角内部）
plt.legend(loc='upper right',
           frameon=True,
           fontsize=15,
           title_fontsize=11)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()