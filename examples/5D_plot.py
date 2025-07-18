"""
==========================
3D Scatter Plot Example
==========================
Plot example for a 3D scatter plot using matplotlib.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def draw_box_edges(ax, xlim, ylim, zlim, color='black', linewidth=2):
    # 立方体的12条边（使用plot3D单独画每条）
    xs = xlim
    ys = ylim
    zs = zlim

    # 四条顶边
    ax.plot([xs[0], xs[1]], [ys[0], ys[0]], [zs[1], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[0], xs[1]], [ys[1], ys[1]], [zs[1], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[0], xs[0]], [ys[0], ys[1]], [zs[1], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[1], xs[1]], [ys[0], ys[1]], [zs[1], zs[1]], color=color, linewidth=linewidth)

    # 四条垂直棱
    ax.plot([xs[0], xs[0]], [ys[0], ys[0]], [zs[0], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[1], xs[1]], [ys[0], ys[0]], [zs[0], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[0], xs[0]], [ys[1], ys[1]], [zs[0], zs[1]], color=color, linewidth=linewidth)
    ax.plot([xs[1], xs[1]], [ys[1], ys[1]], [zs[0], zs[1]], color=color, linewidth=linewidth)

# 设置文件路径（请修改为你的实际路径）
file_path = "example_outputs/training_data_highlowmodice_temp_formatted.res"

# 读取数据
df = pd.read_csv(file_path, sep=r"\s+", header=None)
df.columns = ['co2', 'obliquity', 'esinw', 'ecosw', 'ice']
# 提取用于绘图的变量
x = df['esinw']        # X轴
y = df['ecosw']        # Y轴
z = df['obliquity']    # Z轴
color = df['co2']  # 映射为颜色，使用对数变换
size = df['ice']       # 映射为点大小

# 缩放点大小
size_scaled = 10 + 150 * (size - size.min()) / (size.max() - size.min())

# 创建三维散点图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=color, s=size_scaled, cmap='autumn_r',vmin=280,vmax=800, alpha=0.8, edgecolors='k')
ax.view_init(elev=25, azim=120)

xrange = [x.min(), x.max()]
yrange = [y.min(), y.max()]
zrange = [z.min(), z.max()]

ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_zlim(zrange)

draw_box_edges(ax, xrange, yrange, zrange, color='grey', linewidth=1)

# 添加颜色条（Colorbar）
cbar = plt.colorbar(sc, pad=0.1, ax=ax)
cbar.set_label("CO$_2$ concentration (ppm)", fontsize=12)

# 设置轴标签
ax.set_xlabel("esinω (X-axis)", fontsize=12)
ax.set_ylabel("ecosω (Y-axis)", fontsize=12)
ax.set_zlabel("Obliquity (°) (Z-axis)", fontsize=12)
ax.set_title("3D Scatter Plot: CO₂ as Color, SeaLevel as Size", fontsize=15)

# 添加点大小图例（Legend for ICE）
# 选取几个代表性的ice值
ice_legend_values = [size.min(), size.median(), size.max()]
size_legend_scaled = 10 + 150 * (np.array(ice_legend_values) - size.min()) / (size.max() - size.min())

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'SL = {ice_legend_values[i]:.1f}',
           markerfacecolor='gray', markersize=np.sqrt(size_legend_scaled[i]))
    for i in range(len(ice_legend_values))
]

ax.legend(handles=legend_elements, loc='upper left', title="Sea Level (circle size)", fontsize=10)

for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
    axis.line.set_linewidth(2.5)

# 网格/框调整
ax.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.set_box_aspect([1,1,1])  # 保持立方体比例
plt.tight_layout()
plt.savefig("3D_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()
