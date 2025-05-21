import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import tkinterdnd2 as tkdnd  # 用于拖放文件夹
from matplotlib.ticker import LogLocator, MaxNLocator
import re

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ------------------ 数据读取与处理函数 -------------------

def read_data(file_path):
    """读取温度数据 CSV 文件"""
    data = pd.read_csv(file_path, skiprows=1)
    data.columns = ['x', 'y', 'z', 'temperature']
    return data


def find_max_temperature_point(data):
    """返回温度最高点的行数据"""
    max_temp_idx = data['temperature'].idxmax()
    return data.loc[max_temp_idx]


def read_e_field_data(file_path):
    """读取电场数据文件"""
    try:
        data = pd.read_csv(file_path, header=None, skiprows=2, dtype=float, low_memory=False)
        data.columns = ['x', 'y', 'z', 'ExRe', 'ExIm', 'EyRe', 'EyIm', 'EzRe', 'EzIm']
        return data
    except Exception as e:
        messagebox.showerror("错误", f"无法读取电场文件: {e}")
        return None


def calculate_e_field(data):
    """计算电场强度 E = sqrt(ExRe² + EyRe² + EzRe²)"""
    if data is None:
        return None
    for col in ['ExRe', 'EyRe', 'EzRe']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['E'] = np.sqrt(data['ExRe'] ** 2 + data['EyRe'] ** 2 + data['EzRe'] ** 2)
    return data


def read_heat_flow_data(file_path):
    """读取热流场数据文件"""
    try:
        data = pd.read_csv(file_path, header=None, skiprows=1, dtype=float, low_memory=False)
        data.columns = ['x', 'y', 'z', 'Qx', 'Qy', 'Qz']
        return data
    except Exception as e:
        messagebox.showerror("错误", f"无法读取热流场文件: {e}")
        return None


def calculate_heat_flow(data):
    """计算热流场强度 Q = sqrt(Qx² + Qy² + Qz²)"""
    if data is None:
        return None
    for col in ['Qx', 'Qy', 'Qz']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['Q'] = np.sqrt(data['Qx'] ** 2 + data['Qy'] ** 2 + data['Qz'] ** 2)
    return data


# ------------------- 绘图函数 -------------------

def plot_temperature_field(ax, data, plane, value, clim_min=None, clim_max=None, scale='linear'):
    print(f"plot_temperature_field: clim_min={clim_min}, clim_max={clim_max}, scale={scale}")  # 打印调试

    max_point = find_max_temperature_point(data)

    if plane == 'x':
        d = data[data['x'] == value]
        if d.empty:
            idx = (data['x'] - value).abs().argsort().iloc[0]
            value = data['x'].iloc[idx]
            d = data[data['x'] == value]
        y = d['y'].values
        z = d['z'].values
        temp = d['temperature'].values
        ax.set_title(f"Temperature Field on YOZ Plane (x={value:.2f})")
        xlabel, ylabel = 'Y', 'Z'
        uy = np.sort(np.unique(y))
        uz = np.sort(np.unique(z))
        yi, zi = np.meshgrid(uy, uz)
        temp_grid = griddata((y, z), temp, (yi, zi), method='cubic')
    elif plane == 'y':
        d = data[data['y'] == value]
        if d.empty:
            idx = (data['y'] - value).abs().argsort().iloc[0]
            value = data['y'].iloc[idx]
            d = data[data['y'] == value]
        x = d['x'].values
        z = d['z'].values
        temp = d['temperature'].values
        ax.set_title(f"Temperature Field on XOZ Plane (y={value:.2f})")
        xlabel, ylabel = 'X', 'Z'
        ux = np.sort(np.unique(x))
        uz = np.sort(np.unique(z))
        xi, zi = np.meshgrid(ux, uz)
        temp_grid = griddata((x, z), temp, (xi, zi), method='cubic')
    elif plane == 'z':
        d = data[data['z'] == value]
        if d.empty:
            idx = (data['z'] - value).abs().argsort().iloc[0]
            value = data['z'].iloc[idx]
            d = data[data['z'] == value]
        x = d['x'].values
        y = d['y'].values
        temp = d['temperature'].values
        ax.set_title(f"Temperature Field on XOY Plane (z={value:.2f})")
        xlabel, ylabel = 'X', 'Y'
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        xi, yi = np.meshgrid(ux, uy)
        temp_grid = griddata((x, y), temp, (xi, yi), method='cubic')
    else:
        messagebox.showerror("错误", "无效的平面选择")
        return None

    # 将小于等于0的值替换为大于0的最小值
    valid_values = temp_grid[temp_grid > 0]
    if len(valid_values) > 0:
        min_positive = np.min(valid_values)
        temp_grid[temp_grid <= 0] = min_positive
    else:
        messagebox.showerror("错误", "所有温度数据都小于等于0")
        return None

    # 使用用户输入的 clim_min 和 clim_max
    if clim_min is None:
        clim_min = np.nanmin(temp_grid)
    if clim_max is None:
        clim_max = np.nanmax(temp_grid)

    # 裁剪数据到 clim_min 和 clim_max 范围内
    temp_grid_clipped = np.clip(temp_grid, clim_min, clim_max)

    if scale == 'log':
        if clim_min <= 0:
            clim_min = np.nanmin(temp_grid_clipped[temp_grid_clipped > 0])
        norm = LogNorm(vmin=clim_min, vmax=clim_max)
        levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 300)
        if plane == 'x':
            cont = ax.contourf(yi, zi, temp_grid_clipped, levels=levels, cmap='jet', norm=norm)
        elif plane == 'y':
            cont = ax.contourf(xi, zi, temp_grid_clipped, levels=levels, cmap='jet', norm=norm)
        elif plane == 'z':
            cont = ax.contourf(xi, yi, temp_grid_clipped, levels=levels, cmap='jet', norm=norm)
    else:
        if plane == 'x':
            cont = ax.contourf(yi, zi, temp_grid_clipped, levels=200, cmap='jet', vmin=clim_min, vmax=clim_max)
        elif plane == 'y':
            cont = ax.contourf(xi, zi, temp_grid_clipped, levels=200, cmap='jet', vmin=clim_min, vmax=clim_max)
        elif plane == 'z':
            cont = ax.contourf(xi, yi, temp_grid_clipped, levels=200, cmap='jet', vmin=clim_min, vmax=clim_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加颜色条并设置更多刻度
    fig = ax.get_figure()
    cbar = fig.colorbar(cont, ax=ax, label="Temperature")

    # 增加颜色条的刻度数量
    if scale == 'log':
        locator = LogLocator(numticks=10)
        cbar.locator = locator
        cbar.update_ticks()
    else:
        locator = MaxNLocator(nbins=10)
        cbar.locator = locator
        cbar.update_ticks()

    # 添加图例，包括温度最高点的坐标
    if plane == 'x':
        ax.scatter([max_point['y']], [max_point['z']], color='white', edgecolor='black', s=100,
                   label=f'Max Temp at ({max_point["x"]:.2f}, {max_point["y"]:.2f}, {max_point["z"]:.2f})')
    elif plane == 'y':
        ax.scatter([max_point['x']], [max_point['z']], color='white', edgecolor='black', s=100,
                   label=f'Max Temp at ({max_point["x"]:.2f}, {max_point["y"]:.2f}, {max_point["z"]:.2f})')
    elif plane == 'z':
        ax.scatter([max_point['x']], [max_point['y']], color='white', edgecolor='black', s=100,
                   label=f'Max Temp at ({max_point["x"]:.2f}, {max_point["y"]:.2f}, {max_point["z"]:.2f})')
    ax.legend()

    # 设置坐标轴范围以确保图形填满整个绘图区域
    if plane == 'x':
        ax.set_xlim(np.min(yi), np.max(yi))
        ax.set_ylim(np.min(zi), np.max(zi))
    elif plane == 'y':
        ax.set_xlim(np.min(xi), np.max(xi))
        ax.set_ylim(np.min(zi), np.max(zi))
    elif plane == 'z':
        ax.set_xlim(np.min(xi), np.max(xi))
        ax.set_ylim(np.min(yi), np.max(yi))

    return cont, cbar


def plot_s_parameter_subplots(tab, folder_path, files):
    """
    绘制 2x2 的 S 参数子图
    """
    file_groups = {
        'S1,1': ['S1,1.txt', 'S1,1_ori.txt'],
        'S1,2': ['S1,2.txt', 'S1,2_ori.txt'],
        'S2,1': ['S2,1.txt', 'S2,1_ori.txt'],
        'S2,2': ['S2,2.txt', 'S2,2_ori.txt']
    }
    min_freq = float('inf')
    max_freq = -float('inf')
    current_dir_files = os.listdir(os.getcwd())
    dragged_files = []

    # 获取拖放文件夹中的文件
    for file in files:
        if file.endswith('.txt') and 'S' in file:
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                min_freq = min(min_freq, data['Frequency'].min())
                max_freq = max(max_freq, data['Frequency'].max())
                dragged_files.append(file)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

    # 创建一个新的 Figure，包含 4 个子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('S Parameter Curves', fontsize=16)

    # 绘制 4 个子图
    for i, (group_name, group_files) in enumerate(file_groups.items()):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.set_title(f'{group_name} Curves', fontsize=12)
        ax.set_xlabel('Frequency / GHz', fontsize=10)
        ax.set_ylabel('S Value / dB', fontsize=10)
        ax.set_xlim(min_freq, max_freq)
        ax.grid(True, linestyle='--', alpha=0.7)

        for file in group_files:
            file_path = None
            if file in dragged_files:
                file_path = os.path.join(folder_path, file)
            elif file in current_dir_files:
                file_path = os.path.join(os.getcwd(), file)
            else:
                continue

            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                if len(data) > 200:
                    indices = np.linspace(0, len(data) - 1, 200, dtype=int)
                    data = data.iloc[indices]
                legend_label = file.split('.')[0]

                # 根据文件名是否以 _ori 结尾来设置线型
                linestyle = '--' if '_ori' in legend_label else '-'

                ax.plot(data['Frequency'], data['S_value'], label=legend_label, linestyle=linestyle)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

        ax.legend()

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 将新创建的 Figure 转换为 Tkinter 的画布
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 更新画布
    canvas.draw()


def plot_s_parameter_main(ax, folder_path, files):
    """
    绘制总的 S 参数曲线图
    """
    file_groups = {
        'S1,1': ['S1,1.txt', 'S1,1_ori.txt'],
        'S1,2': ['S1,2.txt', 'S1,2_ori.txt'],
        'S2,1': ['S2,1.txt', 'S2,1_ori.txt'],
        'S2,2': ['S2,2.txt', 'S2,2_ori.txt']
    }
    min_freq = float('inf')
    max_freq = -float('inf')
    current_dir_files = os.listdir(os.getcwd())
    dragged_files = []

    # 获取拖放文件夹中的文件
    for file in files:
        if file.endswith('.txt') and 'S' in file:
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                min_freq = min(min_freq, data['Frequency'].min())
                max_freq = max(max_freq, data['Frequency'].max())
                dragged_files.append(file)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

    # 绘制主图
    ax.set_title('All S Parameters Curves', fontsize=14)
    ax.set_xlabel('Frequency / GHz', fontsize=12)
    ax.set_ylabel('S Value / dB', fontsize=12)
    ax.set_xlim(min_freq, max_freq)
    ax.grid(True, linestyle='--', alpha=0.7)

    for group_name, group_files in file_groups.items():
        for file in group_files:
            file_path = None
            if file in dragged_files:
                file_path = os.path.join(folder_path, file)
            elif file in current_dir_files:
                file_path = os.path.join(os.getcwd(), file)
            else:
                continue

            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                if len(data) > 200:
                    indices = np.linspace(0, len(data) - 1, 200, dtype=int)
                    data = data.iloc[indices]
                legend_label = file.split('.')[0]

                # 根据文件名是否以 _ori 结尾来设置线型
                linestyle = '--' if '_ori' in legend_label else '-'

                ax.plot(data['Frequency'], data['S_value'], label=legend_label, linestyle=linestyle)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

    ax.legend()


def create_s_parameter_subplots_figure(self):
    """
    创建 S 参数子图的 Figure
    """
    file_groups = {
        'S1,1': ['S1,1.txt', 'S1,1_ori.txt'],
        'S1,2': ['S1,2.txt', 'S1,2_ori.txt'],
        'S2,1': ['S2,1.txt', 'S2,1_ori.txt'],
        'S2,2': ['S2,2.txt', 'S2,2_ori.txt']
    }
    min_freq = float('inf')
    max_freq = -float('inf')

    # 获取拖放文件夹中的文件
    for file in self.dragged_files:
        if file.endswith('.txt') and 'S' in file:
            file_path = os.path.join(self.folder_path, file)
            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                min_freq = min(min_freq, data['Frequency'].min())
                max_freq = max(max_freq, data['Frequency'].max())
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

    # 创建一个新的 Figure，包含 4 个子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('S Parameter Curves', fontsize=16)

    # 绘制 4 个子图
    for i, (group_name, group_files) in enumerate(file_groups.items()):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.set_title(f'{group_name} Curves', fontsize=12)
        ax.set_xlabel('Frequency / GHz', fontsize=10)
        ax.set_ylabel('S Value / dB', fontsize=10)
        ax.set_xlim(min_freq, max_freq)
        ax.grid(True, linestyle='--', alpha=0.7)

        for file in group_files:
            file_path = None
            if file in self.dragged_files:
                file_path = os.path.join(self.folder_path, file)
            else:
                continue

            try:
                data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                if len(data) > 200:
                    indices = np.linspace(0, len(data) - 1, 200, dtype=int)
                    data = data.iloc[indices]
                legend_label = file.split('.')[0]

                # 根据文件名是否以 _ori 结尾来设置线型
                linestyle = '--' if '_ori' in legend_label else '-'

                ax.plot(data['Frequency'], data['S_value'], label=legend_label, linestyle=linestyle)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue

        ax.legend()

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig, axs


def plot_e_field_distribution(ax, data, plane, value, clim_min=None, clim_max=None, scale='linear'):
    print(f"plot_e_field_distribution: clim_min={clim_min}, clim_max={clim_max}, scale={scale}")  # 打印调试

    if data is None:
        return None

    # 初始化 xi, yi, zi
    xi, yi, zi = None, None, None
    grid = None

    if plane == 'x':
        d = data[data['x'] == value]
        if d.empty:
            idx = (data['x'] - value).abs().argsort().iloc[0]
            value = data['x'].iloc[idx]
            d = data[data['x'] == value]
        y = d['y'].values
        z = d['z'].values
        E = d['E'].values
        ax.set_title(f"Electric Field on YOZ Plane (x={value:.2f})")
        xlabel, ylabel = 'Y', 'Z'
        uy = np.sort(np.unique(y))
        uz = np.sort(np.unique(z))
        yi, zi = np.meshgrid(uy, uz)
        grid = griddata((y, z), E, (yi, zi), method='cubic')
    elif plane == 'y':
        d = data[data['y'] == value]
        if d.empty:
            idx = (data['y'] - value).abs().argsort().iloc[0]
            value = data['y'].iloc[idx]
            d = data[data['y'] == value]
        x = d['x'].values
        z = d['z'].values
        E = d['E'].values
        ax.set_title(f"Electric Field on XOZ Plane (y={value:.2f})")
        xlabel, ylabel = 'X', 'Z'
        ux = np.sort(np.unique(x))
        uz = np.sort(np.unique(z))
        xi, zi = np.meshgrid(ux, uz)
        grid = griddata((x, z), E, (xi, zi), method='cubic')
    elif plane == 'z':
        d = data[data['z'] == value]
        if d.empty:
            idx = (data['z'] - value).abs().argsort().iloc[0]
            value = data['z'].iloc[idx]
            d = data[data['z'] == value]
        x = d['x'].values
        y = d['y'].values
        E = d['E'].values
        ax.set_title(f"Electric Field on XOY Plane (z={value:.2f})")
        xlabel, ylabel = 'X', 'Y'
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        xi, yi = np.meshgrid(ux, uy)
        grid = griddata((x, y), E, (xi, yi), method='cubic')
    else:
        messagebox.showerror("错误", "无效的平面选择")
        return None

    # 将小于0的值替换为大于等于0的最小值
    valid_values = grid[grid >= 0]
    if len(valid_values) > 0:
        min_positive = np.min(valid_values)
        grid[grid < 0] = min_positive
    else:
        messagebox.showerror("错误", "所有电场数据都小于等于0")
        return None

    # 如果 clim_min 或 clim_max 为空，使用数据的最小值和最大值
    if clim_min is None:
        clim_min = np.nanmin(grid)
    if clim_max is None:
        clim_max = np.nanmax(grid)

    # 确保 xi 和 yi 都是二维数组
    if plane == 'x':
        if yi is None or zi is None:
            messagebox.showerror("错误", "网格变量 yi 或 zi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 500)
            cont = ax.contourf(yi, zi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(yi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
    elif plane == 'y':
        if xi is None or zi is None:
            messagebox.showerror("错误", "网格变量 xi 或 zi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 500)
            cont = ax.contourf(xi, zi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(xi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
    elif plane == 'z':
        if xi is None or yi is None:
            messagebox.showerror("错误", "网格变量 xi 或 yi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 500)
            cont = ax.contourf(xi, yi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(xi, yi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加颜色条并设置更多刻度
    fig = ax.get_figure()
    cbar = fig.colorbar(cont, ax=ax, label="Electric Field")

    # 增加颜色条的刻度数量
    if scale == 'log':
        locator = LogLocator(numticks=10)  # 设置刻度数量
        cbar.locator = locator
        cbar.update_ticks()
    else:
        locator = MaxNLocator(nbins=10)  # 设置刻度数量
        cbar.locator = locator
        cbar.update_ticks()

    return cont, cbar


def plot_heat_flow_distribution(ax, data, plane, value, clim_min=None, clim_max=None, scale='linear'):
    print(f"plot_heat_flow_distribution: clim_min={clim_min}, clim_max={clim_max}, scale={scale}")  # 打印调试

    if data is None:
        return None

    # 初始化 xi, yi, zi
    xi, yi, zi = None, None, None
    grid = None

    if plane == 'x':
        d = data[data['x'] == value]
        if d.empty:
            idx = (data['x'] - value).abs().argsort().iloc[0]
            value = data['x'].iloc[idx]
            d = data[data['x'] == value]
        y = d['y'].values
        z = d['z'].values
        Q = d['Q'].values
        ax.set_title(f"Heat Flow on YOZ Plane (x={value:.2f})")
        xlabel, ylabel = 'Y', 'Z'
        uy = np.sort(np.unique(y))
        uz = np.sort(np.unique(z))
        yi, zi = np.meshgrid(uy, uz)
        grid = griddata((y, z), Q, (yi, zi), method='cubic')
    elif plane == 'y':
        d = data[data['y'] == value]
        if d.empty:
            idx = (data['y'] - value).abs().argsort().iloc[0]
            value = data['y'].iloc[idx]
            d = data[data['y'] == value]
        x = d['x'].values
        z = d['z'].values
        Q = d['Q'].values
        ax.set_title(f"Heat Flow on XOZ Plane (y={value:.2f})")
        xlabel, ylabel = 'X', 'Z'
        ux = np.sort(np.unique(x))
        uz = np.sort(np.unique(z))
        xi, zi = np.meshgrid(ux, uz)
        grid = griddata((x, z), Q, (xi, zi), method='cubic')
    elif plane == 'z':
        d = data[data['z'] == value]
        if d.empty:
            idx = (data['z'] - value).abs().argsort().iloc[0]
            value = data['z'].iloc[idx]
            d = data[data['z'] == value]
        x = d['x'].values
        y = d['y'].values
        Q = d['Q'].values
        ax.set_title(f"Heat Flow on XOY Plane (z={value:.2f})")
        xlabel, ylabel = 'X', 'Y'
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        xi, yi = np.meshgrid(ux, uy)
        grid = griddata((x, y), Q, (xi, yi), method='cubic')
    else:
        messagebox.showerror("错误", "无效的平面选择")
        return None

    # 将小于0的值替换为大于等于0的最小值
    valid_values = grid[grid >= 0]
    if len(valid_values) > 0:
        min_positive = np.min(valid_values)
        grid[grid < 0] = min_positive
    else:
        messagebox.showerror("错误", "所有热流场数据都小于等于0")
        return None

    # 如果 clim_min 或 clim_max 为空，使用数据的最小值和最大值
    if clim_min is None:
        clim_min = np.nanmin(grid)
    if clim_max is None:
        clim_max = np.nanmax(grid)

    # 确保 xi 和 yi 都是二维数组
    if plane == 'x':
        if yi is None or zi is None:
            messagebox.showerror("错误", "网格变量 yi 或 zi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 300)
            cont = ax.contourf(yi, zi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(yi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
    elif plane == 'y':
        if xi is None or zi is None:
            messagebox.showerror("错误", "网格变量 xi 或 zi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 300)
            cont = ax.contourf(xi, zi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(xi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
    elif plane == 'z':
        if xi is None or yi is None:
            messagebox.showerror("错误", "网格变量 xi 或 yi 未定义")
            return None
        if scale == 'log':
            if clim_min <= 0:
                clim_min = np.nanmin(grid[grid > 0])
            norm = LogNorm(vmin=clim_min, vmax=clim_max)
            levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 300)
            cont = ax.contourf(xi, yi, grid, levels=levels, cmap='jet', norm=norm)
        else:
            cont = ax.contourf(xi, yi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加颜色条并设置更多刻度
    fig = ax.get_figure()
    cbar = fig.colorbar(cont, ax=ax, label="Heat Flow")

    # 增加颜色条的刻度数量
    if scale == 'log':
        locator = LogLocator(numticks=10)  # 设置刻度数量
        cbar.locator = locator
        cbar.update_ticks()
    else:
        locator = MaxNLocator(nbins=10)  # 设置刻度数量
        cbar.locator = locator
        cbar.update_ticks()

    return cont, cbar


def read_displacement_data(file_path):
    """读取位移场数据文件"""
    try:
        data = pd.read_csv(file_path, header=None, skiprows=1, dtype=float, low_memory=False)
        data.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz']
        return data
    except Exception as e:
        messagebox.showerror("错误", f"无法读取位移场文件: {e}")
        return None


def calculate_displacement(data):
    """计算位移场强度 d = sqrt(dx² + dy² + dz²)"""
    if data is None:
        return None
    for col in ['dx', 'dy', 'dz']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['d'] = np.sqrt(data['dx'] ** 2 + data['dy'] ** 2 + data['dz'] ** 2)
    return data


# ------------------ 新增力场绘图函数 -------------------
def plot_force_field_distribution(ax, data, plane, value, clim_min=None, clim_max=None, scale='linear'):
    print(f"plot_force_field_distribution: clim_min={clim_min}, clim_max={clim_max}, scale={scale}")  # 打印调试

    if data is None:
        return None

    # 初始化网格变量
    xi, yi, zi = None, None, None
    grid = None

    if plane == 'x':
        d = data[data['x'] == value]
        if d.empty:
            idx = (data['x'] - value).abs().argsort().iloc[0]
            value = data['x'].iloc[idx]
            d = data[data['x'] == value]
        y = d['y'].values
        z = d['z'].values
        disp = d['d'].values
        ax.set_title(f"Displacement Field on YOZ Plane (x={value:.2f})")
        xlabel, ylabel = 'Y', 'Z'
        uy = np.sort(np.unique(y))
        uz = np.sort(np.unique(z))
        yi, zi = np.meshgrid(uy, uz)
        grid = griddata((y, z), disp, (yi, zi), method='cubic')
    elif plane == 'y':
        d = data[data['y'] == value]
        if d.empty:
            idx = (data['y'] - value).abs().argsort().iloc[0]
            value = data['y'].iloc[idx]
            d = data[data['y'] == value]
        x = d['x'].values
        z = d['z'].values
        disp = d['d'].values
        ax.set_title(f"Displacement Field on XOZ Plane (y={value:.2f})")
        xlabel, ylabel = 'X', 'Z'
        ux = np.sort(np.unique(x))
        uz = np.sort(np.unique(z))
        xi, zi = np.meshgrid(ux, uz)
        grid = griddata((x, z), disp, (xi, zi), method='cubic')
    elif plane == 'z':
        d = data[data['z'] == value]
        if d.empty:
            idx = (data['z'] - value).abs().argsort().iloc[0]
            value = data['z'].iloc[idx]
            d = data[data['z'] == value]
        x = d['x'].values
        y = d['y'].values
        disp = d['d'].values
        ax.set_title(f"Displacement Field on XOY Plane (z={value:.2f})")
        xlabel, ylabel = 'X', 'Y'
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        xi, yi = np.meshgrid(ux, uy)
        grid = griddata((x, y), disp, (xi, yi), method='cubic')
    else:
        messagebox.showerror("错误", "无效的平面选择")
        return None

    # 处理非正值数据
    valid_values = grid[grid >= 0]
    if len(valid_values) > 0:
        min_positive = np.min(valid_values)
        grid[grid < 0] = min_positive
    else:
        messagebox.showerror("错误", "所有位移场数据都小于等于0")
        return None

    # 设置颜色范围
    if clim_min is None:
        clim_min = np.nanmin(grid)
    if clim_max is None:
        clim_max = np.nanmax(grid)

    # 绘制等高线图
    if scale == 'log':
        if clim_min <= 0:
            clim_min = np.nanmin(grid[grid > 0])
        norm = LogNorm(vmin=clim_min, vmax=clim_max)
        levels = np.logspace(np.log10(clim_min), np.log10(clim_max), 300)
        if plane == 'x':
            cont = ax.contourf(yi, zi, grid, levels=levels, cmap='jet', norm=norm)
        elif plane == 'y':
            cont = ax.contourf(xi, zi, grid, levels=levels, cmap='jet', norm=norm)
        elif plane == 'z':
            cont = ax.contourf(xi, yi, grid, levels=levels, cmap='jet', norm=norm)
    else:
        if plane == 'x':
            cont = ax.contourf(yi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
        elif plane == 'y':
            cont = ax.contourf(xi, zi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)
        elif plane == 'z':
            cont = ax.contourf(xi, yi, grid, levels=500, cmap='jet', vmin=clim_min, vmax=clim_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加颜色条
    fig = ax.get_figure()
    cbar = fig.colorbar(cont, ax=ax, label="Displacement")

    # 设置颜色条刻度
    if scale == 'log':
        cbar.locator = LogLocator(numticks=10)
    else:
        cbar.locator = MaxNLocator(nbins=10)
    cbar.update_ticks()

    return cont, cbar


# ------------------- 交互式 APP -------------------

class InteractiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("交互式数据分析与绘图 APP")
        self.folder_path = None
        self.data_temperature = None
        self.data_e_field = None
        self.data_heat_flow = None
        self.data_displacement = None
        self.dragged_files = []  # 用于识别 S 参数文件

        # 新增：存放参数文件夹信息，键为 (signal, theta, phi, ratio, monitor_f)
        self.param_folders = {}
        # 用于存放各参数的唯一值列表
        self.signal_list = []
        self.theta_list = []
        self.phi_list = []
        self.ratio_list = []
        self.monitor_f_list = []

        self.s_param_mode = tk.StringVar(value="subplots")  # 默认显示子图

        # 设置现代化主题
        self.setup_modern_theme()

        # 右侧控制面板
        self.control_frame = tk.Frame(root, padx=5, pady=22, bg="#ffffff", width=300)  # 设置固定宽度
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)  # 不扩展，始终显示在右侧
        self.create_dragdrop_area()
        self.create_controls()
        self.create_parameter_controls()  # 新增：创建参数选择区

        # 左侧使用 Notebook 展示各图
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # 电场 Tab
        self.tab_e = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_e, text=" 电场 ")
        self.fig_e = plt.Figure(figsize=(6, 4), dpi=200)
        self.ax_e = self.fig_e.add_subplot(111)
        self.canvas_e = FigureCanvasTkAgg(self.fig_e, master=self.tab_e)
        self.canvas_e.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 热流场 Tab
        self.tab_heat = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_heat, text=" 热流场 ")
        self.fig_heat = plt.Figure(figsize=(6, 4), dpi=200)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=self.tab_heat)
        self.canvas_heat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 温度场 Tab
        self.tab_temp = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_temp, text=" 温度场 ")
        self.fig_temp = plt.Figure(figsize=(6, 4), dpi=200)
        self.ax_temp = self.fig_temp.add_subplot(111)
        self.canvas_temp = FigureCanvasTkAgg(self.fig_temp, master=self.tab_temp)
        self.canvas_temp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 力场Tab
        self.tab_force = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_force, text=" 力场 ")
        self.fig_force = plt.Figure(figsize=(6, 4), dpi=200)
        self.ax_force = self.fig_force.add_subplot(111)
        self.canvas_force = FigureCanvasTkAgg(self.fig_force, master=self.tab_force)
        self.canvas_force.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # S 参数 Tab
        self.tab_s = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_s, text=" S 参数曲线 ")
        self.fig_s = plt.Figure(figsize=(6, 4), dpi=200)
        self.ax_s = self.fig_s.add_subplot(111)
        self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=self.tab_s)
        self.canvas_s.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_modern_theme(self):
        """设置现代化主题"""
        self.root.configure(bg="#ffffff")
        style = ttk.Style()
        style.theme_use('xpnative')
        style.configure('TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', foreground='#333333', font=('Microsoft YaHei', 12))
        style.configure('TButton', background='#ffffff', foreground='#333333', font=('Microsoft YaHei', 12), padding=5,
                        borderwidth=1, relief=tk.FLAT)
        style.configure('TRadiobutton', background='#ffffff', foreground='#333333', font=('Microsoft YaHei', 12))
        style.configure('TNotebook', background='#ffffff')
        style.configure('TNotebook.Tab', background='#ffffff', foreground='#333333', padding=(5, 3),
                        font=('Microsoft YaHei', 12))
        style.map('TNotebook.Tab', background=[('selected', '#e6f2ff')],
                  font=[('selected', ('Microsoft YaHei', 15, 'bold'))],
                  padding=[('selected', (12, 6))])

    def create_dragdrop_area(self):
        self.drop_label = tk.Label(
            self.control_frame,
            text="拖放文件夹到此处",
            bg="#f0f0f0",
            fg="#000000",
            relief=tk.FLAT,
            borderwidth=1,
            width=20,
            height=5,
            font=('Microsoft YaHei', 16)
        )
        self.drop_label.pack(pady=23)
        self.drop_label.drop_target_register(tkdnd.DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

    def create_controls(self):
        # 定义自定义样式：浅蓝色背景
        style = ttk.Style()
        style.configure('LightBlue.TCombobox',
                        fieldbackground='#ffffff',
                        background='#e6f2ff',
                        font=('Microsoft YaHei', 12))
        """创建控制面板控件"""
        folder_button = tk.Button(
            self.control_frame,
            text="  选择数据文件夹  ",
            command=self.select_folder,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 15),
            relief=tk.FLAT,
            borderwidth=1,
            activebackground="#e6f2ff",
            activeforeground="#000000"
        )
        folder_button.pack(pady=20, fill=tk.X)
        folder_button.bind("<Enter>", lambda event: folder_button.config(bg="#e6f2ff"))
        folder_button.bind("<Leave>", lambda event: folder_button.config(bg="#ffffff"))

        self.frame_field = tk.LabelFrame(
            self.control_frame,
            text="场切面",
            bg="#ffffff",
            fg="#333333",
            font=('Microsoft YaHei', 12),
            bd=1,
            relief="ridge",
            labelanchor='n'
        )
        self.frame_field.pack(pady=15, fill=tk.X)
        tk.Label(
            self.frame_field,
            text="选择平面:",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=0, column=0, sticky=tk.W)
        self.plane_var = tk.StringVar(value='y')
        self.plane_combo = ttk.Combobox(
            self.frame_field,
            textvariable=self.plane_var,
            values=['x', 'y', 'z'],
            width=10,
            # font=('Microsoft YaHei', 12),
            state='readonly',
            style='LightBlue.TCombobox'
        )
        self.plane_combo.grid(row=0, column=1, padx=15, pady=3)
        tk.Label(
            self.frame_field,
            text="坐标值:",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=1, column=0, sticky=tk.W)
        self.coord_entry = tk.Entry(
            self.frame_field,
            width=10,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            insertbackground="#000000",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.coord_entry.grid(row=1, column=1, padx=15, pady=3)
        self.coord_entry.bind("<KeyRelease>", lambda event: self.update_current_tab())
        self.plane_combo.bind("<<ComboboxSelected>>", lambda event: self.update_current_tab())

        self.frame_common = tk.LabelFrame(
            self.control_frame,
            text="Colorbar",
            bg="#ffffff",
            fg="#333333",
            font=('Microsoft YaHei', 12),
            bd=1,
            relief='ridge',
            labelanchor='n'
        )
        self.frame_common.pack(pady=15, fill=tk.X)
        tk.Label(
            self.frame_common,
            text="颜色尺度：",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=0, column=0, sticky=tk.W)
        self.scale_var = tk.StringVar(value="linear")
        tk.Radiobutton(
            self.frame_common,
            text="Linear",
            variable=self.scale_var,
            value="linear",
            command=self.update_current_tab,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            selectcolor="#e6f2ff"
        ).grid(row=0, column=1)
        tk.Radiobutton(
            self.frame_common,
            text="Log",
            variable=self.scale_var,
            value="log",
            command=self.update_current_tab,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            selectcolor="#e6f2ff"
        ).grid(row=0, column=2)
        tk.Label(
            self.frame_common,
            text="最小值:",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=1, column=0, sticky=tk.W)
        self.min_entry = tk.Entry(
            self.frame_common,
            width=10,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            insertbackground="#000000",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.min_entry.grid(row=1, column=1, padx=5, pady=3)
        tk.Label(
            self.frame_common,
            text="最大值:",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=2, column=0, sticky=tk.W)
        self.max_entry = tk.Entry(
            self.frame_common,
            width=10,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            insertbackground="#000000",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.max_entry.grid(row=2, column=1, padx=5, pady=3)
        self.min_entry.bind("<KeyRelease>", lambda event: self.update_current_tab())
        self.max_entry.bind("<KeyRelease>", lambda event: self.update_current_tab())

        self.s_param_mode_frame = tk.LabelFrame(
            self.control_frame,
            text="S参数",
            bg="#ffffff",
            fg="#333333",
            font=('Microsoft YaHei', 12),
            bd=1,
            relief='ridge',
            labelanchor='n'
        )
        self.s_param_mode_frame.pack_forget()
        tk.Label(
            self.s_param_mode_frame,
            text="选择查看结果:",
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12)
        ).grid(row=0, column=0, sticky=tk.W)
        self.s_param_mode = tk.StringVar(value="subplots")
        tk.Radiobutton(
            self.s_param_mode_frame,
            text="2x2 子图",
            variable=self.s_param_mode,
            value="subplots",
            command=self.update_current_tab,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            selectcolor="#e6f2ff"
        ).grid(row=1, column=0)
        tk.Radiobutton(
            self.s_param_mode_frame,
            text="总图",
            variable=self.s_param_mode,
            value="main",
            command=self.update_current_tab,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 12),
            selectcolor="#e6f2ff"
        ).grid(row=1, column=1)

        self.save_button = tk.Button(
            self.control_frame,
            text="   保存当前图形   ",
            command=self.save_current_tab,
            bg="#ffffff",
            fg="#000000",
            font=('Microsoft YaHei', 15),
            relief=tk.FLAT,
            borderwidth=1,
            activebackground="#e6f2ff",
            activeforeground="#000000"
        )
        self.save_button.place(relx=0.5, rely=1.0, anchor="s", y=-5)
        self.save_button.bind("<Enter>", lambda event: self.save_button.config(bg="#e6f2ff"))
        self.save_button.bind("<Leave>", lambda event: self.save_button.config(bg="#ffffff"))

    def create_parameter_controls(self):
        # 定义自定义样式：浅蓝色背景
        style = ttk.Style()
        style.configure('LightBlue.TCombobox',
                        fieldbackground='#ffffff',
                        background='#e6f2ff',
                        font=('Microsoft YaHei', 12))

        self.frame_param = tk.LabelFrame(
            self.control_frame,
            text="参数选择",
            bg="#ffffff",
            fg="#333333",
            font=('Microsoft YaHei', 12),
            bd=1,
            relief="ridge",
            labelanchor='n'
        )
        self.frame_param.pack(pady=15, fill=tk.X)

        # 每个参数使用与“选择平面”相似的布局方式
        tk.Label(self.frame_param, text="signal:", bg="#ffffff", fg="#000000", font=('Microsoft YaHei', 12)) \
            .grid(row=0, column=0, sticky=tk.W)
        self.signal_var = tk.StringVar()
        self.signal_combo = ttk.Combobox(self.frame_param, textvariable=self.signal_var,
                                         values=[], width=10, state='readonly', style='LightBlue.TCombobox')
        self.signal_combo.grid(row=0, column=1, padx=5, pady=3)
        self.signal_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_selection())

        tk.Label(self.frame_param, text="theta:", bg="#ffffff", fg="#000000", font=('Microsoft YaHei', 12)) \
            .grid(row=1, column=0, sticky=tk.W)
        self.theta_var = tk.StringVar()
        self.theta_combo = ttk.Combobox(self.frame_param, textvariable=self.theta_var,
                                        values=[], width=10, state='readonly', style='LightBlue.TCombobox')
        self.theta_combo.grid(row=1, column=1, padx=5, pady=3)
        self.theta_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_selection())

        tk.Label(self.frame_param, text="phi:", bg="#ffffff", fg="#000000", font=('Microsoft YaHei', 12)) \
            .grid(row=2, column=0, sticky=tk.W)
        self.phi_var = tk.StringVar()
        self.phi_combo = ttk.Combobox(self.frame_param, textvariable=self.phi_var,
                                      values=[], width=10, state='readonly', style='LightBlue.TCombobox')
        self.phi_combo.grid(row=2, column=1, padx=5, pady=3)
        self.phi_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_selection())

        tk.Label(self.frame_param, text="ratio:", bg="#ffffff", fg="#000000", font=('Microsoft YaHei', 12)) \
            .grid(row=3, column=0, sticky=tk.W)
        self.ratio_var = tk.StringVar()
        self.ratio_combo = ttk.Combobox(self.frame_param, textvariable=self.ratio_var,
                                        values=[], width=10, state='readonly', style='LightBlue.TCombobox')
        self.ratio_combo.grid(row=3, column=1, padx=5, pady=3)
        self.ratio_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_selection())

        tk.Label(self.frame_param, text="frequency:", bg="#ffffff", fg="#000000", font=('Microsoft YaHei', 12)) \
            .grid(row=4, column=0, sticky=tk.W)
        self.monitor_f_var = tk.StringVar()
        self.monitor_f_combo = ttk.Combobox(self.frame_param, textvariable=self.monitor_f_var,
                                            values=[], width=10, state='readonly', style='LightBlue.TCombobox')
        self.monitor_f_combo.grid(row=4, column=1, padx=5, pady=3)
        self.monitor_f_combo.bind("<<ComboboxSelected>>", lambda e: self.on_param_selection())

    def scan_parameter_folders(self):
        """扫描同目录下满足参数格式的文件夹，更新各参数选项，同时如果当前 folder_path 匹配则自动选中对应参数"""
        if not self.folder_path:
            return
        parent_dir = os.path.dirname(self.folder_path)
        pattern = re.compile(r"signal=([^,]+),\s*theta=([^,]+),\s*phi=([^,]+),\s*ratio=([^,]+),\s*monitor_f=([^,]+)")
        self.param_folders.clear()
        self.signal_list.clear()
        self.theta_list.clear()
        self.phi_list.clear()
        self.ratio_list.clear()
        self.monitor_f_list.clear()

        for name in os.listdir(parent_dir):
            folder_fullpath = os.path.join(parent_dir, name)
            if os.path.isdir(folder_fullpath):
                m = pattern.match(name)
                if m:
                    signal_val, theta_val, phi_val, ratio_val, monitor_f_val = [v.strip() for v in m.groups()]
                    key = (signal_val, theta_val, phi_val, ratio_val, monitor_f_val)
                    self.param_folders[key] = folder_fullpath
                    if signal_val not in self.signal_list:
                        self.signal_list.append(signal_val)
                    if theta_val not in self.theta_list:
                        self.theta_list.append(theta_val)
                    if phi_val not in self.phi_list:
                        self.phi_list.append(phi_val)
                    if ratio_val not in self.ratio_list:
                        self.ratio_list.append(ratio_val)
                    if monitor_f_val not in self.monitor_f_list:
                        self.monitor_f_list.append(monitor_f_val)
        # 排序（可根据需要调整排序规则）
        self.signal_list.sort()
        self.theta_list.sort(key=lambda x: float(x))
        self.phi_list.sort(key=lambda x: float(x))
        self.monitor_f_list.sort(key=lambda x: float(x))

        # 处理 ratio 的排序：数字排在前面，字符串排在后面
        def sort_ratio_key(ratio_val):
            try:
                # 如果可以转换为浮点数，则返回浮点数
                return float(ratio_val)
            except ValueError:
                # 如果是字符串，则返回一个非常大的值，确保字符串排在数字后面
                return float('inf')

        self.ratio_list.sort(key=sort_ratio_key)

        # 更新各 Combobox 的选项
        self.signal_combo['values'] = self.signal_list
        self.theta_combo['values'] = self.theta_list
        self.phi_combo['values'] = self.phi_list
        self.ratio_combo['values'] = self.ratio_list
        self.monitor_f_combo['values'] = self.monitor_f_list

        # 如果当前数据文件夹属于某个参数组合，自动选中对应参数
        cur_abs = os.path.abspath(self.folder_path)
        for key, folder in self.param_folders.items():
            if os.path.abspath(folder) == cur_abs:
                self.signal_var.set(key[0])
                self.theta_var.set(key[1])
                self.phi_var.set(key[2])
                self.ratio_var.set(key[3])
                self.monitor_f_var.set(key[4])
                break

    def on_param_selection(self):
        """
        当用户选择参数后，如果所有参数都有值，则加载对应文件夹的数据，
        并只更新当前 Tab 的图，而不是更新所有 Tab。
        """
        signal = self.signal_var.get()
        theta = self.theta_var.get()
        phi = self.phi_var.get()
        ratio = self.ratio_var.get()
        monitor_f = self.monitor_f_var.get()
        if signal and theta and phi and ratio and monitor_f:
            key = (signal, theta, phi, ratio, monitor_f)
            if key in self.param_folders:
                target_folder = self.param_folders[key]
                self.folder_path = target_folder
                self.root.title(f"交互式数据分析与绘图 APP - {os.path.basename(target_folder)}")
                files = os.listdir(target_folder)
                if 'Temperature.csv' in files:
                    try:
                        self.data_temperature = read_data(os.path.join(target_folder, 'Temperature.csv'))
                    except Exception as e:
                        messagebox.showerror("错误", f"读取 Temperature.csv 失败：{e}")
                else:
                    messagebox.showerror("错误", f"{target_folder} 中没有 Temperature.csv 文件")
                    return
                self.dragged_files = [f for f in files if f.endswith('.txt') and 'S' in f]
                e_list = [f for f in files if f.startswith('e-field') and f.endswith('.csv')]
                if e_list:
                    self.data_e_field = read_e_field_data(os.path.join(target_folder, e_list[0]))
                    if self.data_e_field is not None:
                        self.data_e_field = calculate_e_field(self.data_e_field)
                else:
                    self.data_e_field = None
                if 'Heat_Flow.csv' in files:
                    self.data_heat_flow = read_heat_flow_data(os.path.join(target_folder, 'Heat_Flow.csv'))
                    if self.data_heat_flow is not None:
                        self.data_heat_flow = calculate_heat_flow(self.data_heat_flow)
                else:
                    self.data_heat_flow = None
                # 力场数据读取
                if 'Displacement.csv' in files:
                    self.data_displacement = read_displacement_data(os.path.join(target_folder, 'Displacement.csv'))
                    if self.data_displacement is not None:
                        self.data_displacement = calculate_displacement(self.data_displacement)
                else:
                    self.data_displacement = None
                # 只更新当前激活的 Tab 图形，而不是所有 Tab
                self.update_current_tab()
            else:
                messagebox.showerror("错误", "未找到匹配参数的文件夹")

    def on_drop(self, event):
        folder = event.data.strip('{}')
        if os.path.isdir(folder):
            self.folder_path = folder
            self.root.title(f"交互式数据分析与绘图 APP - {os.path.basename(folder)}")
            files = os.listdir(folder)
            if 'Temperature.csv' in files:
                try:
                    self.data_temperature = read_data(os.path.join(folder, 'Temperature.csv'))
                except Exception as e:
                    messagebox.showerror("错误", f"读取 Temperature.csv 失败：{e}")
            else:
                messagebox.showerror("错误", "拖放的文件夹中没有 Temperature.csv 文件")
                return
            self.dragged_files = [f for f in files if f.endswith('.txt') and 'S' in f]
            e_list = [f for f in files if f.startswith('e-field') and f.endswith('.csv')]
            if e_list:
                self.data_e_field = read_e_field_data(os.path.join(folder, e_list[0]))
                if self.data_e_field is not None:
                    self.data_e_field = calculate_e_field(self.data_e_field)
            else:
                self.data_e_field = None
            if 'Heat_Flow.csv' in files:
                self.data_heat_flow = read_heat_flow_data(os.path.join(folder, 'Heat_Flow.csv'))
                if self.data_heat_flow is not None:
                    self.data_heat_flow = calculate_heat_flow(self.data_heat_flow)
            else:
                self.data_heat_flow = None
            # 力场数据读取
            if 'Displacement.csv' in files:
                self.data_displacement = read_displacement_data(os.path.join(folder, 'Displacement.csv'))
                if self.data_displacement is not None:
                    self.data_displacement = calculate_displacement(self.data_displacement)
            else:
                self.data_displacement = None
            self.update_all_tabs()
            # 扫描参数文件夹并自动选中对应的参数
            self.scan_parameter_folders()
        else:
            messagebox.showerror("错误", "拖放的不是有效的文件夹路径")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.root.title(f"交互式数据分析与绘图 APP - {os.path.basename(folder)}")
            files = os.listdir(folder)
            if 'Temperature.csv' in files:
                try:
                    self.data_temperature = read_data(os.path.join(folder, 'Temperature.csv'))
                except Exception as e:
                    messagebox.showerror("错误", f"读取 Temperature.csv 失败：{e}")
            else:
                messagebox.showerror("错误", "文件夹中没有 Temperature.csv 文件")
                self.folder_path = None
                self.root.title("交互式数据分析与绘图 APP")
                return
            self.dragged_files = [f for f in files if f.endswith('.txt') and 'S' in f]
            e_list = [f for f in files if f.startswith('e-field') and f.endswith('.csv')]
            if e_list:
                self.data_e_field = read_e_field_data(os.path.join(folder, e_list[0]))
                if self.data_e_field is not None:
                    self.data_e_field = calculate_e_field(self.data_e_field)
            else:
                self.data_e_field = None
            if 'Heat_Flow.csv' in files:
                self.data_heat_flow = read_heat_flow_data(os.path.join(folder, 'Heat_Flow.csv'))
                if self.data_heat_flow is not None:
                    self.data_heat_flow = calculate_heat_flow(self.data_heat_flow)
            else:
                self.data_heat_flow = None
            # 力场数据读取
            if 'Displacement.csv' in files:
                self.data_displacement = read_displacement_data(os.path.join(folder, 'Displacement.csv'))
                if self.data_displacement is not None:
                    self.data_displacement = calculate_displacement(self.data_displacement)
            else:
                self.data_displacement = None
            self.update_all_tabs()
            # 扫描参数文件夹并自动选中当前数据对应参数（如果存在）
            self.scan_parameter_folders()

    def get_range(self):
        try:
            clim_min = float(self.min_entry.get())
        except:
            clim_min = None
        try:
            clim_max = float(self.max_entry.get())
        except:
            clim_max = None
        return clim_min, clim_max

    def on_tab_changed(self, event):
        self.update_current_tab()
        self.save_button.pack_forget()
        self.save_button.place(relx=0.5, rely=1.0, anchor="s", y=-5)

    def update_current_tab(self):
        tab_index = self.notebook.index(self.notebook.select())
        self.frame_common.pack_forget()
        self.frame_field.pack_forget()
        self.s_param_mode_frame.pack_forget()

        if tab_index == 0:  # 电场 Tab
            self.frame_field.pack(pady=15, fill=tk.X)
            self.frame_common.pack(pady=15, fill=tk.X)
            self.update_e_field_tab()
        elif tab_index == 1:  # 热流场 Tab
            self.frame_field.pack(pady=15, fill=tk.X)
            self.frame_common.pack(pady=15, fill=tk.X)
            self.update_heat_flow_tab()
        elif tab_index == 2:  # 温度场 Tab
            self.frame_field.pack(pady=15, fill=tk.X)
            self.frame_common.pack(pady=15, fill=tk.X)
            self.update_temperature_tab()
        elif tab_index == 3:  # 力场Tab
            self.frame_field.pack(pady=15, fill=tk.X)
            self.frame_common.pack(pady=15, fill=tk.X)
            self.update_force_field_tab()
        elif tab_index == 4:  # S 参数曲线 Tab
            self.s_param_mode_frame.pack(pady=15, fill=tk.X)
            self.update_s_parameter_tab()

    def update_all_tabs(self):
        self.update_temperature_tab()
        self.update_s_parameter_tab()
        self.update_e_field_tab()
        self.update_heat_flow_tab()
        self.update_force_field_tab()

    def update_temperature_tab(self):
        self.fig_temp.clf()
        self.ax_temp = self.fig_temp.add_subplot(111)
        if self.data_temperature is None:
            self.ax_temp.text(0.5, 0.5, "无温度数据", ha="center", va="center")
        else:
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            clim_min, clim_max = self.get_range()
            scale = self.scale_var.get()
            result = plot_temperature_field(self.ax_temp, self.data_temperature, plane, coord_value, clim_min, clim_max,
                                            scale)
            if result is None:
                self.ax_temp.text(0.5, 0.5, "无法绘制温度场，请检查输入参数", ha="center", va="center")
        self.fig_temp.tight_layout()
        self.canvas_temp.draw()

    def update_s_parameter_tab(self):
        for widget in self.tab_s.winfo_children():
            widget.destroy()
        if self.folder_path is None:
            self.ax_s = self.fig_s.add_subplot(111)
            self.ax_s.text(0.5, 0.5, "请先选择数据文件夹", ha="center", va="center")
            self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=self.tab_s)
            self.canvas_s.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvas_s.draw()
            return
        view_mode = self.s_param_mode.get()
        if view_mode == "subplots":
            plot_s_parameter_subplots(self.tab_s, self.folder_path, self.dragged_files)
        else:
            self.fig_s = plt.Figure(figsize=(6, 4), dpi=150)
            self.ax_s = self.fig_s.add_subplot(111)
            plot_s_parameter_main(self.ax_s, self.folder_path, self.dragged_files)
            self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=self.tab_s)
            self.canvas_s.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.fig_s.tight_layout()
            self.canvas_s.draw()

    def update_force_field_tab(self):
        self.fig_force.clf()
        self.ax_force = self.fig_force.add_subplot(111)
        if self.data_displacement is None:
            self.ax_force.text(0.5, 0.5, "无力场数据", ha="center", va="center")
        else:
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            clim_min, clim_max = self.get_range()
            scale = self.scale_var.get()
            result = plot_force_field_distribution(self.ax_force, self.data_displacement,
                                                   plane, coord_value, clim_min, clim_max, scale)
            if result is None:
                self.ax_force.text(0.5, 0.5, "无法绘制力场，请检查输入参数", ha="center", va="center")
        self.fig_force.tight_layout()
        self.canvas_force.draw()

    def update_e_field_tab(self):
        self.fig_e.clf()
        self.ax_e = self.fig_e.add_subplot(111)
        if self.data_e_field is None:
            self.ax_e.text(0.5, 0.5, "无电场数据", ha="center", va="center")
        else:
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            clim_min, clim_max = self.get_range()
            scale = self.scale_var.get()
            result = plot_e_field_distribution(self.ax_e, self.data_e_field, plane, coord_value, clim_min, clim_max,
                                               scale)
            if result is None:
                self.ax_e.text(0.5, 0.5, "无法绘制电场，请检查输入参数", ha="center", va="center")
        self.fig_e.tight_layout()
        self.canvas_e.draw()

    def update_heat_flow_tab(self):
        self.fig_heat.clf()
        self.ax_heat = self.fig_heat.add_subplot(111)
        if self.data_heat_flow is None:
            self.ax_heat.text(0.5, 0.5, "无热流场数据", ha="center", va="center")
        else:
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            clim_min, clim_max = self.get_range()
            scale = self.scale_var.get()
            result = plot_heat_flow_distribution(self.ax_heat, self.data_heat_flow, plane, coord_value, clim_min,
                                                 clim_max, scale)
            if result is None:
                self.ax_heat.text(0.5, 0.5, "无法绘制热流场，请检查输入参数", ha="center", va="center")
        self.fig_heat.tight_layout()
        self.canvas_heat.draw()

    def save_current_tab(self):
        tab_index = self.notebook.index(self.notebook.select())
        save_folder = self.folder_path if self.folder_path else filedialog.askdirectory(title="选择保存图片的文件夹")
        if not save_folder:
            return

        if tab_index == 2:  # 温度场 Tab
            if self.data_temperature is None:
                messagebox.showerror("错误", "无温度数据")
                return
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            if plane == 'x':
                filename = f"Temperature_YOZplane_x={coord_value:.2f}.png"
            elif plane == 'y':
                filename = f"Temperature_XOZplane_y={coord_value:.2f}.png"
            elif plane == 'z':
                filename = f"Temperature_XOYplane_z={coord_value:.2f}.png"
            else:
                filename = "Temperature.png"
            filepath = os.path.join(save_folder, filename)
            self.fig_temp.savefig(filepath, dpi=400, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"温度场图已保存到: {filepath}")

        elif tab_index == 4:  # S 参数曲线 Tab
            view_mode = self.s_param_mode.get()
            if view_mode == "subplots":
                filename = "Subplots_S_Parameters.png"
                fig, _ = self.create_s_parameter_subplots_figure()
                filepath = os.path.join(save_folder, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                messagebox.showinfo("保存成功", f"S 参数子图已保存到: {filepath}")
            else:
                filename = "All_S_Parameters.png"
                filepath = os.path.join(save_folder, filename)
                self.fig_s.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("保存成功", f"S 参数主图已保存到: {filepath}")

        elif tab_index == 0:  # 电场 Tab
            if self.data_e_field is None:
                messagebox.showerror("错误", "无电场数据")
                return
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            if plane == 'x':
                filename = f"Electric_Field_YOZplane_x={coord_value:.2f}.png"
            elif plane == 'y':
                filename = f"Electric_Field_XOZplane_y={coord_value:.2f}.png"
            elif plane == 'z':
                filename = f"Electric_Field_XOYplane_z={coord_value:.2f}.png"
            else:
                filename = "Electric_Field.png"
            filepath = os.path.join(save_folder, filename)
            self.fig_e.savefig(filepath, dpi=600, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"电场图已保存到: {filepath}")

        elif tab_index == 1:  # 热流场 Tab
            if self.data_heat_flow is None:
                messagebox.showerror("错误", "无热流场数据")
                return
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            if plane == 'x':
                filename = f"Heat_Flow_YOZplane_x={coord_value:.2f}.png"
            elif plane == 'y':
                filename = f"Heat_Flow_XOZplane_y={coord_value:.2f}.png"
            elif plane == 'z':
                filename = f"Heat_Flow_XOYplane_z={coord_value:.2f}.png"
            else:
                filename = "Heat_Flow.png"
            filepath = os.path.join(save_folder, filename)
            self.fig_heat.savefig(filepath, dpi=500, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"热流场图已保存到: {filepath}")

        elif tab_index == 3:  # 力场Tab
            if self.data_displacement is None:
                messagebox.showerror("错误", "无力场数据")
                return
            plane = self.plane_var.get()
            try:
                coord_value = float(self.coord_entry.get())
            except:
                coord_value = 0.0
            if plane == 'x':
                filename = f"Displacement_YOZplane_x={coord_value:.2f}.png"
            elif plane == 'y':
                filename = f"Displacement_XOZplane_y={coord_value:.2f}.png"
            elif plane == 'z':
                filename = f"Displacement_XOYplane_z={coord_value:.2f}.png"
            else:
                filename = "Displacement.png"
            filepath = os.path.join(save_folder, filename)
            self.fig_force.savefig(filepath, dpi=500, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"力场图已保存到: {filepath}")

    def create_s_parameter_subplots_figure(self):
        file_groups = {
            'S1,1': ['S1,1.txt', 'S1,1_ori.txt'],
            'S1,2': ['S1,2.txt', 'S1,2_ori.txt'],
            'S2,1': ['S2,1.txt', 'S2,1_ori.txt'],
            'S2,2': ['S2,2.txt', 'S2,2_ori.txt']
        }
        min_freq = float('inf')
        max_freq = -float('inf')

        for file in self.dragged_files:
            if file.endswith('.txt') and 'S' in file:
                file_path = os.path.join(self.folder_path, file)
                try:
                    data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                    min_freq = min(min_freq, data['Frequency'].min())
                    max_freq = max(max_freq, data['Frequency'].max())
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('S Parameter Curves', fontsize=16)

        for i, (group_name, group_files) in enumerate(file_groups.items()):
            row = i // 2
            col = i % 2
            ax = axs[row, col]
            ax.set_title(f'{group_name} Curves', fontsize=12)
            ax.set_xlabel('Frequency / GHz', fontsize=10)
            ax.set_ylabel('S Value / dB', fontsize=10)
            ax.set_xlim(min_freq, max_freq)
            ax.grid(True, linestyle='--', alpha=0.7)

            for file in group_files:
                file_path = None
                if file in self.dragged_files:
                    file_path = os.path.join(self.folder_path, file)
                else:
                    continue

                try:
                    data = pd.read_csv(file_path, sep='\s+', skiprows=2, names=['Frequency', 'S_value'])
                    if len(data) > 200:
                        indices = np.linspace(0, len(data) - 1, 200, dtype=int)
                        data = data.iloc[indices]
                    legend_label = file.split('.')[0]
                    linestyle = '--' if '_ori' in legend_label else '-'
                    ax.plot(data['Frequency'], data['S_value'], label=legend_label, linestyle=linestyle)
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    continue

            ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return fig, axs


if __name__ == "__main__":
    root = tkdnd.TkinterDnD.Tk()
    root.state("zoomed")  # 窗口最大化
    app = InteractiveApp(root)
    root.mainloop()
