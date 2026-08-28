# -*- coding: utf-8 -*-
"""
2023年高教社杯全国大学生数学建模竞赛 A题 问题1 —— 定日镜场的优化设计

计算内容：
  1. 每月21日（12个月）当地时间 9:00 / 10:30 / 12:00 / 13:30 / 15:00 五个时点的
     平均光学效率、平均余弦效率、平均阴影遮挡效率、平均截断效率、
     单位面积镜面平均输出热功率（对应题目表 1）；
  2. 上述指标的年平均值与年平均输出热功率（对应题目表 2）。

主要修正（相对早期版本）：
  - 吸收塔高度按题意取 80 m（集热器中心离地高度），而不是 84 m；
  - 太阳赤纬按“以春分 3 月 21 日为第 0 天起算”的天数 D 计算，
    每月 21 日对应 D = -59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275；
  - 太阳方向矢量使用题目附录的太阳方位角 γ（分上午/下午两个象限），
    不再用纬度 φ 近似；
  - 阴影/遮挡判定要求射线交点位于射线前进方向（t > 0），避免把
    位于反方向的相邻镜子误判为遮挡；
  - 截断效率不再取常数 0.8，而是按太阳光锥半顶角 4.65 mrad 计算反射光斑
    与集热器（直径 7 m）的相交面积占比，且只统计未被阴影遮挡的网格点。
"""

import time

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# 基础参数（题意给定）
# ----------------------------------------------------------------------
H_MIRROR = 4.0        # 定日镜安装高度（镜面中心离地高度），m
W_MIRROR = 6.0        # 镜面宽度（左右两边之间距离），m
H_MIRROR_SIZE = 6.0   # 镜面高度（上下两边之间距离），m
H_HEAT = 80.0         # 吸收塔高度（集热器中心离地高度），m
R_REC = 3.5           # 集热器半径（圆柱直径 7 m），m
THETA_SUN = 4.65e-3   # 太阳光锥半顶角，rad
PHI = 39.4 * np.pi / 180.0   # 当地纬度（北纬为正）
H_SEA = 3.0           # 海拔高度，km
G0 = 1.366            # 太阳常数，kW/m^2
REF_EFF = 0.92        # 镜面反射率
GROUND_VEC = np.array([0.0, 0.0, -1.0])   # 地面法向（向下）
N_NEAREST = 6         # 阴影/遮挡只考察最近 6 面相邻定日镜
N_W = 20              # 镜面宽度方向网格数
N_D = 20              # 镜面高度方向网格数

# 每月 21 日以春分（3 月 21 日，一年中第 80 天）为第 0 天起算的天数 D
DAYS_D = [-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
MONTHS = [f'{m}月21日' for m in range(1, 13)]
STS = [9.0, 10.5, 12.0, 13.5, 15.0]       # 当地时间，h


# ----------------------------------------------------------------------
# 太阳位置与辐照度
# ----------------------------------------------------------------------
def calc_basic_data(day_D, st, phi):
    """太阳时角 ω、赤纬角 δ、高度角 α、方位角 γ（题目附录公式）"""
    omega = np.pi / 12.0 * (st - 12.0)
    delta = np.arcsin(np.sin(2.0 * np.pi * day_D / 365.0) *
                      np.sin(2.0 * np.pi * 23.45 / 360.0))
    sin_alpha = (np.cos(delta) * np.cos(phi) * np.cos(omega) +
                 np.sin(delta) * np.sin(phi))
    alpha = np.arcsin(np.clip(sin_alpha, -1.0, 1.0))
    cos_gamma = ((np.sin(delta) - np.sin(alpha) * np.sin(phi)) /
                 (np.cos(alpha) * np.cos(phi)))
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    if omega > 0.0:                        # 下午太阳偏西
        gamma = 2.0 * np.pi - gamma
    return omega, delta, alpha, gamma


def sun_vector(alpha, gamma):
    """太阳方向单位矢量（从定日镜指向太阳），x 东、y 北、z 上"""
    return np.array([np.cos(alpha) * np.sin(gamma),
                     np.cos(alpha) * np.cos(gamma),
                     np.sin(alpha)])


def calc_DNI(h_sea, alpha, G0):
    """法向直接辐射辐照度，kW/m^2（题目附录公式）"""
    a = 0.4237 - 0.00821 * (6.0 - h_sea) ** 2
    b = 0.5055 + 0.00595 * (6.5 - h_sea) ** 2
    c = 0.2711 + 0.01858 * (2.5 - h_sea) ** 2
    return G0 * (a + b * np.exp(-c / np.sin(alpha)))


# ----------------------------------------------------------------------
# 镜面几何
# ----------------------------------------------------------------------
def calc_mirror_geometry(x, y, u):
    """镜面中心 base、反射方向 v、法向 n 及镜面局部坐标轴 axis_x / axis_y"""
    base = np.column_stack([x, y, np.full_like(x, H_MIRROR)])
    tower = np.array([0.0, 0.0, H_HEAT])
    v = tower - base
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # 法向取太阳方向与反射方向的角平分方向，满足反射定律
    n = u + v
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    axis_x = np.cross(n, GROUND_VEC)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
    axis_y = np.cross(n, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
    return base, v, n, axis_x, axis_y


# ----------------------------------------------------------------------
# 阴影遮挡效率
# ----------------------------------------------------------------------
def calc_block_eff(base, v, n, axis_x, axis_y, u, neighbors):
    """
    阴影遮挡效率：把每面镜面划分为 N_W x N_D 个网格点，
    对每个网格点分别做“指向太阳”（+u，入射阴影）与“指向集热器”（+v，反射遮挡）
    的射线，判断射线与最近 6 面相邻镜面的交点是否落在该镜面矩形内。
    交点必须位于射线前进方向（t > 0）。
    """
    n_mirror = len(base)
    mesh_p, mesh_q = np.meshgrid(np.arange(N_W), np.arange(N_D))
    a_off = (mesh_p - (N_W - 1) / 2.0) * (W_MIRROR / N_W)    # (N_D, N_W)
    b_off = (mesh_q - (N_D - 1) / 2.0) * (H_MIRROR_SIZE / N_D)
    offsets = (a_off[None, :, :, None] * axis_x[:, None, None, :] +
               b_off[None, :, :, None] * axis_y[:, None, None, :])
    little = base[:, None, None, :] + offsets                # (N, N_D, N_W, 3)

    block_num = np.zeros(n_mirror, dtype=int)
    blocked_grid = np.zeros((n_mirror, N_D, N_W), dtype=bool)

    for i in range(n_mirror):
        ind = neighbors[i]
        diff = base[ind, None, None, :] - little[i]          # (K, N_D, N_W, 3)
        n_j = n[ind][:, None, None, :]                       # (K, 1, 1, 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_out = np.sum(n_j * diff, axis=-1) / np.sum(n_j * u, axis=-1)
            t_in = np.sum(n_j * diff, axis=-1) / np.sum(n_j * v[i], axis=-1)

        loc_out = little[i] + t_out[..., None] * u           # 朝太阳的射线交点
        loc_in = little[i] + t_in[..., None] * v[i]          # 朝集热器的射线交点

        ax_j = axis_x[ind][:, None, None, :]
        ay_j = axis_y[ind][:, None, None, :]
        to_out = loc_out - base[ind, None, None, :]
        to_in = loc_in - base[ind, None, None, :]
        x_out = np.sum(to_out * ax_j, axis=-1)
        y_out = np.sum(to_out * ay_j, axis=-1)
        x_in = np.sum(to_in * ax_j, axis=-1)
        y_in = np.sum(to_in * ay_j, axis=-1)

        hit_out = ((np.abs(x_out) <= W_MIRROR / 2.0) &
                   (np.abs(y_out) <= H_MIRROR_SIZE / 2.0) & (t_out > 1e-9))
        hit_in = ((np.abs(x_in) <= W_MIRROR / 2.0) &
                  (np.abs(y_in) <= H_MIRROR_SIZE / 2.0) & (t_in > 1e-9))
        total_hit = (hit_out | hit_in).any(axis=0)           # (N_D, N_W)
        block_num[i] = total_hit.sum()
        blocked_grid[i] = total_hit

    block_eff = 1.0 - block_num / (N_W * N_D)
    return block_eff, blocked_grid


# ----------------------------------------------------------------------
# 截断效率
# ----------------------------------------------------------------------
def circle_overlap(r1, r2, d):
    """半径 r1、r2 两圆圆心距为 d 时的相交面积"""
    d = np.asarray(d, dtype=float)
    r1 = (np.full(d.shape, r1, dtype=float) if np.ndim(r1) == 0
          else np.broadcast_to(r1, d.shape).astype(float))
    r2 = (np.full(d.shape, r2, dtype=float) if np.ndim(r2) == 0
          else np.broadcast_to(r2, d.shape).astype(float))
    out = np.zeros_like(d)
    inside = d <= np.abs(r1 - r2)
    out[inside] = np.pi * np.minimum(r1, r2)[inside] ** 2
    inter = (d > np.abs(r1 - r2)) & (d < r1 + r2)
    a, b, c = r1[inter], r2[inter], d[inter]
    x = (c ** 2 + a ** 2 - b ** 2) / (2.0 * c)
    y = (c ** 2 + b ** 2 - a ** 2) / (2.0 * c)
    h = np.sqrt(np.clip(a ** 2 - x ** 2, 0.0, None))
    out[inter] = (a ** 2 * np.arccos(np.clip(x / a, -1.0, 1.0)) +
                  b ** 2 * np.arccos(np.clip(y / b, -1.0, 1.0)) -
                  c * h)
    return out


def calc_trunc_eff(base, v, axis_x, axis_y, d_HR, blocked_grid):
    """
    截断效率：太阳光为半顶角 THETA_SUN 的锥形光束，反射后在垂直于 v 的
    集热器中心平面上形成半径为 rho = d_HR*tan(THETA_SUN) 的光斑；
    计算各网格点光斑与集热器圆盘（半径 R_REC）的相交面积占比，
    并只统计未被阴影遮挡的网格点。
    """
    mesh_p, mesh_q = np.meshgrid(np.arange(N_W), np.arange(N_D))
    a_off = (mesh_p - (N_W - 1) / 2.0) * (W_MIRROR / N_W)
    b_off = (mesh_q - (N_D - 1) / 2.0) * (H_MIRROR_SIZE / N_D)
    v_ax = np.sum(v * axis_x, axis=1)       # v 在镜面宽度方向上的分量
    v_ay = np.sum(v * axis_y, axis=1)       # v 在镜面高度方向上的分量
    # 网格点相对镜面中心的偏移在垂直于 v 的平面上的投影模长 |q|
    q2 = (a_off[None, :, :] ** 2 + b_off[None, :, :] ** 2 -
          (a_off[None, :, :] * v_ax[:, None, None] +
           b_off[None, :, :] * v_ay[:, None, None]) ** 2)
    q = np.sqrt(np.clip(q2, 0.0, None))     # (N, N_D, N_W)
    rho = d_HR[:, None, None] * np.tan(THETA_SUN)
    overlap = circle_overlap(R_REC, rho, q)
    use = ~blocked_grid
    numer = np.sum(overlap * use, axis=(1, 2))
    denom = np.sum(np.pi * rho ** 2 * use, axis=(1, 2))
    trunc_eff = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return trunc_eff


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f'正在运行，镜面网格{N_W}*{N_D}')

    data_path = Path(__file__).resolve().with_name('附件.xlsx')
    data = pd.read_excel(data_path, skiprows=1, header=None)
    x = data.iloc[:, 0].values.astype(float)
    y = data.iloc[:, 1].values.astype(float)
    n_mirror = len(x)

    # 最近 N_NEAREST 面相邻定日镜（按底座中心距离）
    dist2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    np.fill_diagonal(dist2, np.inf)
    neighbors = np.argpartition(dist2, N_NEAREST, axis=1)[:, :N_NEAREST]

    # 镜面中心到集热器中心的距离及大气透射率（附录公式）
    d_HR = np.sqrt(x ** 2 + y ** 2 + (H_HEAT - H_MIRROR) ** 2)
    air_eff = 0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR ** 2

    A_mirror = W_MIRROR * H_MIRROR_SIZE
    n_days, n_sts = len(DAYS_D), len(STS)

    # 表1：每月 21 日（光学/余弦/阴影遮挡/截断/单位面积输出热功率）
    monthly = np.zeros((n_days, 5))
    # 表2：年平均（前四项 + 单位面积输出热功率）
    annual = np.zeros(5)
    annual_power = 0.0        # 年平均输出热功率，kW

    for a, day_D in enumerate(DAYS_D):
        for b, st in enumerate(STS):
            _, _, alpha, gamma = calc_basic_data(day_D, st, PHI)
            u = sun_vector(alpha, gamma)
            DNI = calc_DNI(H_SEA, alpha, G0)
            base, v, n, axis_x, axis_y = calc_mirror_geometry(x, y, u)

            cos_eff = np.sum(n * u, axis=1)                   # 余弦效率
            block_eff, blocked_grid = calc_block_eff(base, v, n, axis_x, axis_y,
                                                     u, neighbors)
            trunc_eff = calc_trunc_eff(base, v, axis_x, axis_y, d_HR, blocked_grid)

            eff = block_eff * cos_eff * air_eff * trunc_eff * REF_EFF
            E_field = DNI * np.sum(A_mirror * eff)            # kW
            P_area = E_field / (n_mirror * A_mirror)          # kW/m^2

            row = np.array([np.mean(eff), np.mean(cos_eff), np.mean(block_eff),
                            np.mean(trunc_eff), P_area])
            monthly[a] += row
            annual += row
            annual_power += E_field

    monthly /= n_sts
    annual /= (n_days * n_sts)
    annual_power /= (n_days * n_sts)

    print('=' * 82)
    print('表1  问题1 每月21日平均光学效率及输出功率')
    print('-' * 82)
    print('%-9s %13s %13s %16s %13s %17s' %
          ('日期', '平均光学效率', '平均余弦效率', '平均阴影遮挡效率',
           '平均截断效率', '单位面积输出热功率(kW/m^2)'))
    for a in range(n_days):
        print('%-9s %13.6f %13.6f %16.6f %13.6f %17.4f' %
              (MONTHS[a], monthly[a, 0], monthly[a, 1], monthly[a, 2],
               monthly[a, 3], monthly[a, 4]))
    print('-' * 82)
    print('表2  问题1 年平均光学效率及输出功率')
    print('-' * 82)
    print('年平均光学效率    : %.6f' % annual[0])
    print('年平均余弦效率    : %.6f' % annual[1])
    print('年平均阴影遮挡效率: %.6f' % annual[2])
    print('年平均截断效率    : %.6f' % annual[3])
    print('年平均输出热功率  : %.3f MW' % (annual_power / 1000.0))
    print('单位面积镜面年平均输出热功率: %.4f kW/m^2' % annual[4])
    print('=' * 82)
    print('网格规格: %d x %d，程序运行时间: %.2f s' %
          (N_W, N_D, time.time() - t0))


if __name__ == '__main__':
    main()
