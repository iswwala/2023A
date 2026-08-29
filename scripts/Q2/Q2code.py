# -*- coding: utf-8 -*-
"""问题2：分区蜂窝镜场的光学计算与粒子群优化。"""
import numpy as np

PHI = np.deg2rad(39.4)
G0, H_SEA, REF_EFF = 1.366, 3.0, 0.92
H_HEAT, R_REC, THETA_SUN = 76.0, 3.5, 4.65e-3
DAYS = [-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
TIMES = [9.0, 10.5, 12.0, 13.5, 15.0]

# 镜宽、高宽比、净距、塔纵坐标、两分区半径、三区相位、过渡带系数
LOWER = np.array([5.8, .65, .50, -120., 135., 60., 0., 0., 0., .75])
UPPER = np.array([8.0, 1.0, .90,   40., 210., 125., .5, .5, .5, 2.50])


def calc_sun(day, st):
    """计算太阳方向和DNI。"""
    omega = np.pi / 12 * (st - 12)
    delta = np.arcsin(np.sin(2 * np.pi * day / 365) * np.sin(np.deg2rad(23.45)))
    alpha = np.arcsin(np.cos(delta) * np.cos(PHI) * np.cos(omega) +
                      np.sin(delta) * np.sin(PHI))
    cos_gamma = ((np.sin(delta) - np.sin(alpha) * np.sin(PHI)) /
                 (np.cos(alpha) * np.cos(PHI)))
    gamma = np.arccos(np.clip(cos_gamma, -1, 1))
    if omega > 0:
        gamma = 2 * np.pi - gamma
    u = np.array([np.cos(alpha) * np.sin(gamma),
                  np.cos(alpha) * np.cos(gamma), np.sin(alpha)])
    a = .4237 - .00821 * (6 - H_SEA) ** 2
    b = .5055 + .00595 * (6.5 - H_SEA) ** 2
    c = .2711 + .01858 * (2.5 - H_SEA) ** 2
    dni = G0 * (a + b * np.exp(-c / np.sin(alpha)))
    return u, dni


def triangular_layout(width, tower_y, phase):
    """在圆形场地内生成三角点阵。"""
    dx = width + 5.0
    dy = np.sqrt(3) * dx / 2
    rows = []
    for j, y in enumerate(np.arange(-350, 351, dy)):
        shift = (j % 2) * dx / 2 + phase * dx
        x = np.arange(-350 - dx, 351 + dx, dx) + shift
        rows.extend((xx, y) for xx in x if xx * xx + y * y <= 350 ** 2 and
                    xx * xx + (y - tower_y) ** 2 >= 100 ** 2)
    return np.asarray(rows)


def make_layout(p):
    """按三区相位生成布局，并删除间距不足的点。"""
    width, tower_y, r1, r2 = p[0], p[3], p[4], min(325., p[4] + p[5])
    tower = np.array([0.0, tower_y])
    points = []
    for k, phase in enumerate(p[6:9]):
        pool = triangular_layout(width, tower_y, phase)
        radius = np.linalg.norm(pool - tower, axis=1)
        keep = radius < r1 if k == 0 else ((radius >= r1) & (radius < r2)
                                            if k == 1 else radius >= r2)
        points.extend(pool[keep])
    points = sorted(points, key=lambda q: np.linalg.norm(q - tower))
    accepted = []
    for q in points:
        if not accepted or np.min(np.linalg.norm(np.asarray(accepted) - q, axis=1)) >= width + 5 - 1e-8:
            accepted.append(q)
    return np.asarray(accepted)


def calc_power(points, width, height, z, tower_y):
    """计算60个规定时点的年平均输出功率。"""
    base = np.column_stack((points, np.full(len(points), z)))
    receiver = np.array([0.0, tower_y, H_HEAT])
    v = receiver - base
    distance = np.linalg.norm(v, axis=1)
    v /= distance[:, None]
    air_eff = .99321 - .0001176 * distance + 1.97e-8 * distance ** 2
    dxy = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(dxy, np.inf)
    clearance = np.min(dxy, axis=1) - width
    power = 0.0
    for day in DAYS:
        for st in TIMES:
            u, dni = calc_sun(day, st)
            n = u + v                       # 法向为入射、反射方向的角平分线
            n /= np.linalg.norm(n, axis=1)[:, None]
            cos_eff = np.maximum(0, n @ u)
            spot = distance * np.tan(THETA_SUN)
            trunc_eff = np.minimum(1.0, (R_REC / spot) ** 2)
            block_eff = np.clip(.94 + .012 * clearance + .04 * u[2], 0, 1)
            eta = REF_EFF * cos_eff * air_eff * block_eff * trunc_eff
            power += dni * width * height * np.sum(eta)
    power /= len(DAYS) * len(TIMES) * 1000
    return power, 1000 * power / (len(points) * width * height)


def evaluate(p):
    width, height = p[0], max(2.0, p[0] * p[1])
    z = height / 2 + p[2]
    points = make_layout(p)
    power, unit_power = calc_power(points, width, height, z, p[3])
    return power, unit_power, points


def fitness(result):
    """先满足60 MW约束，再最大化单位面积功率。"""
    power, unit_power = result[:2]
    return (1, unit_power, power) if power >= 60 else (0, power, unit_power)


def particle_swarm(n_particle=10, n_iter=6):
    rng = np.random.default_rng(2023)
    x = rng.uniform(LOWER, UPPER, (n_particle, len(LOWER)))
    vel = rng.uniform(-.08 * (UPPER - LOWER), .08 * (UPPER - LOWER), x.shape)
    p_best, p_result = x.copy(), [None] * n_particle
    g_best, g_result = x[0].copy(), None
    for _ in range(n_iter):
        for i in range(n_particle):
            result = evaluate(x[i])
            if p_result[i] is None or fitness(result) > fitness(p_result[i]):
                p_best[i], p_result[i] = x[i].copy(), result
            if g_result is None or fitness(result) > fitness(g_result):
                g_best, g_result = x[i].copy(), result
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        vel = .72 * vel + 1.49 * r1 * (p_best - x) + 1.49 * r2 * (g_best - x)
        vel = np.clip(vel, -.22 * (UPPER - LOWER), .22 * (UPPER - LOWER))
        x = np.clip(x + vel, LOWER, UPPER)
    return g_best, g_result


if __name__ == '__main__':
    best, result = particle_swarm()
    print('最优参数：', np.round(best, 6))
    print('年平均功率：%.4f MW，单位面积功率：%.6f kW/m^2' % result[:2])
