# -*- coding: utf-8 -*-
"""问题3：异质镜面分区布局的光学计算与遗传算法优化。"""
import numpy as np

PHI = np.deg2rad(39.4)
G0, H_SEA, REF_EFF = 1.366, 3.0, 0.92
H_HEAT, R_REC, THETA_SUN = 76.0, 3.5, 4.65e-3
DAYS = [-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
TIMES = [9.0, 10.5, 12.0, 13.5, 15.0]

# 塔纵坐标、分区半径、内外区镜宽、两区相位、高宽比和镜面净距
LOWER = np.array([-120., 120., 2., 2., 0., 0., .85, .85, .50])
UPPER = np.array([  20., 320., 8., 8., .99, .99, 1., 1., .90])


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
    """生成给定镜宽对应的三角点阵。"""
    dx, dy = width + 5, np.sqrt(3) * (width + 5) / 2
    rows = []
    for j, y in enumerate(np.arange(-350, 351, dy)):
        shift = (j % 2) * dx / 2 + phase * dx
        for x in np.arange(-350 - dx, 351 + dx, dx) + shift:
            if x * x + y * y <= 350 ** 2 and x * x + (y - tower_y) ** 2 >= 100 ** 2:
                rows.append((x, y))
    return np.asarray(rows)


def make_layout(p):
    """从空场生成内外区点阵，并逐镜设置尺寸和安装高度。"""
    tower_y, boundary, inner_w, outer_w = p[:4]
    points, widths, heights = [], [], []
    zones = ((inner_w, p[4], p[6]), (outer_w, p[5], p[7]))
    for k, (width, phase, ratio) in enumerate(zones):
        pool = triangular_layout(width, tower_y, phase)
        radius = np.linalg.norm(pool - np.array([0.0, tower_y]), axis=1)
        pool = pool[radius < boundary] if k == 0 else pool[radius >= boundary]
        for q in pool:
            limit = ((width + np.asarray(widths)) / 2 + 5
                     if widths else np.array([]))
            if not points or np.all(np.linalg.norm(np.asarray(points) - q, axis=1) >= limit - 1e-8):
                points.append(q)
                widths.append(width)
                heights.append(width * ratio)
    points = np.asarray(points)
    widths, heights = np.asarray(widths), np.asarray(heights)
    z = np.maximum(2.0, heights / 2 + p[8])
    return points, widths, heights, z


def calc_power(points, widths, heights, z, tower_y):
    """按每面镜的真实面积汇总60个时点的输出热功率。"""
    base = np.column_stack((points, z))
    receiver = np.array([0.0, tower_y, H_HEAT])
    v = receiver - base
    distance = np.linalg.norm(v, axis=1)
    v /= distance[:, None]
    air_eff = .99321 - .0001176 * distance + 1.97e-8 * distance ** 2
    mirror_area = widths * heights
    dxy = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(dxy, np.inf)
    clearance = np.min(dxy - (widths[:, None] + widths[None, :]) / 2, axis=1)
    power = 0.0
    for day in DAYS:
        for st in TIMES:
            u, dni = calc_sun(day, st)
            n = u + v
            n /= np.linalg.norm(n, axis=1)[:, None]
            cos_eff = np.maximum(0, n @ u)
            spot = distance * np.tan(THETA_SUN)
            trunc_eff = np.minimum(1.0, (R_REC / spot) ** 2)
            block_eff = np.clip(.94 + .012 * clearance + .04 * u[2], 0, 1)
            eta = REF_EFF * cos_eff * air_eff * block_eff * trunc_eff
            power += dni * np.sum(mirror_area * eta)
    power /= len(DAYS) * len(TIMES) * 1000
    return power, 1000 * power / np.sum(mirror_area)


def evaluate(p):
    p = np.clip(p, LOWER, UPPER)
    points, widths, heights, z = make_layout(p)
    power, unit_power = calc_power(points, widths, heights, z, p[0])
    return power, unit_power, points, widths, heights, z


def fitness(result):
    """容量可行时最大化单位面积功率，否则优先缩小容量缺口。"""
    power, unit_power = result[:2]
    return (1, unit_power, power) if power >= 60 else (0, power, unit_power)


def genetic_algorithm(n_population=14, n_generation=5):
    rng = np.random.default_rng(20230901)
    population = rng.uniform(LOWER, UPPER, (n_population, len(LOWER)))
    best, best_result = population[0].copy(), None
    for _ in range(n_generation):
        results = [evaluate(p) for p in population]
        order = sorted(range(n_population), key=lambda i: fitness(results[i]), reverse=True)
        if best_result is None or fitness(results[order[0]]) > fitness(best_result):
            best, best_result = population[order[0]].copy(), results[order[0]]
        elite = population[order[:max(3, n_population // 3)]]
        children = [elite[0].copy(), elite[1].copy()]
        while len(children) < n_population:
            a, b = elite[rng.integers(len(elite), size=2)]
            weight = rng.uniform(.2, .8, len(LOWER))
            child = weight * a + (1 - weight) * b
            mutation = rng.random(len(LOWER)) < .30
            child += mutation * rng.normal(0, .07, len(LOWER)) * (UPPER - LOWER)
            children.append(np.clip(child, LOWER, UPPER))
        population = np.asarray(children)
    return best, best_result


if __name__ == '__main__':
    best, result = genetic_algorithm()
    print('最优参数：', np.round(best, 6))
    print('年平均功率：%.4f MW，单位面积功率：%.6f kW/m^2' % result[:2])
