# -*- coding: utf-8 -*-
"""RFD 评分

score1：置信度 —— "LHS 条件成立时 RHS 也成立"的比例；
score2：紧致度 —— LHS 各属性阈值与 RHS 阈值，分别按该属性最大差值归一化后的平均值；
get_Score：两者取平均。
"""


def ok(row, rfd):
    """0: LHS 不成立；1: LHS 成立但 RHS 不成立；2: LHS、RHS 都成立"""
    for a, t in rfd[0]:
        if row[a] > t:
            return 0
    (ra, rt) = rfd[1]
    if row[ra] > rt:
        return 1
    return 2


def get_Score1(data, rfd):
    """置信度 = 满足 LHS 的样本中同时满足 RHS 的比例"""
    down = up = 0
    for row in data:
        flag = ok(row, rfd)
        if flag == 1:
            down += 1
        elif flag == 2:
            up += 1
            down += 1
    if down == 0:
        return 0.0
    return up / down


def get_Score2(data, rfd, max_dis=None):
    """阈值紧致度：LHS 与 RHS 的阈值各自按其属性最大差值归一化后取平均（0~1）"""
    if max_dis is None:
        max_dis = [max(col) for col in zip(*data)]
    ratios = []
    for a, t in rfd[0]:
        ratios.append(0.0 if max_dis[a] == 0 else t / max_dis[a])
    (ra, rt) = rfd[1]
    ratios.append(0.0 if max_dis[ra] == 0 else rt / max_dis[ra])
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def get_Score(data, rfd, max_dis=None):
    return (get_Score1(data, rfd) + get_Score2(data, rfd, max_dis)) / 2


def get_Score_all(data, rfds, max_dis=None):
    if max_dis is None:
        max_dis = [max(col) for col in zip(*data)]
    if not rfds:
        return 0.0
    return sum(get_Score(data, r, max_dis) for r in rfds) / len(rfds)
