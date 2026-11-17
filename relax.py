# -*- coding: utf-8 -*-
"""差分关系构造与 RFD 挖掘

对每个 (RHS 属性 y, RHS 阈值 t)：
  1. 违约集合 V = {差分行 : dy > t}，这些行必须被 LHS 条件排除；
  2. 在 1~max_lhs 个 LHS 属性上枚举阈值（阈值取自该属性出现过的差值）；
  3. 阈值取"刚好低于某组违约行的最小差值"的整齐数（nice_below），
     保证这些违约行在 <= 判定下必然被排除，即规则置信度 = 1；
  4. 取支持度最大的候选。

阈值向下取整到 4 位有效数字后再输出：既好读，也保证按打印值复算时仍然
严格小于原边界（否则会因四舍五入多出一条反例）。
"""
import math

import numpy as np


class FD:
    def __init__(self, LHS_list, RHS):
        self.LHS = LHS_list
        self.RHS = RHS

    def __repr__(self):
        return f"FD({self.LHS} -> {self.RHS})"


def nice_below(v, digits=4):
    """返回严格小于 v 的整齐数（保留 digits 位有效数字）。

    例：nice_below(0.033) -> 0.03299；nice_below(49.0) -> 48.99。
    这样打印出来（%.6g）再解析回来，仍然严格小于原始边界值。
    """
    v = float(v)
    if v <= 0:
        return 0.0
    exp = int(math.floor(math.log10(abs(v))))
    step = 10.0 ** (exp - digits + 1)
    t = math.floor(v / step + 1e-12) * step
    if t >= v:
        t -= step
    decimals = max(0, digits - 1 - exp)
    return round(t, decimals)


def nice_floor(v, digits=4):
    """返回不大于 v 的整齐数（保留 digits 位有效数字）。

    RHS 阈值取自数据里出现过的差值（如 0.6000000000000014），直接打印会显示成
    0.6，与内部值不一致。先向下取整到整齐数，内部值与打印值就完全对得上。
    """
    v = float(v)
    if v <= 0:
        return 0.0
    exp = int(math.floor(math.log10(abs(v))))
    step = 10.0 ** (exp - digits + 1)
    t = math.floor(v / step + 1e-12) * step
    decimals = max(0, digits - 1 - exp)
    return round(t, decimals)


class RFD:
    """松弛函数依赖：(X, thr_X) -> (y, thr_y)

    LHS: [(属性下标, 阈值), ...]；RHS: (属性下标, 阈值)
    support: 满足全部 LHS 条件的差分行数
    """

    def __init__(self, LHS, RHS, support=0, total=0):
        self.LHS = list(LHS)
        self.RHS = RHS
        self.support = support
        self.total = total

    def lhs_holds_mask(self, D):
        m = np.ones(len(D), dtype=bool)
        for a, t in self.LHS:
            m &= D[:, a] <= t
        return m

    def rhs_holds_mask(self, D):
        a, t = self.RHS
        return D[:, a] <= t

    def confidence(self, D):
        lhs = self.lhs_holds_mask(D)
        down = int(lhs.sum())
        return 0.0 if down == 0 else int((lhs & self.rhs_holds_mask(D)).sum()) / down

    def support_count(self, D):
        return int(self.lhs_holds_mask(D).sum())

    def key(self):
        return (tuple(sorted(self.LHS)), self.RHS)

    def __repr__(self):
        lhs = ", ".join(f"{a}<={t:g}" for a, t in sorted(self.LHS))
        return f"{lhs} -> {self.RHS[0]}<={self.RHS[1]:g}"


def diff_relation(rows):
    """两两绝对差分行"""
    n = len(rows)
    return [[abs(rows[i][k] - rows[j][k]) for k in range(len(rows[0]))]
            for i in range(n - 1) for j in range(i + 1, n)]


def candidate_thresholds(column, cap):
    """候选阈值：该属性上出现过的差值；太多时按分位点压缩到 cap 个"""
    vals = np.unique(column)
    if len(vals) <= cap:
        return vals
    idx = np.linspace(0, len(vals) - 1, cap).round().astype(int)
    return vals[idx]


def _best_lhs_for(D, y, t, max_lhs, thr_cap):
    """在 (y, t) 下找支持度最大的 1~max_lhs 属性 LHS（阈值取可行范围内最大）"""
    n, A = D.shape
    viol = D[:, y] > t          # 必须被 LHS 排除的差分行
    if not viol.any():
        return None             # RHS 对全部差分行都成立：平凡规则，不输出

    others = [k for k in range(A) if k != y]
    cands = []                  # (support, -len(LHS), 阈值和, LHS)

    # --- 单属性 LHS：阈值取全部违约行在该属性上的最小值再降一格 ---
    for k in others:
        thr = nice_below(D[viol, k].min())
        sup = int((D[:, k] <= thr).sum())
        cands.append((sup, -1, thr, [(k, thr)]))

    # --- 双属性：枚举 k1 阈值，k2 阈值取"刚好排除剩余违约行"的值 ---
    if max_lhs >= 2:
        for i, k1 in enumerate(others):
            for thr1 in candidate_thresholds(D[:, k1], thr_cap):
                if thr1 <= 0:
                    continue
                thr1 = float(nice_below(thr1))
                rest = viol & (D[:, k1] <= thr1)     # k1 没排除掉的违约行
                if not rest.any():
                    continue                          # k1 已排干净，单属性候选已覆盖
                for k2 in others[i + 1:]:
                    thr2 = nice_below(D[rest, k2].min())
                    sup = int(((D[:, k1] <= thr1) & (D[:, k2] <= thr2)).sum())
                    cands.append((sup, -2, thr1 + thr2, [(k1, thr1), (k2, thr2)]))

    if not cands:
        return None
    sup, neg_len, _, lhs = max(cands, key=lambda c: (c[0], c[1], c[2]))
    return RFD(lhs, (y, t), support=sup, total=n)


def mine_rfds(D, rhs_attrs=None, max_lhs=2, thr_cap=60, min_support=1, verbose=False):
    """对每个 (RHS 属性, RHS 阈值) 求支持度最大的有效 RFD"""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] == 0:
        return [], 0
    n, A = D.shape
    if rhs_attrs is None:
        rhs_attrs = range(A)

    found, tries = [], 0
    for y in rhs_attrs:
        rhs_candidates = sorted({nice_floor(t) for t in np.unique(D[:, y])})
        for t in rhs_candidates:
            tries += 1
            rfd = _best_lhs_for(D, y, float(t), max_lhs, thr_cap)
            if rfd is None or not rfd.LHS or rfd.support < min_support:
                continue
            # 逐条校验，只收无反例的规则
            if rfd.confidence(D) < 1.0 - 1e-12:
                continue
            found.append(rfd)

    if verbose:
        print(f"[mine] 尝试 {tries} 个 (RHS属性, 阈值) 组合，得到 {len(found)} 条有效候选")

    # 同一 LHS 只保留 RHS 阈值最紧（最强）的那条
    best = {}
    for r in found:
        lhs_key = tuple(sorted(r.LHS))
        if lhs_key not in best or r.RHS[1] < best[lhs_key].RHS[1]:
            best[lhs_key] = r
    rules = [r for r in best.values() if r.confidence(D) >= 1.0 - 1e-12]
    rules.sort(key=lambda r: (-r.support, len(r.LHS), r.RHS[0], r.RHS[1]))
    return rules, tries
