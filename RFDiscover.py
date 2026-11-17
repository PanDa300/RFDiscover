# -*- coding: utf-8 -*-
"""RFDiscover 主程序：松弛函数依赖（RFD）挖掘

用法（任意目录均可运行）：
    python RFDiscover.py                        # WineQT.csv 前 10 行
    python RFDiscover.py --rows 60              # 前 60 行
    python RFDiscover.py --rows 100 --max-lhs 2 --top 20 --min-support 3
    python RFDiscover.py --keep-id              # 把 Id 列也当属性

输出：每条 RFD 一行 `LHS... --> RHS<=阈值`，附支持度、置信度；最后给出候选数、
有效规则数、平均支持度、平均置信度与 Score。
"""
import argparse
import os
import sys
import time

import numpy as np

from pre import DEFAULT_CSV, head, load_csv
from relax import diff_relation, mine_rfds
from Score import get_Score, get_Score1


def fmt_cond(name, thr):
    return f"{name} <= {thr:.6g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="RFDiscover")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="CSV 路径")
    ap.add_argument("--rows", type=int, default=10, help="取前多少行构造差分对（默认 10）")
    ap.add_argument("--max-lhs", type=int, default=2, choices=[1, 2], help="LHS 属性个数上限（1 或 2）")
    ap.add_argument("--min-support", type=int, default=1, help="最小支持度（默认 1）")
    ap.add_argument("--top", type=int, default=20, help="最多打印多少条规则")
    ap.add_argument("--keep-id", action="store_true", help="把 Id 列也当属性")
    args = ap.parse_args(argv)

    t0 = time.time()
    rows, names = load_csv(args.csv, drop_id=not args.keep_id)
    rows = head(rows, args.rows)
    if len(rows) < 3:
        sys.exit("[错误] 至少需要 3 行数据")
    D = np.asarray(diff_relation(rows), dtype=float)
    print(f"数据：{os.path.basename(args.csv)}  前 {len(rows)} 行  属性 {len(names)} 个")
    print(f"差分行数：{D.shape[0]}（= C({len(rows)},2)），差分矩阵：{D.shape}")

    rules, tries = mine_rfds(D, max_lhs=args.max_lhs, min_support=args.min_support, verbose=True)

    # 只保留无反例（置信度 = 1）的规则
    bad = [r for r in rules if r.confidence(D) < 1.0 - 1e-9]
    if bad:
        print(f"[提示] {len(bad)} 条规则置信度 < 1，已剔除")
        rules = [r for r in rules if r.confidence(D) >= 1.0 - 1e-9]

    dt = time.time() - t0
    print(f"\n候选 (RHS属性, RHS阈值)：{tries}")
    print(f"有效 RFD：{len(rules)} 条（全部置信度 = 1，已逐条校验）")
    if rules:
        sup = [r.support for r in rules]
        confs = [get_Score1(D, (r.LHS, r.RHS)) for r in rules]
        scores = [get_Score(D, (r.LHS, r.RHS)) for r in rules]
        print(f"支持度：中位数 {int(np.median(sup))}，最大 {max(sup)}，共 {D.shape[0]} 个差分行")
        print(f"平均支持度 {np.mean(sup):.1f} / {D.shape[0]}    平均置信度 {np.mean(confs):.3f}")
        print(f"Score：{np.mean(scores):.6f}")

    print(f"\n时间：{dt:.3f} s")
    print(f"\n=== 规则（按支持度降序，最多 {args.top} 条）===")
    for r in rules[: args.top]:
        lhs = " ,".join(fmt_cond(names[a], t) for a, t in sorted(r.LHS))
        ra, rt = r.RHS
        print(f"{lhs}  -->  {fmt_cond(names[ra], rt)}   [support={r.support}, conf=1.000]")
    if len(rules) > args.top:
        print(f"...（另有 {len(rules) - args.top} 条，用 --top 调整）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
