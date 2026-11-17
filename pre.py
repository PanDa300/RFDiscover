# -*- coding: utf-8 -*-
"""数据读入

* 路径相对本文件解析，从任何目录启动都能跑；
* 属性名取 CSV 表头，"Id" 之类的列默认排除（可用 --keep-id 保留）；
* 只保留能转成数值的列。
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "WineQT.csv")

EXCLUDE_DEFAULT = {"id", "ID", "Id", "index", "Unnamed: 0"}


def load_csv(csv_path=DEFAULT_CSV, drop_id=True):
    """读 CSV，返回 (数据矩阵 list[list[float]], 属性名列表)"""
    with open(csv_path, encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    names = [c for c in df.columns]
    if drop_id:
        names = [c for c in names if c.strip() not in EXCLUDE_DEFAULT and c.strip().lower() != "id"]
    df = df[names]
    numeric_names, cols = [], []
    for c in names:
        try:
            cols.append([float(v) for v in df[c].tolist()])
            numeric_names.append(c)
        except (TypeError, ValueError):
            print(f"[pre] 跳过非数值列：{c}")
    matrix = [[cols[k][i] for k in range(len(cols))] for i in range(len(cols[0]))] if cols else []
    return matrix, numeric_names


def head(rows, n):
    """取前 n 行"""
    return [list(r) for r in rows[:n]]
