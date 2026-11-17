# RFDiscover

**松弛函数依赖（Relaxed Functional Dependency, RFD）** 发现工具：在数值数据上挖掘形如
`A1 <= t1 ∧ A2 <= t2 --> y <= t` 的规则，并给出支持度、置信度与综合评分。

论文实现：*[ICDE 2024] Efficient Relaxed Functional Dependency Discovery*

纯 Python 实现（NumPy / pandas），单机 CPU 秒级出结果。

---

## 方法概览

RFD 是对传统函数依赖的松弛：不再要求 `X 相等 ⇒ y 相等`，而允许
`X 的差 ≤ 阈值 ⇒ y 的差 ≤ 阈值`，从而能刻画连续属性上的近似规律。

挖掘流程（`relax.py`）：

1. **差分关系**：对数据两两求绝对差，得到 `C(n,2) × 属性数` 的差分矩阵 `D`。
2. **逐 RHS 枚举**：对每个 (RHS 属性 `y`, 阈值 `t`)，违约集合 `V = {行 : D[:, y] > t}`
   必须被 LHS 条件全部排除。
3. **构造 LHS**：枚举 1~`max-lhs` 个属性；阈值取"刚好低于该违约行最小差值"的整齐数
   （`nice_below`），保证这些违约行在 `<=` 判定下必然被排除，即置信度 = 1。
4. **择优**：同一 RHS 下取支持度最大的候选；同一 LHS 下只保留 RHS 阈值最紧（最强）的那条。
5. **校验**：逐条复算置信度，只保留无任何反例（置信度 = 1）的规则。

评分（`Score.py`）：

- `Score1` = **置信度**：LHS 成立的差分行中 RHS 同时成立的比例；
- `Score2` = **紧致度**：LHS / RHS 各阈值按其属性最大差值归一化后的平均值（越小越紧）；
- `Score` = 两者平均。

阈值打印时会向下取整到 4 位有效数字（如 `0.03299`、`48.99`），
既好读，也保证按打印值复算时仍然严格小于原始边界，不会因四舍五入多出反例。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `RFDiscover.py` | 主程序入口（命令行、流程编排、结果输出） |
| `relax.py` | 差分关系构造与 RFD 挖掘（`diff_relation` / `mine_rfds` / `RFD` / `FD`） |
| `Score.py` | 评分函数（置信度、紧致度、综合 Score） |
| `pre.py` | 数据读入（CSV → 数值矩阵，默认排除 `Id` 列） |
| `WineQT.csv` | 示例数据（UCI Wine Quality，红葡萄酒 1143 行 12 属性） |

## 环境

```bash
pip install -r requirements.txt
```

Python ≥ 3.8，依赖 `numpy`、`pandas`（仅 `pre.py` 读取 CSV 用）。

## 运行

```bash
python RFDiscover.py                                   # WineQT.csv 前 10 行
python RFDiscover.py --rows 60                         # 前 60 行
python RFDiscover.py --rows 100 --max-lhs 2 --top 20 --min-support 3
python RFDiscover.py --keep-id                         # 把 Id 列也当属性
python RFDiscover.py --csv D:\data\mydata.csv --rows 200
```

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--csv` | 本目录 `WineQT.csv` | 输入 CSV 路径 |
| `--rows` | 10 | 取前多少行构造差分对（差分行数 = C(n,2)） |
| `--max-lhs` | 2 | LHS 属性个数上限（1 或 2） |
| `--min-support` | 1 | 最小支持度 |
| `--top` | 20 | 最多打印多少条规则 |
| `--keep-id` | 关闭 | 把 `Id` 列也当属性（默认排除） |

> 耗时随差分行数 `C(rows,2)` 近似线性增长：`--rows 10`（45 行）约 1 秒，
> `--rows 60`（1770 行）约 40 秒，`--rows 100`（4950 行）需 1~2 分钟。
> 时间主要花在 `--max-lhs 2` 的双属性枚举上，用 `--max-lhs 1` 会快一个量级。

### 输出示例

```
数据：WineQT.csv  前 10 行  属性 12 个
差分行数：45（= C(10,2)），差分矩阵：(45, 12)
[mine] 尝试 204 个 (RHS属性, 阈值) 组合，得到 66 条有效候选

候选 (RHS属性, RHS阈值)：204
有效 RFD：43 条（全部置信度 = 1，已逐条校验）
支持度：中位数 36，最大 44，共 45 个差分行
平均支持度 33.9 / 45    平均置信度 1.000
Score：0.865713

时间：0.875 s

=== 规则（按支持度降序，最多 20 条）===
chlorides <= 0.03299  -->  residual sugar <= 1.1   [support=44, conf=1.000]
residual sugar <= 1.399  -->  chlorides <= 0.032   [support=44, conf=1.000]
total sulfur dioxide <= 48.99  -->  free sulfur dioxide <= 14   [support=44, conf=1.000]
...
```

## 使用自己的数据

1. 准备一个表头行 + 数值列的 CSV，放到任意位置；
2. `python RFDiscover.py --csv 路径.csv --rows 100`；
3. 非数值列会被自动跳过（`pre.py` 打印 `[pre] 跳过非数值列：...`）；
4. 如果标识列不叫 `Id`/`index`/`Unnamed: 0`，建议先在 CSV 里改名或删除。

## 说明

- 只输出**置信度严格等于 1** 的规则，因此规则在给定数据上无反例；
  若需要放宽，可去掉 `RFDiscover.py` 中 `confidence < 1` 的过滤。
- `relax.py` 里 `candidate_thresholds()` 对每个属性的候选阈值做了上限压缩（默认 60 个分位点），
  属性取值域很大时可通过 `mine_rfds(..., thr_cap=...)` 调整。
- 仓库暂未附带开源许可证文件，如需复用请先与作者联系。

## 引用

```bibtex
@inproceedings{rfdiscover2024,
  title     = {Efficient Relaxed Functional Dependency Discovery},
  booktitle = {IEEE International Conference on Data Engineering (ICDE)},
  year      = {2024}
}
```
