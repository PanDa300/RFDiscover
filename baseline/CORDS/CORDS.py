from relax import FD
from relax import RFD
import numpy as np
import math
import pandas as pd
import random
import sys
import os


def all_one(S):
    for i in S:
        if i == 0:
            return False
    return True


def my_or(x, y):
    z = []
    for i in range(len(x)):
        if x[i] == 1 or y[i] == 1:
            z.append(1)
        else:
            z.append(0)
    return z


def my_contain(x, y):
    flag = 1
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            flag = 0
            break
    if flag:
        return True
    flag = 1
    for i in range(len(y)):
        if y[i] == 0 and x[i] == 1:
            flag = 0
            break
    if flag:
        return True
    return False


def fd_ok(op):
    ok = [[True] * len(op)] * len(op)  # ok[i][j] 表示i 和 j两个能否同时出现
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            if my_contain(op[i], op[j]):
                ok[i][j] = ok[j][i] = True
    res = []
    for i in range(len(op)):
        res.append(0)
        for j in range(len(op)):
            if i == j or ok[i][j]:
                res[i] = res[i] << 1 | 1
            else:
                res[i] = res[i] << 1
    return res


def get_full_or(op, RHS):
    res = [[0] * len(op[0])] * (len(op) + 1)
    for i in range(len(op) - 1, -1, -1):
        if i == RHS:
            res[i] = res[i + 1]
            continue
        res[i] = my_or(res[i + 1], op[i])
    return res


def my_ok(x, y):
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            return False
    return True


def my_calc_or(x, y):
    res = []
    flag = 0
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            flag = 1
        res.append(x[i] | y[i])
    if flag == 0:
        res[0] = -1
    return res


def my_all_one(S):
    for i in S:
        if i == 0:
            return False
    return True


def reduce(li):  # 二进制转化为数组
    new_li = list(set(li))
    res = []
    for i in new_li:
        res.append(bit_to_list(i))
    return res


def gen_LHS(data, RHS):
    x = len(data[0])
    res = []
    for i in range(x):
        if i == RHS:
            continue
        x_simple = [data[ll][RHS] for ll in range(len(data))]
        y_simple = [data[ll][i] for ll in range(len(data))]
        score = np.corrcoef(x_simple, y_simple)
        score = score[0][1]
        if abs(score) > 0.4:
            res.append(i)
    return res


def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]


def bit_to_list(t):
    S = []
    cnt = -1
    while t:
        cnt += 1
        op = t % 2
        t = t >> 1
        if op == 1:
            S.append(cnt)
    return S


def CORDS(data):
    difference_list = data
    data = difference_list
    res = []
    data_tol = data
    for i in range(len(data[0])):
        '''print(i)'''
        df = [data[ll][i] for ll in range(len(data))]
        LHS_list = gen_LHS(data, i)
        df = list(set(df))
        step = 1
        k_id = 1
        while k_id < len(df):
            '''print(k_id)'''
            k = df[k_id]
            if k == df[0]:
                continue
            j = LHS_list
            now_FD = FD(j, i)
            new_RFD = RFD(now_FD, k)
            new_RFD.generator(data)
            res.append(new_RFD)
            k_id += step
            step *= 2

        data = data_tol
    return res


from pre import pre_glass
from Score import get_Score


def my_print(ans, file):
    f = open(file, 'w')
    for i in ans:
        for j in i.LHS:
            print("(%s , <= , %.2f)" % (name_list[j[0]], j[1]), end=' ', file=f)

        if len(i.LHS) != 0:
            print(",", end='  ', file=f)
        print("(%s , <= , %.2f)" % (name_list[i.RHS[0]], i.RHS[1]), file=f)
    f.close()


def make_data(Distance_new, rate):
    for i in range(len(Distance_new)):
        rad = random.random()
        if rad < rate:
            Distance_new[i] = Distance_new[i] * rate
    return Distance_new


def test(Distance, name_list):
    import time

    start = time.time()
    ans = CORDS(Distance)
    end = time.time()

    my_print(ans, "123.txt")
    print("Shape:", Distance.shape)
    print("Time:", (end - start))
    print("Cnt:", len(ans))
    ans2 = []
    for i in ans:
        kk = []
        for j in i.LHS:
            kk.append((j[0], j[1]))
        ans2.append((kk, [i.RHS[0], i.RHS[1]]))
    print("Score:", get_Score(Distance, ans2))


def work():
    global name_list
    Distance, name_list = pre_glass(50, 6)
    test(Distance, name_list)


if __name__ == "__main__":
    work()
