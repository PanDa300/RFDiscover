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


def fd_LHS(op, RHS):  # 采用BFS的方法挑选LHS候选
    ok = fd_ok(op)
    queue = [[[0] * len(op[0]), 0, 0]]  # 第一个表示目前已经覆盖的行，第二表示目前已经挑选的列,第三个表示到了第几个列
    full_or = get_full_or(op, RHS)
    head = tail = 0
    ans = []
    while head <= tail:
        e = queue[head]
        al_cover_row = e[0]
        al_cover_col = e[1]
        al_num = e[2]
        head += 1
        if my_all_one(al_cover_row):
            ans.append(al_cover_col)
            continue
        if al_num >= len(op):
            continue
        if not my_ok(full_or[al_num], al_cover_row):
            continue
        for i in range(al_num, len(op)):
            if i == RHS:
                continue
            if ok[i] & al_cover_col:
                continue
            cnt = my_calc_or(al_cover_row, op[i])
            if cnt[0] == -1:
                continue
            tail += 1
            queue.append([cnt, al_cover_col | (1 << i), i + 1])
    '''print(ans)'''
    return ans


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
        if score > 0.6:
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


def generte(data):
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


from winepre import pre_glass
from Score import get_Score


def my_print(ans, file):
    f = open(file, 'w')
    for i in ans:
        for j in i.LHS:
            print("%s <= %.2f" % (name_list[j[0]], j[1]), end=' ,', file=f)

        if len(i.LHS) != 0:
            print("-->", end='  ', file=f)
        print("%s <= %.2f" % (name_list[i.RHS[0]], i.RHS[1]), file=f)
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
    ans = generte(Distance)
    end = time.time()

    f = open("wine.txt", 'a+')
    '''my_print(ans, "123.txt")'''
    print("Shape:", Distance.shape, file=f)
    print("Time:", (end - start), file=f)
    print("Cnt:", len(ans), file=f)
    ans2 = []
    for i in ans:
        kk = []
        for j in i.LHS:
            kk.append((j[0], j[1]))
        ans2.append((kk, [i.RHS[0], i.RHS[1]]))
    print("Score:", get_Score(Distance, ans2), file=f)
    f.close()


def work():
    global name_list
    datasize = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    colsize = [4, 5, 6, 7, 8, 9, 10, 11, 12]

    f = open("wine.txt", 'w')
    for i in datasize:
        Distance, name_list = pre_glass(i, 12)
        print("Shape:", i, " X 14", file=f)
        test(Distance, name_list)
    for i in colsize:
        Distance, name_list = pre_glass(2000, i)
        print("Shape:", "300 X ", i, file=f)
        test(Distance, name_list)
    f.close()


if __name__ == "__main__":
    work()
