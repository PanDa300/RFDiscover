from relax import FD
from relax import RFD
import numpy as np
import math
import pandas as pd
import random
import sys
import os

'''
 判断是否全为1，用于集合覆盖过程，判断是否所有的数据都被覆盖
 @param S 集合，为一个01组成的数组
 @return bool True:全为1  False：不全为1
'''
def all_one(S):
    for i in S:
        if i == 0:
            return False
    return True


'''
 针对01串进行或操作
 @param x,y 两个01串，为01组成的数组
 @return z 执行或操作后的结果，为01数组
'''
def my_or(x, y):
    z = []
    for i in range(len(x)):
        if x[i] == 1 or y[i] == 1:
            z.append(1)
        else:
            z.append(0)
    return z


'''
 判断两个01串是否存在包含关系，即是否x覆盖的数据y都覆盖，或这反之成立
 @param x,y 两个01串，为01组成的数组
 @return bool True:存在x覆盖y或者y覆盖x  False：不存在***
'''
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


'''
 根据对应的集合覆盖求得所有的极小覆盖集合（即LHS候选集合）
 @param op k*n k个长度为n个覆盖
 @return res 极小覆盖集合
'''
def fd_ok(op):
    ok = [[False] * len(op)] * len(op)  # ok[i][j] 表示i 和 j两个能否同时出现
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            if my_contain(op[i], op[j]):
                ok[i][j] = ok[j][i] = True
    res = []
    for i in range(len(op)):
        res.append(0)
        for j in range(len(op)):
            if i == j or not ok[i][j]:
                res[i] = res[i] << 1 | 1
            else:
                res[i] = res[i] << 1
    return res


'''
 根据对应的覆盖集合，生成后k条覆盖集合的或
 @param op，RHS op：k*n k个长度为n个覆盖  RHS：跳过的RHS属性
 @return res 后k条覆盖集合集
'''


def get_full_or(op, RHS):
    res = [[0] * len(op[0])] * (len(op) + 1)
    for i in range(len(op) - 1, -1, -1):
        if i == RHS:
            res[i] = res[i + 1]
            continue
        res[i] = my_or(res[i + 1], op[i])
    return res


'''
 判断2个01串是否或为1
 @param x,y 01串
 @return bool True:为1  False：不全为1
'''

def my_ok(x, y):
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0:
            return False
    return True

'''
 进行or运算
 @param x,y 01串
 @return or结果
'''

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

'''
 （和之前的重了）
 判断是否全为1，用于集合覆盖过程，判断是否所有的数据都被覆盖
 @param S 集合，为一个01组成的数组
 @return bool True:全为1  False：不全为1
'''

def my_all_one(S):
    for i in S:
        if i == 0:
            return False
    return True

'''
 采用BFS的方法挑选LHS候选
 @param op，RHS op：k*n k个长度为n个覆盖  RHS：跳过的RHS属性
 @return ans LHS候选
'''

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
    return ans


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


'''
 将int转化为01数组
 @param li 一个整型数组
 @return res 01串组成的list
'''

def reduce(li):  # 二进制转化为数组
    new_li = list(set(li))
    res = []
    for i in new_li:
        res.append(bit_to_list(i))
    return res

'''
 生成LHS候选，该函数主要是为了给fd_LHS作预处理
 @param data, RHS, RHSv   data：元组对  RHS：RHS属性 RHS：RHS阈值
 @return res LHS候选
'''


def gen_LHS(data, RHS, RHSv):
    beta_1 = [x for x in data if x[RHS] <= RHSv]
    beta_2 = [x for x in data if x[RHS] > RHSv]
    min_value = []
    for j in range(len(beta_1[0])):
        maxn = 0
        for i in range(len(beta_1)):
            maxn = max(maxn, beta_1[i][j])
        min_value.append(maxn)
    op = [[] for i in range(len(beta_1[0]))]
    for i in beta_2:
        for j in range(len(i)):
            if i[j] < min_value[j]:
                op[j].append(0)
            else:
                op[j].append(1)

    res = fd_LHS(op, RHS)
    res = reduce(res)
    return res


'''
 把矩阵按照第i列排序
 @param diff_list, i   diff_list：元组对  i：某一列
 @return 排序后结果
'''
def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]




def generte(data):
    res = []
    for i in range(len(data[0])):
        df = [data[ll][i] for ll in range(len(data))]
        df = list(set(df))
        df.sort()
        p = len(df) - 1
        step = 1
        while p >= 0:

            k = df[p]
            LHS_list = gen_LHS(data, i, k)
            for j in LHS_list:
                now_FD = FD(j, i)
                new_RFD = RFD(now_FD, k)
                new_RFD.generator(data)
                res.append(new_RFD)
            p = p - step
            '''step *= 2'''
    return res


from pre import pre_glass
from Score import get_Score


def test(Distance, name_list):
    import time

    start = time.time()
    ans = generte(Distance)
    end = time.time()
    for i in ans:
        for j in i.LHS:
            print("%s <= %.2f" % (name_list[j[0]], j[1]), end=' ,')
        print("-->", end='  ')
        print("%s <= %.2f" % (name_list[i.RHS[0]], i.RHS[1]))
    print("Glass:", Distance.shape)
    print("Time:", (end - start))
    print("Cnt:", len(ans))
    ans2 = []
    for i in ans:
        kk = []
        for j in i.LHS:
            kk.append((j[0], j[1]))
        ans2.append((kk, [i.RHS[0], i.RHS[1]]))
    print("Score:", get_Score(Distance, ans2))


def make_data(Distance_new, rate):
    for i in range(len(Distance_new)):
        rad = random.random()
        if rad < rate:
            Distance_new[i] = Distance_new[i] * rate
    return Distance_new


if __name__ == "__main__":
    global name_list
    Distance, name_list = pre_glass(10)  #生成前10个
    print("No.", 10)

    test(Distance, name_list)
