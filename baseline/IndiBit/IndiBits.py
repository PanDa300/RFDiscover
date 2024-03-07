import math
from RFD import RFD, FD


def CalcDistance(x, y):
    return math.fabs(x - y)


def CalcSatisfie(x, y):
    if y == -1:
        return True
    if x > y:
        return False
    return True


def insert_update_vector(r, t, Phi, S):
    id = len(r)
    for B in range(len(r[0])):
        S[B].append(1 << id)
        for k in range(id):
            if CalcSatisfie(CalcDistance(t[B], r[k][B]), Phi[B]):
                S[B][k] += 1 << id
                S[B][id] += 1 << k
    return S


def validate(fd, S):
    S1 = [0] * len(S)
    LHS = fd.LHS
    RHS = fd.RHS
    for i in LHS:
        for j in range(len(S1)):
            S1[j] = S1[j] & S[i][j]
    S2 = S1
    for j in range(len(S1)):
        S2[j] = S2[j] & S[RHS][j]
    for i in range(len(S1)):
        if S1[i] != S2[i]:
            return False
    return True


def contain(rfd, rfds):
    cnt = 0
    for r in rfds:
        if r.RHS != rfd.RHS:
            continue
        if len(r.LHS) != len(rfd.LHS):
            continue
        for i in range(len(r.LHS)):
            if r.LHS[i] != rfd.LHS[i]:
                continue
        return cnt
        cnt += 1
    return -1


def pruneRFDs(rfds, phi, m):
    Attr_List = phi.LHS
    Attr_List.append(phi.RHS)
    for i in range(m):
        if i in Attr_List:
            continue
        kk_new = phi
        kk_new.LHS.append(i)
        res = contain(kk_new, rfds)
        if res != -1:
            rfds = rfds[:res] + rfds[res + 1:]
    return rfds


def specialize(phi, m):
    Attr_List = phi.LHS
    Attr_List.append(phi.RHS)
    res = []
    for i in range(m):
        if i in Attr_List:
            continue
        kk_new = phi
        kk_new.LHS.append(i)
        res.append(kk_new)
    return res


def insert_update_RFD(RFD_c, invalidRFDs, S, m):
    for level in range(min(2,m)):
        rfdsInvalidated = []
        for phi in RFD_c[level]:
            if not validate(phi, S):
                rfdsInvalidated.append(phi)
                invalidRFDs[level].append(phi)
            else:
                if level+1==m:
                    continue
                RFD_c[level + 1] = pruneRFDs(RFD_c[level + 1], phi, m)
        for phi in rfdsInvalidated:
            res = contain(phi, RFD_c[level])
            RFD_c[level] = RFD_c[level][:res] + RFD_c[level][res + 1:]
            spec = specialize(phi, m)
            for phi in spec:
                if contain(phi, RFD_c[level + 1]) == -1:
                    RFD_c[level + 1].append(phi)
    return RFD_c, invalidRFDs


def CalcEqual(x, y):
    if len(x) != len(y):
        return False
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True


def getTupleID(t, r):
    for i in range(len(r)):
        if CalcEqual(r[i], t):
            return i
    return -1


def getBits(S, l, r):
    res = S
    res = res & ((1 << r) - 1)
    res = res >> l
    return res


def concat(v1, v2, id):
    return v1 + (v2 << id)


def delete_update_vector(r, t, Phi, S):
    id = getTupleID(t, r)
    if id == -1:
        return S
    for B in range(r[0]):
        for k in range(len(r)):
            if k == id:
                continue
            v1 = getBits(S[B][k], 0, id - 1)
            v2 = getBits(S[B][k], id + 1, len(r) - 1)
            S[B][k] = concat(v1, v2, id)
    S = S[id - 1] + S[id + 1]
    return S


def generalize(phi):
    Attr_List = phi.LHS
    res = []
    for i in range(Attr_List):
        if i in Attr_List:
            continue
        kk_new = phi
        kk_new.LHS.remove(i)
        res.append(kk_new)
    return res


def containGeneralization(phi, RFDs):
    Attr_List = phi.LHS
    res = []
    for i in range(Attr_List):
        if i in Attr_List:
            continue
        kk_new = phi
        kk_new.LHS.remove(i)
        if contain(kk_new, RFDs):
            return False
    return True


def delete_update_RFD(RFD_c, invalidRFDs, S, m):
    if len(S) == 1:
        RFD_c = [[]] * len(RFD_c)
        invalidRFDs = [[]] * len(RFD_c)
        return RFD_c, invalidRFDs
    validRFD = []
    for level in range(m, 0, -1):
        for phi in invalidRFDs[level]:
            if validate(phi, S):
                invalidRFDs[level - 1] = invalidRFDs[level - 1] + generalize(phi)
                invalidRFDs[level].remove(phi)
                validRFD = validRFD.append(phi)
    for phi in validRFD:
        if not containGeneralization(phi, RFD_c):
            RFD_c.append(phi)
    return RFD_c, invalidRFDs


def insert(r, t, Phi, RFD_c, invalidRFDs, S, m):
    S = insert_update_vector(r, t, Phi, S)
    r.append(t)
    RFD_c, invalidRFDs = insert_update_RFD(RFD_c, invalidRFDs, S, m)
    return r, S, RFD_c, invalidRFDs


def delete(r, t, Phi, RFD_c, invalidRFDs, S, m):
    S = delete_update_vector(r, t, Phi, S)
    id = getTupleID(t, r)
    r = r[:id] + r[id + 1:]
    RFD_c, invalidRFDs = delete_update_RFD(RFD_c, invalidRFDs, S, m)
    return r, S, RFD_c, invalidRFDs


def CalcAll(m):
    res = [[]] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            res[0].append(FD([i], j))
    return res


def InDiBits(data, Phi):
    r = [data[0]]
    m = len(data[0])
    RFD_c = CalcAll(m)
    invalidRFDs = [[]] * len(data[0])
    S = [[1]] * m
    for i in range(1, len(data)):
        #print(i,len(data))
        r, S, RFD_c, invalidRFDs = insert(r, data[i], Phi, RFD_c, invalidRFDs, S, m)
    return RFD_c


from pre import pre_glass
from Score import get_Score


def FindMax(data, i):
    res = 0
    for r in data:
        res = max(res, r[i])
    return res


def FindMin(data, i):
    res = 99999999
    for r in data:
        res = min(res, r[i])
    return res


if __name__ == "__main__":
    Data, name_list = pre_glass()
    import time

    start = time.time()
    Data1 = Data[:int(len(Data)-1000)]
    Data2 = Data[(int(len(Data))-1000):]
    x=int(len(Data1)/50)
    Phi = []
    for i in range(len(Data[0])):
        Phi.append(FindMax(Data, i)*0.7 + FindMin(Data, i)*0.3)
    ans=[]
    for i in range(0,len(Data1),x):
        l=i
        r=min(len(Data1),l+200)
        print(l,r)
        ans += InDiBits(Data1[l:r], Phi)
    end = time.time()
    print("No.1")
    print("Glass:", Data.shape)
    print("Time:", (end - start))
    #print(ans)
    res=[]
    for i in ans:
        for j in i:
            res.append(RFD(j,Phi))
    print("Cnt:", len(res))
    print("Score:", get_Score(Data2,res))
    # pli.print()
    '''for i in ans:
        print(i.LHS,i.RHS)'''
    # print(len(pli.calc([0,14])[0])+len(pli.calc([0,14])[1]))
    '''data = [["Alex", "Smith", "m", 55302, "Brighton"], ["John", "Kramer", "m", 55301, "Brighton"],
            ["Lindsay", "Miller", "f", 55301, "Rapid Falls"], ["Alex", "Smith", "m", 55302, "Brighton"],
            ["Alex", "Miller", "f", 55301, "Brighton"]]
    P = Pyro(data, 0.1, 0.1)
    print(len(P))'''
