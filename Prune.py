import numpy as np
from scipy.optimize import linprog

num_states = 2

def White_Lark_Pruning(W):
    D = []
    count = 0
    while len(W) != 0:
        count+=1

        idx = np.random.choice(len(W))
        vec = W[idx][0]
        flag = False
        for other_vec, act in D:
            vec_is_dominated = True
            for s in range(vec.shape[0]):
                if vec[s]>other_vec[s]:
                    vec_is_dominated = False
                    break
            #vec_is_dominated remains True if vec[s] <= other_vec[s] for all s
            if vec_is_dominated == True:
                W.pop(idx)
                flag = True
                break
        if flag == False:
            b1 = FindBeliefStd(D, vec)
            if b1 == None:
                W.pop(idx)

            else:
                # MUST KEEP TRACK OF IDX/ROOT ACTION
                b = np.array([b1, 1.0 -b1])
                vec_tup_id = BestVector(b,W)
                D.append(W[vec_tup_id])

                W.pop(vec_tup_id)

    return D



def BestVector(b, W):
    """

    :param b: belief vector nparray [b1, 1-b1]
    :param W: alpha_set [ (vec, root_action), ()...]
    :return: id of best vector tuple
    """
    # count = 0
    max_val = - float('inf')
    best_vec_tuple = None
    best_id = None
    for id in range(len(W)):

        vec_tuple = W[id]
        val = np.dot(b,vec_tuple[0])
        if val == max_val:
            best_id = lexMax(W, best_id, id)
        if val > max_val:
            max_val = val
            best_id = id

    return best_id


def lexMax(W, best_id, id):
    """
    :param W
    :param best_id: list idx of best_vec_tuple
    :param id: list idx of vec
    :return: id of the better vector
    """
    best_vec_tup = W[best_id]
    vec_tup = W[id]
    for s in range(num_states):
        if best_vec_tup[0][s]>vec_tup[0][s]:
            return best_id
        if best_vec_tup[0][s]<vec_tup[0][s]:
            return id
    return id

def FindBeliefStd(D, vec):
    """

    :param D: alpha_set (pruned result): [ (vec, root_action), ()...]
    :param vec: just a vector
    :return: belief of state1 that maximises value diff between vec and all d in D
    """
    if len(D) == 0:
        a = np.random.rand()
        return a

    c = np.array([0, -1])
    Aub = np.zeros((len(D),num_states))
    b_ub = np.zeros((len(D),1))
    x0_bounds = (0, 1)
    x1_bounds = (None, None)
    Aub[:,1]=1.0
    for i in range(len(D)):
        d, act = D[i]
        r = vec - d
        Aub[i,0] = r[1]-r[0]
        b_ub[i,0] = r[1]


    res = linprog(c, A_ub=Aub, b_ub=b_ub, bounds=[x0_bounds, x1_bounds])
    b1, d = res.x
    # print("LP sol= ",b1, d)

    if d>=0:
        return b1
    else:
        return None

"""
c = [-1, 4]
c= np.array(c)
A = [[-3, 1], [1, 2]]
A= np.array(A)
b = [6, 4]
b= np.array(b)
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
print(res)
"""

# # Check FindBeliefStd :
#
# v1 = np.array([6, 3])
# v2 = np.array([5, 5])
# v3 = np.array([2, 7])
#
# w1 = np.array([1, 2])
# w2 = np.array([10,10])
#
# D = [(v1, 0), (v2, 1), (v3, 2)]
#
# bel = FindBeliefStd(D, w2)
# print(bel)


#
# # Check White_Lark_Pruning
# v1 = np.array([6, 3])
# v2 = np.array([5, 5])
# v3 = np.array([2, 7])
#
# w1 = np.array([1, 2])
# w2 = np.array([0,1])
# w3 = np.array([2.5, 0])
# #
# W = [(v1, 0), (v2, 1), (v3, 2), (w1, 2), (w2, 2), (w3, 2)]
# # W.remove((v1,0))
# # print(W)
#
# D = White_Lark_Pruning(W)
# print(D)

"""
def test_policy_layout():
    tiger = Tiger_world()

    vlist = ['a','b', 'c']
    count =0
    for a in tiger.actions:
        for v1 in vlist:
            for v2 in vlist:
                print(a, 'o1' ,v1, 'o2' ,v2)
                count +=1

    print(count)
"""