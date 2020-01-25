from Tiger_env import Tiger_world
import numpy as np
import matplotlib.pyplot as plt
from Prune import White_Lark_Pruning


def init_gamma(tiger):
    #set of alpha vectors
    #list of sets. gamma[0] will be set of vectors when 1 decision is to be made
    #each alpha_set is list of tuples (vec, root_action)
    gamma = []
    alpha_set = []

    # root_action = []
    R = tiger.get_reward_mat()
    for action in tiger.actions:

        vec = R[:,action]
        # print("test, ",vec, vec.shape)
        alpha_set.append((vec,action))

    gamma.append(alpha_set)
    return gamma


# find_maxAlphas_and_beliefRegions(gamma[0])
def find_value_and_maxAlpha(b, alpha_set):
    max_val = -100000
    shape = alpha_set[0][0].shape
    max_vec = np.zeros(shape)
    best_action = None
    for vec, action in alpha_set:
        # print("vec and action",vec, type(vec), vec.shape, action)
        value = np.dot(vec, b)
        if value >= max_val:
            max_val = value
            max_vec = vec
            best_action = action

    return max_val, max_vec, best_action

def generate_belief(b):
    return np.array([b,1.0-b])

def action_colour(a):
    if a == 0:
        return 'r'
    elif a == 1:
        return 'b'
    elif a == 2:
        return 'g'

def test_and_see_Value_function(gamma, time_horizon = 0):
    b=np.linspace(0,1,101)
    vals = np.zeros((101,))
    for i in range(101):
        temp = find_value_and_maxAlpha(generate_belief(b[i]), gamma[time_horizon])
        vals[i] = temp[0]
        col = action_colour(temp[2])
        plt.scatter(b[i],vals[i], color = col)
    plt.show()


def expected_future_return(tiger, s, a, V_o1, V_o2):
    """returns the second term of of the value funcn for a policy at t
        Policy at t is defined by root action a, and subtrees corresponding to policies at t-1"""
    T = tiger.get_trans_mat()
    O = tiger.get_obs_mat()
    sum_over_states = 0
    for s_dash in tiger.states:
        sum_over_states += T[s][a][s_dash] * ( O[0][a][s_dash]*V_o1[s_dash]  +  O[1][a][s_dash]*V_o2[s_dash] )
    return sum_over_states


def generate_next_alphaSet(tiger, gamma):
    current_alpha_set = gamma[-1]
    # print("test_gamma-1", current_alpha_set)
    new_alpha_set = []

    R = tiger.get_reward_mat()

    # create an alpha vector in the gamma[next] for each possible policy graph
    # the 3 loops together make one policy graph at time t+1

    for a in tiger.actions:
        """Two inner loops because 2 possible obs."""
        for V_o1, sub_action1 in current_alpha_set:
            for V_o2, sub_action2 in current_alpha_set:
                new_vec = np.array([0] * tiger.num_states)
                for s in tiger.states:

                    exp_fut_returns = expected_future_return(tiger, s, a, V_o1, V_o2)
                    new_vec[s] = R[s][a] + exp_fut_returns

                new_alpha_set.append((new_vec, a))

    gamma.append(new_alpha_set)
    return gamma


def vanilla_solver(tiger, horizon =2):
    gamma = init_gamma(tiger)
    for i in range(2):
        gamma = generate_next_alphaSet(tiger, gamma)

    return gamma


def solver_with_Prune(tiger, horizon = 1):
    gamma = init_gamma(tiger)
    for i in range(horizon):
        gamma = generate_next_alphaSet(tiger, gamma)
        # print("gamma[-1], len =",gamma[-1], len(gamma[-1]))
        D = White_Lark_Pruning(gamma[-1])
        gamma[-1] = D
    return gamma



def main(solver, horizon):
    tiger = Tiger_world()
    gamma = None

    if solver == 'vanilla':
        gamma = vanilla_solver(tiger, horizon=horizon)
    elif solver == 'with_Prune':
        gamma = solver_with_Prune(tiger, horizon=horizon)

    for h in range(horizon):
        test_and_see_Value_function(gamma, time_horizon = h)



