import numpy as np

class Tiger_world:

    def __init__(self):
        self.states = [0, 1]
        self.actions = [0, 1, 2]
        self.obs = [0, 1]

        # self.states = ['Sl', 'Sr']
        # self.actions = ['left', 'right', 'listen']
        # self.obs = ['Tl', 'Tr']

        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.num_obs = len(self.obs)

        #initial state
        self.cur_state = np.random.binomial(size=1, n=1, p= 0.5)

        self.cur_belief = np.array([0.5, 0.5])

        self.get_reward_mat()
        self.get_obs_mat()
        self.get_trans_mat()


    def reset_world(self):
        self.cur_state = np.random.binomial(size=1, n=1, p= 0.5)
        self.cur_belief = np.array([0.5, 0.5])


    def get_trans_mat(self):
        """
        |S| x |A| x |S|
        """
        self.P_trans = np.array([[[0.5, 0.5],
                                  [0.5, 0.5],
                                  [1, 0]],

                                 [[0.5, 0.5],
                                  [0.5, 0.5],
                                  [0, 1]]])
        return self.P_trans


    def get_obs_mat(self):
        """
        |O| x |A| x |S|
        """
        self.P_obs = np.array([ [[0.5, 0.5],
                                   [0.5, 0.5],
                                   [0.85, 0.15]],

                                  [[0.5, 0.5],
                                   [0.5, 0.5],
                                   [0.15, 0.85]]   ])
        return self.P_obs


    def get_reward_mat(self):
        """
        |S| x |A|
        """
        self.rewards = np.asarray([[-100, 10, -1],
                                   [10, -100, -1]])
        return self.rewards


    def belief_update(self, old_belief, action, obs):

        prior_sl, prior_sr = old_belief

        # unless action is listen, reset the probabilities to 0.5 (by problem definition)
        post_sl = 0.5
        post_sr = 0.5

        if action == 2:  # if action is listen
            P_obs_given_sl = self.P_obs[obs][action][0]
            P_obs_given_sr = 1 - P_obs_given_sl
            # print("check 0.85 =", P_obs_given_sl)

            post_sl = P_obs_given_sl * prior_sl
            post_sr = P_obs_given_sr * prior_sr
            normalization = post_sl + post_sr

            post_sl = post_sl / normalization
            post_sr = post_sr / normalization

        new_belief = np.array([post_sl, post_sr])
        return new_belief


    def make_a_move(self, action):
        obs = None
        s = self.cur_state
        reward = self.rewards[s, action]


        if action == 0:
            print("Opened left door")
            if s == 0:
                print("Wrong")
            else:
                print("Correct")

        elif action == 1:
            print("Opened right door")
            if s == 1:
                print("Wrong")
            else :
                print("Correct")

        elif action == 2:
            P_Tl = self.P_obs[0, action, s]
            rand = np.random.rand()

            if P_Tl > rand:
                obs = 0
            else:
                obs = 1

        return obs, reward


env = Tiger_world()
a=env.get_reward_mat()
print(a[0][1])




