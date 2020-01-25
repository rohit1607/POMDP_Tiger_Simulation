from value_iteration import *
import datetime
import sys
import os

"""
INPUT VARS - make these command line args later
"""

save_figs = True
horizon = 5


time_str = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
pic_count = 0

def visualise_simulation(state, alpha_set, cur_bel, action=None, obs=None):
    tiger = plt.imread('tiger.png', 0)
    diamond = plt.imread('diamond.png', 0)

    agent = plt.imread('agent.png', 0)

    obs_left = plt.imread('obs_left.png', 0)
    obs_right = plt.imread('obs_right.png', 0)

    open_left = plt.imread('open_left.png', 0)
    open_right = plt.imread('open_right.png', 0)
    listen = plt.imread('listen.png', 0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    left_state_pos = [0.1, 0.6, 0.2, 0.2]
    right_state_pos = [0.3, 0.6, 0.2, 0.2]
    agent_pos = [0.2, 0.2, 0.2, 0.2]
    left_action_pos = [0.15, 0.4, 0.15, 0.15]
    right_action_pos = [0.25, 0.4, 0.15, 0.15]
    listen_pos = [0.225, 0.4, 0.15, 0.15]

    left_obs_pos = [0.1, 0.3, 0.2, 0.2]
    right_obs_pos = [0.3, 0.3, 0.2, 0.2]

    agent_ax = fig.add_axes(agent_pos, zorder=1)
    agent_ax.imshow(agent)

    if state == 0:
        tig_ax = fig.add_axes(left_state_pos, zorder=1)
        tig_ax.imshow(tiger)
        daim_ax = fig.add_axes(right_state_pos, zorder=1)
        daim_ax.imshow(diamond)
    else:
        tig_ax = fig.add_axes(right_state_pos, zorder=1)
        tig_ax.imshow(tiger)
        daim_ax = fig.add_axes(left_state_pos, zorder=1)
        daim_ax.imshow(diamond)

    if action == 0:
        action_ax = fig.add_axes(left_action_pos, zorder=1)
        action_ax.imshow(open_left)
        action_ax.axis('off')
    elif action == 1:
        action_ax = fig.add_axes(right_action_pos, zorder=1)
        action_ax.imshow(open_right)
        action_ax.axis('off')
    elif action == 2:
        action_ax = fig.add_axes(listen_pos, zorder=1)
        action_ax.imshow(listen)
        action_ax.axis('off')

    if obs == 0:
        obs_ax = fig.add_axes(left_obs_pos, zorder=1)
        obs_ax.imshow(obs_left)
        obs_ax.axis('off')
    elif obs == 1:
        obs_ax = fig.add_axes(right_obs_pos, zorder=1)
        obs_ax.imshow(obs_right)
        obs_ax.axis('off')

    tig_ax.axis('off')
    daim_ax.axis('off')
    agent_ax.axis('off')
    ax[0].tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([-3, 14])
    ax[1].axhline(y=0, xmin=0, xmax=1, lw=6, alpha=0.5, zorder=-1)
    ax[0].axhline(y=0.65, xmin=0, xmax=0.45, color='k', lw=10, zorder=-1)
    ax[0].axhline(y=0.65, xmin=0.55, xmax=1, color='k', lw=10, zorder=-1)

    ax[1].set_xlabel('b(Tiger on Left)', fontsize=25)
    ax[1].set_ylabel('V(b)', fontsize=25)
    ax[1].tick_params(axis='both', which='major', labelsize=20)

    action_map = ['Open Left Door', 'Open Right Door', 'Listen']

    # plot nth(input) horizon value funtion
    num_points = 501
    b = np.linspace(0, 1, num_points)
    vals = np.zeros((num_points,))
    old_col = None
    for i in range(num_points):
        temp = find_value_and_maxAlpha(generate_belief(b[i]), alpha_set)
        vals[i] = temp[0]
        col = action_colour(temp[2])
        if col != old_col:
            ax[1].scatter(b[i], vals[i], color=col, label=action_map[temp[2]])
        else:
            ax[1].scatter(b[i], vals[i], color=col)

        old_col = col

    # display current belief
    b = cur_bel[0]
    string = 'bel = ' + str(round(b, 3))
    ax[1].scatter(b, 0, s=1000, color='c', label='Current Belief')
    ax[0].text(0.4, 0.05, string, bbox=dict(facecolor='c', alpha=0.5), fontsize=25)

    # state annotations on second axes
    string_left = 'Tiger on Right'
    string_right = 'Tiger on Left'
    ax[1].text(-0.15, -4.5, string_left, bbox=dict(facecolor='y', alpha=0.5), fontsize=15)
    ax[1].text(0.95, -4.5, string_right, bbox=dict(facecolor='y', alpha=0.5), fontsize=15)

    # legend
    ax[1].legend(loc='upper center', fontsize='xx-large')

    global pic_count
    pic_count+=1
    if save_figs:
        dir_name = get_dir_name(time_str)
        filename = 'Tiger_sim__' + '_' + str(pic_count)+ '.png'
        plt.savefig(dir_name +'/'+ filename)
    else:
        plt.show()

    return



def get_dir_name(time_string):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, time_string)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        print(final_directory)
    return final_directory

def simulate_an_episode(tiger, horizon, learned_gamma=None, savefig = False):

    # learned_gamma = solver_with_Prune(tiger, horizon=horizon)
    Returns = 0
    tiger.reset_world()
    print("state= ", tiger.cur_state)
    pic_count = 0
    for h in reversed(range(horizon)):
        print("-----horizon ", h, "---------------")
        max_val, max_vec, best_action = find_value_and_maxAlpha(tiger.cur_belief, learned_gamma[h])
        print("best action= ", best_action)
        visualise_simulation(tiger.cur_state, learned_gamma[h], tiger.cur_belief, action = best_action, obs = None)

        obs, reward = tiger.make_a_move(best_action)
        visualise_simulation(tiger.cur_state, learned_gamma[h], tiger.cur_belief, action = None, obs = obs )

        print("obs= ", obs)
        print("rew= ", reward)

        Returns += reward

        if best_action < 2: # if either door is opened, end episode
            print("DOOR OPENED")
            break

        new_b = tiger.belief_update(tiger.cur_belief, best_action, obs)
        print("new_b= ", new_b)
        tiger.cur_belief = new_b

    return Returns


tiger = Tiger_world()
gamma = solver_with_Prune(tiger, horizon)
returns = simulate_an_episode(tiger, horizon, learned_gamma=gamma)
print(returns)




"""
if __name__ == "__main__":
    input_list = sys.argv

    tiger = Tiger_world()

    horizon = int(input_list[0])
    gamma = solver_with_Prune(tiger, horizon)
    returns = simulate_an_episode(tiger, horizon, learned_gamma=gamma)
    print(returns)
"""