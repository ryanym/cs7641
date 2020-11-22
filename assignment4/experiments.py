import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.mdp as mdp
import hiive.mdptoolbox.example as example
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import *

plt.style.use('seaborn-whitegrid')
np.set_printoptions(precision=3)

max_iter = 10000

def build_matrix(environment, n_states, n_actions):
    """
    Convert openai discrete environment to Probability matrix and Reward matrix
    :param environment:
    :param n_states:
    :param n_actions:
    :return:
    """
    reward_matrix = np.zeros((n_states, n_actions))
    probability_matrix = np.zeros((n_actions, n_states, n_states))

    for state in range(n_states):
        for action in range(n_actions):
            for data_list in environment.env.P[state][action]:
                prob, next_state, reward, done = data_list
                reward_matrix[state, action] = reward
                probability_matrix[action, state, next_state] = prob
                probability_matrix[action, state, :] = probability_matrix[action, state, :] / \
                                                       np.sum(probability_matrix[action, state, :])

    return probability_matrix, reward_matrix


def vi_experiment_epsilon(P, R):
    """
    experiments on effectiveness of epsilon on # of iteration and time
    :param P:
    :param R:
    :param problem:
    :return:
    """

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    vi_stats = []
    for epsilon in epsilons:
        vi = mdp.ValueIteration(P, R, 0.65, epsilon=epsilon)
        vi_stat = vi.run()
        vi_stats.append(vi_stat)

    return vi_stats, epsilons


def vi_experiment_gamma(P, R):
    """
    experiments on effectiveness of gamma on # of iterations and rewards
    :param P:
    :param R:
    :param problem:
    :return:
    """
    gammas = np.linspace(0.05, 0.95, 10)
    vi_stats = []
    for gamma in gammas:
        vi = mdp.ValueIteration(P, R, gamma, epsilon=0.001)
        vi_stat = vi.run()
        vi_stats.append(vi_stat)

    return vi_stats, gammas

def vi_experiment_extended(P, R):
    """
    compares regular VI vs Gauss-Seidel Value iteration
    :param P:
    :param R:
    :return:
    """
    vi_reg = mdp.ValueIteration(P, R, 0.65)
    vi_reg_stat = vi_reg.run()
    print(vi_reg.iter, vi_reg.time)

    vi_gs = mdp.ValueIterationGS(P, R, 0.65, max_iter=1000)
    vi_gs_stat = vi_gs.run()
    print(vi_gs.iter, vi_gs.time)

    return vi_reg_stat, vi_gs_stat

def pi_experiment_gamma(P, R):
    """
    experiments on effectiveness of gamma on # of iterations and rewards

    :param P:
    :param R:
    :return:
    """
    gammas = np.linspace(0.05, 0.95, 10)
    vi_stats = []
    for gamma in gammas:
        vi = mdp.PolicyIteration(P, R, gamma)
        vi_stat = vi.run()
        vi_stats.append(vi_stat)

    return vi_stats, gammas

def pi_experiment_iter_linear(P, R):
    vi_iter = mdp.PolicyIteration(P, R, 0.6, eval_type=1).run()
    vi_linear = mdp.PolicyIteration(P, R, 0.6, eval_type=0).run()

    print('Iterative stats: ', vi_iter[-1])
    print('LP stats: ', vi_linear[-1])

    return vi_iter, vi_linear

def vi_pi_comp(P, R):
    vi = mdp.ValueIteration(P, R, 0.60, epsilon=0.001).run()
    pi = mdp.PolicyIteration(P, R, 0.60, eval_type=1).run()

    return vi, pi

def q_decay_rate(P, R):
    decays = [0.99, 0.9, 0.7, 0.5]
    q_stats = []
    for decay in decays:
        q = mdp.QLearning(P, R, 0.9, alpha=0.01, alpha_decay=0.99, epsilon_decay=decay, n_iter=max_iter).run()
        q_stats.append(q)

    return q_stats, decays

def q_learing_rate_decay(P, R):
    decays = [0.99, 0.9, 0.7, 0.5]
    q_stats = []
    for decay in decays:
        q = mdp.QLearning(P, R, 0.9, alpha=0.5, alpha_decay=decay, n_iter=max_iter).run()
        q_stats.append(q)

    return q_stats, decays

def q_learing_rate_init(P, R):
    rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    q_stats = []
    for rate in rates:
        q = mdp.QLearning(P, R, 0.9, alpha=rate,epsilon_decay=1, n_iter=max_iter).run()
        q_stats.append(q)

    return q_stats, rates

def q_gamma(P, R):
    gammas = np.linspace(0.05, 0.95, 3)
    q_stats = []
    for gamma in gammas:
        q = mdp.QLearning(P, R, gamma, alpha=0.2, alpha_decay=0.99, epsilon_decay=0.99, n_iter=max_iter).run()
        q_stats.append(q)

    return q_stats, gammas

def vi_pi_q_comp(P, R):
    vi = mdp.ValueIteration(P, R, 0.60, epsilon=0.001).run()
    pi = mdp.PolicyIteration(P, R, 0.60, eval_type=1).run()
    q = mdp.QLearning(P, R, 0.6, alpha=0.2).run()
    return vi, pi, q

P1, R1 = example.forest(10000, p=0.5)
# P1, R1 = example.forest(10000)

env = gym.make('Taxi-v3')
states = env.observation_space.n
actions = env.action_space.n
P2, R2 = build_matrix(env, states, actions)

# VI Gamma
vi_gamma_forest, gamma_forest = vi_experiment_gamma(P1, R1)
vi_gamma_taxi, gamma_taxi = vi_experiment_gamma(P2, R2)
print(vi_gamma_taxi[-2][-1], vi_gamma_forest[-2][-1])
plot_vi_gamma(vi_gamma_taxi, vi_gamma_forest, gamma_taxi, gamma_forest)

# VI Epsilon
vi_epsilon_forest, epsilon_forest = vi_experiment_epsilon(P1, R1)
vi_epsilon_taxi, epsilon_taxi = vi_experiment_epsilon(P2, R2)

plot_vi_epsilon(vi_epsilon_taxi, vi_epsilon_forest, epsilon_taxi, epsilon_forest)

# VI ext
vi_reg_forest, vi_gs_forest = vi_experiment_extended(P1, R1)
vi_reg_taxi, vi_gs_taxi = vi_experiment_extended(P2, R2)
plot_vi_extended(vi_reg_taxi, vi_gs_taxi, vi_reg_forest, vi_gs_forest)

# PI Gamma
pi_gamma_forest, gamma_forest = vi_experiment_gamma(P1, R1)
pi_gamma_taxi, gamma_taxi = vi_experiment_gamma(P2, R2)

plot_pi_gamma(pi_gamma_taxi, pi_gamma_forest, gamma_taxi, gamma_forest)

# PI iter vs lp
vi_iter1, vi_lp1 = pi_experiment_iter_linear(P2, R2)
vi_iter2, vi_lp2 = pi_experiment_iter_linear(P1, R1)

plot_pi_iter_lp(vi_iter1, vi_lp1, vi_iter2, vi_lp2)

# VI vs PI
vi1, pi1 = vi_pi_comp(P2, R2)
vi2, pi2 = vi_pi_comp(P1, R1)

plot_vi_pi(vi1, pi1, vi2, pi2)


# Q Learning learning rate
q1, decay1 = q_learing_rate_decay(P2, R2)
q2, decay2 = q_learing_rate_decay(P1, R1)
plot_q_learning_decay(q1, decay1, q2, decay2)

# Q Learning learning rate init
q1, decay1 = q_learing_rate_init(P2, R2)
q2, decay2 = q_learing_rate_init(P1, R1)
plot_q_learning_init(q1, decay1, q2, decay2)

# Q Learning epsilon rate
q1, decay1 = q_decay_rate(P2, R2)
q2, decay2 = q_decay_rate(P1, R1)
plot_q_decay(q1, decay1, q2, decay2)

# Q Learning gamma
q1, gamma1 = q_gamma(P2, R2)
q2, gamma2 = q_gamma(P1, R1)
plot_q_gamma(q1, gamma1, q2, gamma2)

# All comp
vi1, pi1, q1 = vi_pi_q_comp(P2, R2)
vi2, pi2, q2 = vi_pi_q_comp(P1, R1)
vi1 = pd.DataFrame(vi1)
vi2 = pd.DataFrame(vi2)
pi1 = pd.DataFrame(pi1)
pi2 = pd.DataFrame(pi2)
q1 = pd.DataFrame(q1)
q2 = pd.DataFrame(q2)