import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', 50)

TAXI = 'Taxi'
FOREST = 'Forest'

def plot_vi_gamma(vi1, vi2, gammas1, gammas2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for vi_stat, gamma in zip(vi1, gammas1):
        df = pd.DataFrame(vi_stat)
        ax1.plot(df.Iteration, df.Error, label=round(gamma, 2))

    for vi_stat, gamma in zip(vi2, gammas2):
        df = pd.DataFrame(vi_stat)
        ax2.plot(df.Iteration, df.Error, label=round(gamma, 2))

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different γ'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different γ'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'vi_gamma.png')

def plot_pi_gamma(pi1, pi2, gammas1, gammas2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for vi_stat, gamma in zip(pi1, gammas1):
        df = pd.DataFrame(vi_stat)
        ax1.plot(df.Iteration, df.Error, label=round(gamma, 2))

    for vi_stat, gamma in zip(pi2, gammas2):
        df = pd.DataFrame(vi_stat)
        ax2.plot(df.Iteration, df.Error, label=round(gamma, 2))

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different γ'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different γ'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'pi_gamma.png')

def plot_pi_iter_lp(vi_iter1, vi_lp1, vi_iter2, vi_lp2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    vi_iter1 = pd.DataFrame(vi_iter1)
    vi_iter2 = pd.DataFrame(vi_iter2)
    vi_lp1 = pd.DataFrame(vi_lp1)
    vi_lp2 = pd.DataFrame(vi_lp2)

    ax1.plot(vi_iter1.Time, vi_iter1.Error, label='Iterative')
    ax1.plot(vi_lp1.Time, vi_lp1.Error, label='Linear Program')

    ax2.plot(vi_iter2.Time, vi_iter2.Error, label='Iterative')
    ax2.plot(vi_lp2.Time, vi_lp2.Error, label='Linear Program')

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Time vs Error'.format(TAXI))
    ax2.set_title('{} - Time vs Error'.format(FOREST))

    ax1.set_xlabel('time')
    ax1.set_ylabel('error')

    ax2.set_xlabel('time')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'pi_iter_lp.png')

def plot_vi_pi(vi1, pi1, vi2, pi2):
    vi1 = pd.DataFrame(vi1)
    pi1 = pd.DataFrame(pi1)
    # pi1.set_value(0, pi1.Error, pi1.Error[1])
    vi2 = pd.DataFrame(vi2)
    pi2 = pd.DataFrame(pi2)
    # pi2.set_value(0, pi2.Error, pi2.Error[1])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    def plot_vi_ext_sub(vi, pi, a1, a2, prob):
        a1.plot(vi.Iteration, vi.Error, color='g', label='VI')
        a1.plot(pi.Iteration, pi.Error, color='b', label='PI')
        a1.set_title('{} - Iteration vs Error'.format(prob))
        a1.set_xlabel('number of iterations')
        a1.set_ylabel('error')

        a2.plot(vi.Time, vi.Error, color='g', label='VI')
        a2.plot(pi.Time, pi.Error, color='b', label='PI')
        a2.set_title('{} - Time vs Error'.format(prob))
        a2.set_xlabel('time')
        a2.set_ylabel('error')

        a1.legend(loc='best')
        a2.legend(loc='best')

    plot_vi_ext_sub(vi1, pi1, ax1, ax3, TAXI)
    plot_vi_ext_sub(vi2, pi2, ax2, ax4, FOREST)

    fig.tight_layout()
    fig.savefig('figures/' + 'vi_pi.png')



def plot_vi_epsilon(vi1, vi2, epsilon1, epsilon2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for vi_stat, epsilon in zip(vi1, epsilon1):
        df = pd.DataFrame(vi_stat)
        ax1.plot(df.Iteration, df.Time, label=epsilon)

    for vi_stat, epsilon in zip(vi2, epsilon2):
        df = pd.DataFrame(vi_stat)
        ax2.plot(df.Iteration, df.Time, label=epsilon)

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Time with different ε'.format(TAXI))
    ax2.set_title('{} - Iteration vs Time with different ε'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('time')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('time')

    fig.tight_layout()
    fig.savefig('figures/' + 'vi_epsilon.png')

def plot_vi_extended(vi_reg1, vi_gs1, vi_reg2, vi_gs2):
    vi_reg1 = pd.DataFrame(vi_reg1)
    vi_gs1 = pd.DataFrame(vi_gs1)
    vi_reg2 = pd.DataFrame(vi_reg2)
    vi_gs2 = pd.DataFrame(vi_gs2)
    print(vi_reg1.head())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    def plot_vi_ext_sub(vi_reg, vi_gs, a1, a2, prob):
        a1.plot(vi_reg.Iteration, vi_reg.Error, color='g', label='VI')
        a1.plot(vi_gs.Iteration, vi_gs.Error, color='b', label='GS')
        a1.set_title('{} - Iteration vs Error'.format(prob))
        a1.set_xlabel('number of iterations')
        a1.set_ylabel('error')

        a2.plot(vi_reg.Iteration, vi_reg.Time, color='g', label='VI')
        a2.plot(vi_gs.Iteration, vi_gs.Time, color='b', label='GS')
        a2.set_title('{} - Iteration vs Time'.format(prob))
        a2.set_xlabel('number of iterations')
        a2.set_ylabel('time')

    plot_vi_ext_sub(vi_reg1, vi_gs1, ax1, ax2, TAXI)
    plot_vi_ext_sub(vi_reg2, vi_gs2, ax3, ax4, FOREST)

    fig.tight_layout()
    fig.savefig('figures/' + 'vi_ext.png')

def plot_q_decay(q1, decay1, q2, decay2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for q, decay in zip(q1, decay1):
        df = pd.DataFrame(q)
        ax1.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)
        # ax1.plot(df.Iteration[::100], df.Error[::100], label=decay)

    for q, decay in zip(q2, decay2):
        df = pd.DataFrame(q)
        # ax2.plot(df.Iteration[::100], df.Error[::100], label=decay)
        ax2.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different ε decay rate'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different ε decay rate'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'q_decay.png')

def plot_q_learning_decay(q1, decay1, q2, decay2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for q, decay in zip(q1, decay1):
        df = pd.DataFrame(q)
        ax1.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)

    for q, decay in zip(q2, decay2):
        df = pd.DataFrame(q)
        ax2.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different ɑ decay rate'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different ɑ decay rate'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'q_learning_rate.png')

def plot_q_learning_init(q1, decay1, q2, decay2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for q, decay in zip(q1, decay1):
        df = pd.DataFrame(q)
        ax1.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)

    for q, decay in zip(q2, decay2):
        df = pd.DataFrame(q)
        ax2.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=decay)

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different init ɑ'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different init ɑ'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'q_learning_rate_init.png')

def plot_q_gamma(q1, decay1, q2, decay2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    for q, gamma in zip(q1, decay1):
        df = pd.DataFrame(q)
        ax1.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=round(gamma, 2))

    for q, gamma in zip(q2, decay2):
        df = pd.DataFrame(q)
        ax2.plot(df.Iteration[:500][::10], df.Error[:500][::10], label=round(gamma, 2))

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_title('{} - Iteration vs Error with different γ'.format(TAXI))
    ax2.set_title('{} - Iteration vs Error with different γ'.format(FOREST))

    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('error')

    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('error')

    fig.tight_layout()
    fig.savefig('figures/' + 'q_gamma.png')

def plot_vi_pi_q(vi1, pi1, vi2, pi2, q1, q2):
    vi1 = pd.DataFrame(vi1)
    pi1 = pd.DataFrame(pi1)
    # pi1.set_value(0, pi1.Error, pi1.Error[1])
    vi2 = pd.DataFrame(vi2)
    pi2 = pd.DataFrame(pi2)
    # pi2.set_value(0, pi2.Error, pi2.Error[1])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    def plot_vi_ext_sub(vi, pi, a1, a2, prob):
        a1.plot(vi.Iteration, vi.Error, color='g', label='VI')
        a1.plot(pi.Iteration, pi.Error, color='b', label='PI')
        a1.set_title('{} - Iteration vs Error'.format(prob))
        a1.set_xlabel('number of iterations')
        a1.set_ylabel('error')

        a2.plot(vi.Time, vi.Error, color='g', label='VI')
        a2.plot(pi.Time, pi.Error, color='b', label='PI')
        a2.set_title('{} - Time vs Error'.format(prob))
        a2.set_xlabel('time')
        a2.set_ylabel('error')

        a1.legend(loc='best')
        a2.legend(loc='best')

    plot_vi_ext_sub(vi1, pi1, ax1, ax3, TAXI)
    plot_vi_ext_sub(vi2, pi2, ax2, ax4, FOREST)

    fig.tight_layout()
    fig.savefig('figures/' + 'vi_pi.png')