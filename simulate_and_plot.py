import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt 
 
# import the class that implements the Q-Learning algorithm
from function_edited_by_Zelamir import Q_Learning
 
#env=gym.make('CartPole-v1',render_mode='human')
env=gym.make('CartPole-v1')
(state,_)=env.reset()
 
# here define the parameters for state discretization
upperBounds=env.observation_space.high
lowerBounds=env.observation_space.low
cartVelocityMin=-3
cartVelocityMax=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
upperBounds[1]=cartVelocityMax
upperBounds[3]=poleAngleVelocityMax
lowerBounds[1]=cartVelocityMin
lowerBounds[3]=poleAngleVelocityMin
 
numberOfBinsPosition=10
numberOfBinsVelocity=10
numberOfBinsAngle=10
numberOfBinsAngleVelocity=10
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]
 
# define the parameters
alpha=0.15
gamma=1.0
# epsilon=0.2
numberEpisodes=15000

reward_list = []
run_length = []
lengths_list = [] 
lengths_by_alpha = {}
lengths_by_gamma = {}
lengths_by_binsize = {}
length = []

eps_values = [0.2 + 0.05*i for i in range(5)]  # [0.2, 0.3, 0.4]

########### Alpha/Epsilon Simulation & Plots ###############
alp_values = [0.10 + 0.05*i for i in range(2)] 

for alpha in alp_values:
    alp_str = f"{alpha:.2f}" 
    lengths_by_alpha[alp_str] = [] 
    path = f'Alpha/Alp{alp_str}'
    for epsilon in eps_values:

        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...

        r_arr_tit = f"R_b10_a{alp_str}_g1.0_e{eps_str}.npy"
        full_r_path = os.path.join(path, r_arr_tit)
        rewards = np.load(full_r_path)
        reward_list.append((epsilon, rewards))

        q_tab_tit = f"Q_b10_a{alp_str}_g1.0_e{eps_str}.npy"
        full_q_path = os.path.join(path, q_tab_tit)
        q_tab = np.load(full_q_path)

        env_local1 = gym.make('CartPole-v1')
        Q1 = Q_Learning(env=env_local1, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        Q1.Qmatrix = q_tab

        run_length = []
        for run in range(100):
            rewards, used_env = Q1.simulateLearnedStrategyLoaded(env=None, render_delay=0)
            used_env.close()
            run_length.append(len(rewards)) 

        run_times = np.array(run_length)
        np.save(os.path.join(path, f"RunTimes_a{alp_str}_e{eps_str}.npy"), run_times)
        lengths_by_alpha[alp_str].append((run_times))

    lengths_list = lengths_by_alpha[alp_str]
    # Plotting: boxplot per epsilon and mean line
    plt.figure(figsize=(8,5))
    positions = range(len(eps_values))
    plt.boxplot(lengths_list, positions=positions, widths=0.6, patch_artist=True)
    means = [np.mean(a) for a in lengths_list]
    plt.plot(positions, means, '-o', color='C2', label='mean', markeredgewidth=1, markeredgecolor='k',markersize=6)
    plt.xticks(positions, [f"{e:.2f}" for e in eps_values])
    plt.xlabel('Epsilon')
    plt.ylabel('Episode duration (timesteps)')
    plt.ylim(0, 1000)
    plt.title(f'Simulation Duration for different epsilon, Alpha = {alp_str} (100 runs each)')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'eval_lengths_a{alp_str}_by_epsilon.png'), dpi=200)
    plt.close()
    run_times = []

    plt.figure(figsize=(8,5))

    for eps, rewards in reward_list:
        plt.plot(rewards, label=f'Epsilon = {eps:.2f}')

    plt.title(f'Epsilon/Alpha Effect on Cumulative Reward, Alpha = {alp_str}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.ylim(0, 6000000)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path,f'Cumulative_Reward_Eps_Comparison_a{alp_str}.png'))
    reward_list = []

########## Gamma/Epsilon Simulation & Plots ###############
gam_values = [1.0 - 0.1*i for i in range(4)] 
alpha = 0.15
lengths_list = [] 

for gamma in gam_values:
    gam_str = f"{gamma:.2f}" 
    lengths_by_gamma[gam_str] = [] 
    path = f'Gamma/Gam{gam_str}'

    for epsilon in eps_values:
        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...

        r_arr_tit = f"R_b10_a0.1_g{gam_str}_e{eps_str}.npy"
        full_r_path = os.path.join(path, r_arr_tit)
        rewards = np.load(full_r_path)
        reward_list.append((epsilon, rewards))
        q_tab_tit = f"Q_b10_a0.1_g{gam_str}_e{eps_str}.npy"
        full_q_path = os.path.join(path, q_tab_tit)
        q_tab = np.load(full_q_path)

        env_local2 = gym.make('CartPole-v1')
        Q2 = Q_Learning(env=env_local2, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        Q2.Qmatrix = q_tab

        run_length = []
        for run in range(100):
            rewards, used_env = Q2.simulateLearnedStrategyLoaded(env=None, render_delay=0)
            used_env.close()
            run_length.append(len(rewards)) 

        run_times = np.array(run_length)
        np.save(os.path.join(path, f"RunTimes_g{gam_str}_e{eps_str}.npy"), run_times)
        lengths_by_gamma[gam_str].append((run_times))

    lengths_list = lengths_by_gamma[gam_str]
    # Plotting: boxplot per epsilon and mean line
    plt.figure(figsize=(8,5))
    positions = range(len(eps_values))
    plt.boxplot(lengths_list, positions=positions, widths=0.6, patch_artist=True)
    means = [np.mean(a) for a in lengths_list]
    plt.plot(positions, means, '-o', color='C2', label='mean', markeredgewidth=1, markeredgecolor='k',markersize=6)
    plt.xticks(positions, [f"{e:.2f}" for e in eps_values])
    plt.xlabel('Epsilon')
    plt.ylabel('Episode duration (timesteps)')
    plt.ylim(0, 500)
    plt.title(f'Simulation Duration for different epsilon, gamma = {gam_str} (100 runs each)')
    plt.grid(axis='y', linestyle='--', alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'eval_lengths_g{gam_str}_by_epsilon.png'), dpi=200)
    plt.close()
    run_times = []

    plt.figure(figsize=(8,5))

    for eps, rewards in reward_list:
        plt.plot(rewards, label=f'Epsilon = {eps:.2f}')

    plt.title(f'Epsilon/Gamma Effect on Cumulative Reward, Gamma = {gam_str}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.ylim(0, 6000000)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path,f'Cumulative_Reward_Eps_Comparison_g{gam_str}.png'))
    reward_list = []

########### Bin Size/Epsilon Simulation & Plots ###############
bin_values = [20 + 5*i for i in range(1)] 
alpha = 0.15
gamma = 1.0
lengths_list = [] 


for binsize in bin_values:
    bin_str = f"{binsize:d}" 
    lengths_by_binsize[bin_str] = [] 
    path = f'Bins/BinSize{bin_str}'
    numberOfBins=[binsize, binsize, binsize, binsize,]

    for epsilon in eps_values:
        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...

        r_arr_tit = f"R_b{bin_str}_a0.1_g1.0_e{eps_str}.npy"
        full_r_path = os.path.join(path, r_arr_tit)
        rewards = np.load(full_r_path)
        reward_list.append((epsilon, rewards))

        q_tab_tit = f"Q_b{bin_str}_a0.1_g1.0_e{eps_str}.npy"
        full_q_path = os.path.join(path, q_tab_tit)
        q_tab = np.load(full_q_path)

        env_local3 = gym.make('CartPole-v1')
        Q3 = Q_Learning(env=env_local3, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        Q3.Qmatrix = q_tab

        run_length = []
        for run in range(100):
            rewards, used_env = Q3.simulateLearnedStrategyLoaded(env=None, render_delay=0)
            used_env.close()
            run_length.append(len(rewards)) 

        run_times = np.array(run_length)
        np.save(os.path.join(path, f"RunTimes_b{bin_str}_e{eps_str}.npy"), run_times)
        lengths_by_binsize[bin_str].append((run_times))

    lengths_list = lengths_by_binsize[bin_str]
    # Plotting: boxplot per epsilon and mean line
    plt.figure(figsize=(8,5))
    positions = range(len(eps_values))
    plt.boxplot(lengths_list, positions=positions, widths=0.6, patch_artist=True)
    means = [np.mean(a) for a in lengths_list]
    plt.plot(positions, means, '-o', color='C2', label='mean', markeredgewidth=1, markeredgecolor='k',markersize=6)
    plt.xticks(positions, [f"{e:.2f}" for e in eps_values])
    plt.xlabel('Epsilon')
    plt.ylabel('Episode duration (timesteps)')
    plt.ylim(0, 1000)
    plt.title(f'Simulation Duration for different epsilon, Bin Size = {bin_str} (100 runs each)')
    plt.grid(axis='y', linestyle='--', alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'eval_lengths_b{bin_str}_by_epsilon.png'), dpi=200)
    plt.close()
    run_times = []

    plt.figure(figsize=(8,5))

    for eps, rewards in reward_list:
        plt.plot(rewards, label=f'Epsilon = {eps:.2f}')

    plt.title(f'Epsilon/Bin Size Effect on Cumulative Reward, Bin Size = {bin_str}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path,f'Cumulative_Reward_Eps_Comparison_b{bin_str}.png'))
    reward_list = []
