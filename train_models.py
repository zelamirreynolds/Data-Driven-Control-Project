#! /home/vboxuser/Control/ddc/bin/python3

########################################################################
'''
This code is sourced from Aleksander Haber, link below. I (Zelamir) have added some of my own code and commented clearly those instances.

“Detailed Explanation and Python Implementation of the Q-Learning Algorithm with Tests in Cart Pole OpenAI Gym Environment – Reinforcement Learning Tutorial”. 
Technical Report, Number 6, Aleksandar Haber, (2023), Publisher: www.aleksandarhaber.com, 
Link: https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/
'''
########################################################################
# Note: 
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)    
     
# tested on     
# gym==0.26.2
# gym-notices==0.0.8
 
#gymnasium==0.27.0
#gymnasium-notices==0.0.1
 
# classical gym 
import gymnasium as gym
# instead of gym, import gymnasium 
# import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt 
 
# import the class that implements the Q-Learning algorithm
from function_edited_by_Zelamir import Q_Learning
 
#env=gym.make('CartPole-v1',render_mode='human')
env=gym.make('CartPole-v1')
(state,_)=env.reset()
#env.render()
#env.close()
 
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

eps_values = [0.20 + 0.05*i for i in range(5)]  # [0.2, 0.3, 0.4]
alp_values = [0.15 + 0.05*i for i in range(4)] 

######### Train Model for Incremented Alpha/Epsilon Values, and save Q-table ##########
for alpha in alp_values:
    alp_str = f"{alpha:.2f}" 
    for epsilon in eps_values:

        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...
        q_tab_tit = f"Q_b10_a{alp_str}_g1.0_e{eps_str}.npy"
        r_arr_tit = f"R_b10_a{alp_str}_g1.0_e{eps_str}.npy"

        env_local1 = gym.make('CartPole-v1')
        # create an object
        Q1 = Q_Learning(env=env_local1, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        # run the Q-Learning algorithm
        Q1.simulateEpisodes(q_table_title = q_tab_tit, reward_arr_title = r_arr_tit, path = f'Alpha/Alp{alp_str}')
        env_local1.close()

######### Train Model for Incremented Gamma/Epsilon Values, and save Q-table ##########
gam_values = [0.9 - 0.1*i for i in range(4)] 
alpha = 0.15

for gamma in gam_values:
    gam_str = f"{gamma:.2f}" 
    for epsilon in eps_values:

        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...
        q_tab_tit = f"Q_b10_a0.15_g{gam_str}_e{eps_str}.npy"
        r_arr_tit = f"R_b10_a0.15_g{gam_str}_e{eps_str}.npy"

        env_local2 = gym.make('CartPole-v1')
        # create an object
        Q2 = Q_Learning(env=env_local2, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        # run the Q-Learning algorithm
        Q2.simulateEpisodes(q_table_title = q_tab_tit, reward_arr_title = r_arr_tit, path = f'Gamma/Gam{gam_str}')
        env_local2.close()


######### Train Model for Incremented Bin Size/Epsilon Values, and save Q-table ##########
bin_values = [10 + 5*i for i in range(3)] 
alpha=0.15
gamma=1.0

for bins in bin_values:
    bin_str = f"{bins:d}" 
    numberOfBins=[bins, bins, bins, bins]
    for epsilon in eps_values:

        eps_str = f"{epsilon:.2f}"  # "0.2", "0.25", "0.3", ...
        q_tab_tit = f"Q_b{bin_str}_a0.15_g1.0_e{eps_str}.npy"
        r_arr_tit = f"R_b{bin_str}_a0.15_g1.0_e{eps_str}.npy"

        env_local3 = gym.make('CartPole-v1')
        # create an object
        Q3 = Q_Learning(env=env_local3, alpha=alpha, gamma=gamma, epsilon=epsilon, numberEpisodes=numberEpisodes, numberOfBins=numberOfBins, lowerBounds=lowerBounds, upperBounds=upperBounds)
        # run the Q-Learning algorithm
        Q3.simulateEpisodes(q_table_title = q_tab_tit, reward_arr_title = r_arr_tit, path = f'Binsize/Bin{bin_str}')
        env_local3.close()
