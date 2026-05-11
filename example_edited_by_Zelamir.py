#!/usr/bin/env python3

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
from function_edited_by_zelamir import Q_Learning
 
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
 
numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]
 
# define the parameters
alpha=0.1
gamma=1
epsilon=0.2
numberEpisodes=15000
 
# create an object
Q1=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()

######### Example Code only uses Epsilon = 0.2, Zelamir added simulation and plots for Epsilon 0.3 and 0.4
plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsCumulative,color='blue',linewidth=1, label='Epsilon = 0.2')

plt.title('Epsilon Effect on Cumulative Reward')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
# plt.yscale('log')
plt.show()
plt.savefig('Cumulative_Reward_Epsilon.png')

epsilon=0.3
 
# create an object
Q2=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds)
# run the Q-Learning algorithm
Q2.simulateEpisodes()

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsCumulative,color='blue',linewidth=1, label='Epsilon = 0.2')
plt.plot(Q2.sumRewardsCumulative,color='red',linewidth=1, label='Epsilon = 0.3')

plt.title('Epsilon Effect on Cumulative Reward')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
# plt.yscale('log')
plt.show()
plt.savefig('Cumulative_Reward_Epsilon.png')
epsilon=0.4
 
# create an object
Q3=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds)
# run the Q-Learning algorithm
Q3.simulateEpisodes()
 
plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsCumulative,color='blue',linewidth=1, label='Epsilon = 0.2')
plt.plot(Q2.sumRewardsCumulative,color='red',linewidth=1, label='Epsilon = 0.3')
plt.plot(Q3.sumRewardsCumulative,color='green',linewidth=1, label='Epsilon = 0.4')

plt.title('Epsilon Effect on Cumulative Reward')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
# plt.yscale('log')
plt.show()
plt.savefig('Cumulative_Reward_Epsilon.png')
 
 # simulate the learned strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
# simulate the learned strategy
(obtainedRewardsOptimal,env1)=Q2.simulateLearnedStrategy()
# simulate the learned strategy
(obtainedRewardsOptimal,env1)=Q3.simulateLearnedStrategy()
 
# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)
 
# now simulate a random strategy
(obtainedRewardsRandom,env2)=Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()
 
# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
