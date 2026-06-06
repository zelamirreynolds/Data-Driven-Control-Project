Analyzing the relationship of Q-Learning hyperparameters and their effect on model robustness, using a simulated cart-pole inverted pendulum

This code is sourced from Aleksander Haber, link below. I (Zelamir) have added some of my own code and commented clearly those instances.

example.py and function.py are the original files provided in Alexsander Haber's tutorial. example_edited_by_Zelamir.py and function_edited_by_Zelamir.py include my edits to these files

train_model.py is adapted from example.py and uses functions from functions_edited_by_Zelamir.py to train multiple Q-learning models using various hyperparameters 
#IMPORTANT! Q-tables are saved in folders that will need to be created and named acordingly

simulate_and_plot.py is also adapted from examply.py and uses functions_edited_by_Zelamir.py to simulate the trained models and plot simulation duration distribution and cumulative reward. This file uses the Q-tables saved in the respective folders by train_model.py

“Detailed Explanation and Python Implementation of the Q-Learning Algorithm with Tests in Cart Pole OpenAI Gym Environment – Reinforcement Learning Tutorial”. Technical Report, Number 6, Aleksandar Haber, (2023), Publisher: www.aleksandarhaber.com, Link: https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/
