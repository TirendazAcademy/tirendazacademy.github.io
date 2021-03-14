---
layout: post
author: Nikolay Kostadinov
title: Solving CartPole-v0 with xgboost
categories: [python, machine learning, reinforcement learning, xgboost, cart pole, open gym, openai, artificial intelligence]
---

An artificial intelligence agent starting to learn by from its own mistakes until it is fit to handle a certain task like an expert? To many this does sound like a science like science fiction, but it is based on a simple principle called Reinforcement Learning. 

# The Cart Pole Problem

Recently, I found the <a href="https://gym.openai.com/" target="_blank">OpenAI Gym</a> and started playing with some of the environments. It certainly is a nice way of getting your head off kaggle.com for a while. This is a start of a series of posts describing solutions to some of the problems posted there. 

As suggested on the <a href="https://gym.openai.com/docs" target="_blank">Getting stared page</a> I got my hands on one of the easier problems, called <a href="https://gym.openai.com/envs/CartPole-v0" target="_blank">CartPole-v0</a>. Basically, you have to balance a pole on a cart. Each time frame you have to choose between one of two "actions" `[1;-1]` and thereby move the pole either left or right. Note, the actual action set `[0;1]`.

## Cache

The first problem you have to solve is figuring out how to structure the data. Obviously, your input data should contain the four observations. Interestingly enough, since we are solving this problem by applying supervised learning, the semantics of this data is not important (black box approach). The tricky part is what comes next. You add the action `[0;1]` taken based on these observations as a fifth input variable. Deciding on how to represent the output variable is probably even trickier. As an output variable you take the count of time frames it takes for the episode to finish - either the pole falls on its side or you reach the maximum of 200 time frames. Ok, let's start by defining a simple class called `Cache`:


{% highlight python %}
class Cache:
    
    def __init__(self):
        self.cache = []
        self.index = 0
   
    def cache_data(self, observation, action, time_frame):
        cache_data = np.append(observation,[action,time_frame])
        indexed_cache_data = np.append(self.index, cache_data)
        self.cache.append(indexed_cache_data)
        self.index += 1
    
    def get_frame(self):
        df_cache = pd.DataFrame(columns=FRAME_COLUMNS, data=self.cache)
        
        # Normalize reward
        future_reward = df_cache['future_reward'].values
        max_future_reward = np.max(future_reward)
        df_cache['future_reward'] = max_future_reward - future_reward
        
        return df_cache
{% endhighlight %}

## Memory

As the episode starts, for each time frame `cache_data` is called to store the observation, the action taken and the time frame index. At the end of the episode the `get_frame` creates a data frame - the valuable peace of data that is later to be learned by a model. Notice the transformation of the output variable (here called `future_reward`) into the count of time frames it takes for the episode to finish. Next, we create a class `Memory`:

{% highlight python %}
class Memory:
    
    def __init__(self):
        self.df_data = pd.DataFrame(columns=FRAME_COLUMNS)
    
    def add_cache(self, cache):
        self.df_data = pd.concat([self.df_data, cache.get_frame()])
{% endhighlight %}

## Brain

The `Memory` class holds all the data that our "AI agent" is going to use when learning. After each episode the "cache" or the short-term memory is added to the "memory" or the long-term memory. The last piece of the puzzle is adding the brain:

{% highlight python %}
class Brain:
    
    def __init__(self, memory):
        self.regressor = None
        self.memory = memory
        
    def train(self):
        
        msk = np.random.rand(len(self.memory.df_data)) < 0.90 # 10% of data is used for early stopping
        train, validation = self.memory.df_data[msk], self.memory.df_data[~msk]
        
        train_xgdmat =  xgb.DMatrix(train[FEATURES], label=train['future_reward'])
        validation_xgdmat =  xgb.DMatrix(validation[FEATURES], label=validation['future_reward'])
        watchlist = [(validation_xgdmat, 'test')]
        self.regressor = xgb.train(XGB_PARAMS, train_xgdmat, MAX_ITERATIONS, watchlist, verbose_eval=False)
        
    def is_exploration(self, episode):
        return episode < 5 or (episode < 15 and episode % 2 == 0)
        
    def decide_action(self, observations, episode):
        if self.is_exploration(episode):
            return random.randint(0, 1)
        else:
            x_0 = np.append(observations, [0]).reshape(1,5)
            x_1 = np.append(observations, [1]).reshape(1,5)
            future_reward_0 = self.regressor.predict(xgb.DMatrix(x_0, feature_names=FEATURES))[0]
            future_reward_1 = self.regressor.predict(xgb.DMatrix(x_1, feature_names=FEATURES))[0]
            return 0 if future_reward_0 > future_reward_1 else 1
{% endhighlight %}

## Putting it all together

After each episode the 'train' function is called - a model is fitted to the data collected so far. I won't get into details, as there is plenty of material online on xgboost or other learning also. However, it took me quite a lot of time in order to fine tune xgboost to perform well, probably a little more than a couple of hours. Next, to learning, the brain also has to decide for an action based on observation. For the first few episodes, the brain should behave randomly. Afterward, it gradually switches to fully conscious decisions by using the regression model. Basically, the regressions model tries to predict which one of the two actions will lead to a higher count of time frames before the episode ends. The whole code is posted below, feel free to reproduce it. This <a href="https://gym.openai.com/evaluations/eval_XxwHyBGS22PX3ha0bLJ9A" target="_blank">solution</a> did quite well and solved the environment after 15 episodes and only 9 seconds. You can see the behavior of the cart pole on the video below:

<iframe width="600" height="400" src="https://openai-kubernetes-prod-scoreboard.s3.amazonaws.com/v1/evaluations/eval_XxwHyBGS22PX3ha0bLJ9A/training_episode_batch_video.mp4" frameborder="0"></iframe>
---
And here is the complete source code for the cart pole solution:

{% highlight python %}
import gym
import pandas as pd
import numpy as np
import xgboost as xgb
import random

FEATURES = ['observation_1','observation_2', 'observation_3', 'observation_4', 'action']
FRAME_COLUMNS = ['index'] + FEATURES + ['future_reward']
XGB_PARAMS = {
              'eta': 0.05,
              'max_depth': 5,
              'silent': 1,
              'gamma':5,
              'lambda': 10
            }
MAX_ITERATIONS = 2000
MAX_TIME_FRAMES = 500
MAX_EPISODES = 200
FINAL_FRAME = 200


class Cache:
    
    def __init__(self):
        self.cache = []
        self.index = 0
   
    def cache_data(self, observation, action, time_frame):
        cache_data = np.append(observation,[action,time_frame])
        indexed_cache_data = np.append(self.index, cache_data)
        self.cache.append(indexed_cache_data)
        self.index += 1
    
    def get_frame(self):
        df_cache = pd.DataFrame(columns=FRAME_COLUMNS, data=self.cache)
        
        # Normalize reward
        future_reward = df_cache['future_reward'].values
        max_future_reward = np.max(future_reward)
        df_cache['future_reward'] = max_future_reward - future_reward
        
        return df_cache
    
    
class Memory:
    
    def __init__(self):
        self.df_data = pd.DataFrame(columns=FRAME_COLUMNS)
    
    def add_cache(self, cache):
        self.df_data = pd.concat([self.df_data, cache.get_frame()])
        

class Brain:
    
    def __init__(self, memory):
        self.regressor = None
        self.memory = memory
        
    def train(self):
        
        msk = np.random.rand(len(self.memory.df_data)) < 0.90 # 10% of data is used for early stopping
        train, validation = self.memory.df_data[msk], self.memory.df_data[~msk]
        
        train_xgdmat =  xgb.DMatrix(train[FEATURES], label=train['future_reward'])
        validation_xgdmat =  xgb.DMatrix(validation[FEATURES], label=validation['future_reward'])
        watchlist = [(validation_xgdmat, 'test')]
        self.regressor = xgb.train(XGB_PARAMS, train_xgdmat, MAX_ITERATIONS, watchlist, verbose_eval=False)
        
    def is_exploration(self, episode):
        return episode < 5 or (episode < 15 and episode % 2 == 0)
        
    def decide_action(self, observations, episode):
        if self.is_exploration(episode):
            return random.randint(0, 1)
        else:
            x_0 = np.append(observations, [0]).reshape(1,5)
            x_1 = np.append(observations, [1]).reshape(1,5)
            future_reward_0 = self.regressor.predict(xgb.DMatrix(x_0, feature_names=FEATURES))[0]
            future_reward_1 = self.regressor.predict(xgb.DMatrix(x_1, feature_names=FEATURES))[0]
            return 0 if future_reward_0 > future_reward_1 else 1

env = gym.make('CartPole-v0')

memory = Memory()
brain=Brain(memory)

for episode in range(MAX_EPISODES):
    
    observation = env.reset()
    cache = Cache()
    
    done = False
    
    for time_frame in range(MAX_TIME_FRAMES):
        action = brain.decide_action(observation, episode)
        cache.cache_data(observation, action, time_frame)
        observation, _, done, _ = env.step(action)
        if done:
            print("Episode %d finished after %d timesteps" % (episode, time_frame+1))
            break
     
    memory.add_cache(cache)
    brain.train(episode)
{% endhighlight %}
