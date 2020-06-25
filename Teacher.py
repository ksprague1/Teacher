##Teacher Has a target success for the agent
##It's state is the initial environment conditions as well as the agent's reward
##from their last game. It's actions are to change the initial conditiions for
##the next game.
##
##It's reward is -|success%-target_success%|
#import gym
#import Games
#from baselines import deepq as learner

from gym import spaces

import sys, gym, time
import mountain_car
import numpy as np
from dqn import DQN
from stable_baselines.deepq.policies import MlpPolicy
class Teacher(gym.Env):
    """The Teacher (Continuous) Environment
    :papam student: The environment in which to train your student
    :param success_thresh: The minimum episodic reward for the student to be considered successful on the episode
    :param success_target: The target probability of a student succeeding in a given task. Should take values in [0,1]
    :param n_stats: The number of runs used to gather the success probability of a student on a task. 
    :param f_n: The directory of the file where the student policy is stored.
    :param timesteps: The estimated number of epochs the student will train for
    :param r_range: The estimated reward range (is used to squish episodic rewards in [-1,1])
    Note: The student must have a vector called initial_conditions as well as a function
    to change those initial conditions called set_initials.
    """
    def __init__(self,student,success_thresh,success_target,f_n="default",timesteps=2000,r_range=50,n_stats=10):

        self.student=student
        self.num_sessions=50
        self.n=n_stats
        self.timesteps = timesteps
        self.thresh=success_thresh
        self.target=success_target
        self.r_range=r_range
        #declaring action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0,shape=student.initial_conditions.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1,high=1,shape=[16])

        #making the network for the student environment
        self.f_n=f_n
        self.build_training()


        self.reset()
        
    def build_training(self):
        """Creates a DQN policy to train on the student environment"""
        #env = gym.make('MountTest-v0')
        self.model = DQN(MlpPolicy, self.student, verbose=1,tensorboard_log="./student")
        #I turned this into a generator that yields returns as deepq learns
        self.act = self.model.learn(total_timesteps=self.timesteps*self.num_sessions*self.n)

    def test_step(self,action,render=True):
        """Runs and the student without training, while rendering the environment"""
        self.student.set_initials(action)
        env=self.student
        model=self.model
        rew=[]
        for i in range(self.n):
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                if render:
                    env.render()
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                episode_rew += rewards
            #print("Episode reward", episode_rew)
            rew+=[episode_rew]
        #env.close()
        full_rew=np.tanh((np.asarray(rew)-self.thresh)/self.r_range)
        self.state[0:4]=np.random.random(4)*2-1
        if self.n>=12:
            self.state[4:16]=full_rew[self.n-12:self.n]
        else:
            self.state[4:16-self.n]=self.state[4+self.n:16]
            self.state[16-self.n:16]=full_rew
        self.num_steps+=1
        reward =-abs(sum([(a>=self.thresh)/self.n for a in rew])-self.target)*2
        done =  (self.num_steps>self.num_sessions)
        return self.state, reward, done, {"rewards":rew}

    
    def step(self,action):
        #action is to change initial conditions
        self.student.set_initials(action)
        self.student.reset()
        try:
            #get a set of rewards from running the student wit your initial conditions
            rew=[next(self.act) for a in range(self.n)]
            #apply a tanh in order to 'squish' the returns into [-1,1]
            full_rew=np.tanh((np.asarray(rew)-self.thresh)/self.r_range)
            #apply add a random seed to the state
            self.state[0:4]=np.random.random(4)*2-1
            #add the last 12 returns to the state
            if self.n>=12:
                self.state[4:16]=full_rew[self.n-12:self.n]
            else:
                self.state[4:16-self.n]=self.state[4+self.n:16]
                self.state[16-self.n:16]=full_rew
            self.num_steps+=1
            #calculate reward for the teacher
            reward =-abs(sum([(a>=self.thresh)/self.n for a in rew])-self.target)*2
            done =  (self.num_steps>self.num_sessions)
            return self.state, reward, done, {}
        except StopIteration as stop:
            stop.value.save(self.f_n+".pkl")
            print("training done")
            return np.array(self.state), reward, True, {"rewards":rew}
        
        
    def reset(self):
        self.model.save(self.f_n+".pkl")
        self.student.defaults()
        self.num_steps=0
        self.avg_success=0
        self.state = np.random.random(16)*2-1
        self.state[4:16]=np.zeros(12)
        return np.array(self.state)
    
print(__name__)
if __name__=="__main__" and False:
    m = mountain_car.MountainCarEnv()
    t = Teacher(m,-100,0.5,n_stats=4)
    for x in range(20):
        vals=t.step(m.initial_conditions)
        print(vals)
    for x in range(20):
        vals=t.test_step(m.initial_conditions)
        print(vals)

