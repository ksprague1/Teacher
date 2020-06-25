import gym
import Teacher
import mountain_car as mc
import single_mountain_car as smc
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import DQN

m = mountain_car.MountainCarEnv()
env = Teacher.Teacher(m,-100,0.1,n_stats=4,timesteps=80000)
#env = gym.make('CartPole-v1')

model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./teacher")
model.save("teacher\\0k.pkl")
env.model.save("student\\0k.pkl")
#model = PPO2.load("teacher.pkl",env=env)
#try:
model.learn(total_timesteps=4000,log_interval=10)
#except:
print("Saving Half way")
model.save("teacher\\4k.pkl")
env.model.save("student\\4k.pkl")
model.learn(total_timesteps=4000,log_interval=10)
model.save("teacher\\8k.pkl")
env.model.save("student\\8k.pkl")
#del model # remove to demonstrate saving and loading

#model = PPO2.load("teacher.pkl")
#mc_model = DQN.load("TRAINED.pkl")
#env.model=mc_model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.test_step(action)
    print(obs,rewards)
    #env.render()
