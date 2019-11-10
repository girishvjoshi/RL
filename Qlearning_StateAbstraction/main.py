import numpy as np
from qlearning import Qagent
import matplotlib as mpl
import matplotlib.pyplot as plt
import gym


def main():
  epsilon = 0.99
  epsilon_min = 0.1
  epsilon_decay = 0.9997
  max_epoch = 10000
  ep_length = 100
  reward_rec = []
  Avgreward = []
  env = gym.make('Pendulum-v0')
  agent = Qagent(env)
  for i in range(max_epoch):
    done = False
    total_reward = 0
    s = env.reset()
    for steps in range(ep_length):
      action = agent.chooseAction(epsilon,s)
      nexts, r, done, info = env.step(action)
      agent.updateQ(s,action,r, done, nexts)
      s = nexts
      total_reward = total_reward + r

    reward_rec.append(total_reward)
    if (i % 100) == 0:
      avg_reward = np.average(reward_rec[-100:])
      Avgreward.append(avg_reward)
      print('Average Reward:',avg_reward)
    epsilon = epsilon*epsilon_decay
    if epsilon < epsilon_min:
      epsilon = epsilon_min
    
  print(epsilon)
  plt.plot(Avgreward)
  plt.xlabel('Epochs X 100')
  plt.ylabel('Avg Reward')
  plt.savefig('plot.png')


if __name__ == "__main__":
  main()
  

