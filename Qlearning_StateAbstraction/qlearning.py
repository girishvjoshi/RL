import numpy as np
import numpy.matlib 
import random
import gym

np.random.seed(1234)

class Qagent(object):
  def __init__(self,env, lr = 0.01, gamma = 0.95):
    self.lr = lr
    self.gamma = gamma
    self.env = env
    self.abstractionN = 5
    self.actionAbstractionN = 3
    self._stateabstraction()
    self._initQ()
    
          
  def _stateabstraction(self):
    s_high = self.env.observation_space.high
    s_low = self.env.observation_space.low
    a_high = self.env.action_space.high
    a_low = self.env.action_space.low

    a = np.linspace(s_low[0], s_high[0], self.abstractionN)
    b = np.linspace(s_low[1], s_high[1], self.abstractionN)
    c = np.linspace(s_low[2], s_high[2], self.abstractionN)
    
    self.state_list = np.array(np.meshgrid(a,b,c)).T.reshape(-1,3)

    self.action_list = np.linspace(a_low, a_high, self.actionAbstractionN)

  def _initQ(self):
    a_dim = len(self.action_list)
    self.Q = -100*np.ones((len(self.state_list), a_dim), dtype=float)

  def _findnearStateIdx(self, state):
    sIdx = np.argmin(np.sum((self.state_list-np.matlib.repmat(state, len(self.state_list), 1))**2,1))
    return sIdx

  def _findnearActionIdx(self,action):
    aIdx = np.argmin(np.sum(np.square(self.action_list-np.matlib.repmat(action, len(self.action_list),1)),1))
    return aIdx


  def updateQ(self, state, action, reward, done, next_state):
    sIdx = self._findnearStateIdx(state)
    aIdx = self._findnearActionIdx(action)
    nsIdx = self._findnearStateIdx(next_state)

    if done:
      self.Q[sIdx][aIdx] = reward
    else:
      self.Q[sIdx][aIdx] = self.Q[sIdx][aIdx] + self.lr*(reward+self.gamma*np.max(self.Q[nsIdx][:])-self.Q[sIdx][aIdx])

  def chooseAction(self, epsilon, state):
    if np.random.rand() <= epsilon:
      a = self.env.action_space.sample()
    else:
      sIdx = self._findnearStateIdx(state)
      aIdx = np.argmax(self.Q[sIdx][:])
      a = self.action_list[aIdx]

    return a


    
    

       

  




