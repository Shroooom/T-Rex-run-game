import gym
from tensorflow.python.util.tf_decorator import rewrap
import gym_chrome_dino
import numpy as np
import pickle
import random
import collections
import time
import sys
from collections import deque
from tensorflow.keras import models, layers, optimizers
from collections import defaultdict
from gym_chrome_dino.utils.wrappers import make_dino

class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 100
        self.replay_size = 1000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        # Hidden layer = 100
        # Input size = 800
        STATE_DIM, ACTION_DIM = 10 * 5 * 1, 2
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.2):
        # Random or by DQN
        # epsilon value decreases by factor of .00001 each random action
        # more training -> less random actions
        if np.random.uniform() < epsilon - self.step * 0.00001 or np.random.uniform() < 0.005:
            return np.random.choice([0, 1])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='t-rex-v0-dqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward,boolean,done):
        # further separate types of rewards to help t-rex learns faster
        if(reward > 1): #t-rex jumps over obstacle
            reward = reward + 100
        elif(reward ==1): # t-rex is just walking
            reward = reward + 5
        else: #t-rex jumps for no reason
            reward = reward - 100
        if(done): # if t-rex dies, penalize a lot
            score = -1000
        self.replay_queue.append((s, a, next_s, reward,boolean,done))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # Each update_freq step, weight of target_model = weight of model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # Update Q table
        for i, replay in enumerate(replay_batch):
            _, a, _, reward,_,_ = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # Train
        self.model.fit(s_batch, Q, verbose=0)

def flatten(data):
    # flatten data for faster calculations
    # 10 * 5 = 50
    # R^50 -> R^2
    result = []
    for i in range(50, 60):
        for j in range(30, 35):
            result.append(data[i][j][0] + data[i][j][1] + data[i][j][2] + data[i][j][3])

    return result

def have_obstacle(data):
    return sum(data) / len(data) != 252

version_tag = '0'
save_gap = 100
if len(sys.argv) == 3:
    version_tag = sys.argv[1]
    save_gap = int(sys.argv[2])
    print('Save model every', save_gap, 'times training with version tag', version_tag)

env = gym.make('ChromeDino-v0')
#env.set_acceleration_value(0)
env = make_dino(env, timer=True, frame_stack=True)
episodes = 1000
score_list = []
agent = DQN()
for i in range(episodes):
    s = flatten(env.reset())
    score = 0
    start_time = time.time()
    # temp_time = time.time()
    while True:
        if time.time() - start_time < 3:
            env.step(0)
            continue
        
        a = agent.act(s)
        reward = 0
        # reward = time.time() - temp_time # alive time
        boolean = have_obstacle(s)
        if boolean:
            if a == 1:
                #print('detecting: should jump, and did jump')
                reward = 100
            else:
                #print('detecting: should jump, but did not jump')
                reward =-1000
        else:
            if a != 1:
                #print('detecting: should not jump, and did not jump')
                reward = 1
            else:
                #print('detecting: should not jump, but did jump')
                reward =-100
        
        next_s, _, done, _ = env.step(a)
        next_s = flatten(next_s)

        # temp_time = time.time()
        agent.remember(s, a, next_s, reward,boolean,done)
        agent.train()
        score += reward
        s = next_s
        if done:
            agent.remember(s, a, next_s, reward,boolean,done)
            agent.train()
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    
    if i % save_gap == 0:
        agent.save_model('t-rex-v' + version_tag + '-dqn' + (str)(i) + '.h5')

agent.save_model()
env.close()
