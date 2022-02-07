import time
import gym
import gym_chrome_dino
import sys
import numpy as np
from tensorflow.keras import models
from gym_chrome_dino.utils.wrappers import make_dino

def flatten(data):
    # 10 * 5 = 50
    # R^50 -> R^2
    result = []
    for i in range(50, 60):
        for j in range(30, 35):
            result.append(data[i][j][0] + data[i][j][1] + data[i][j][2] + data[i][j][3])

    return result

if len(sys.argv) != 2:
    print('Run argument required: model_file')
else:
    env = gym.make('ChromeDino-v0')
    env = make_dino(env, timer=True, frame_stack=True)
    model = models.load_model(str(sys.argv[1]))
    s = env.reset()
    env.step(1)
    score = 0
    env.step(1)
    env.step(1)
    env.step(1)
    while True:
        #time.sleep(0.01)
        s = flatten(s)
        a = np.argmax(model.predict(np.array([s]))[0])
        s, reward, done, X = env.step(a)
        score += reward
        if done:
            print('score:', score)
            s = env.reset()
            env.step(1)
            score = 0

    env.close()
