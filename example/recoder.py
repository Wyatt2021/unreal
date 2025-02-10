import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import os
import numpy as np
os.environ['UnrealEnv'] = r'C:\\Users\\86188\Desktop\\tasks\\unreal\\unreal\\gym_unrealcv\\envs\\UnrealEnv'

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-Ocean_Map-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", "--render", default=False, metavar='G', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = env.unwrapped
    agent_0 = RandomAgent(env.action_space[1])
    agent_1 = RandomAgent(env.action_space[2])

    episode_count = 100
    rewards = 0
    done = False

    for i in range(episode_count):
        env.seed(i)
        #env.direction = 2*np.pi/8.0 * env.count_eps
        env.direction = 0
        obs = env.reset()
        #os.mkdir("%03d" % env.count_eps)
        #cv2.imwrite(os.path.join("%03d" % env.count_eps, "%03d" % env.count_steps+'.png'), obs[0])
        count_step = 0
        t0 = time.time()
        obs, rewards, done, info = env.step([[0,0],[0,0]])
        while True:
            # action_0 = agent_0.act(obs, rewards, done)
            # action_1 = agent_1.act(obs, rewards, done)    
            # action_1 = agent_0.act(obs, rewards, done)
            target_yaw = info['Relative_Pose'][0][1][1]
            yaw = info['angle'][0]
            if abs(target_yaw - yaw) <= 0.5:
                action_0 = [0,500]
            elif yaw < target_yaw:
                action_0 = [10,0]
            else:
                action_0 = [-10,0]
            action_1 = [0,0]
            print(yaw,target_yaw,action_0)
            obs, rewards, done, info = env.step([action_0, action_1])
            # recoder
            #cv2.imwrite(os.path.join("%03d" % env.count_eps, "%03d" % env.count_steps + '.png'), obs[0])
            if info['Done']:
                import json
                with open(os.path.join("%03d" % env.count_eps, 'info.json'), 'w') as f:
                    json.dump(env.trajectory, f)
            count_step += 1
            if args.render:
                img = env.render(mode='rgb_array')
                #  img = img[..., ::-1]  # bgr->rgb
                cv2.imshow('show', img)
                cv2.waitKey(1)
            if done:
                fps = count_step / (time.time() - t0)
                print ('Fps:' + str(fps))
                break

    # Close the env and write monitor result info to disk
    env.close()


