# place multiple agent in the environment, each agent share the same action space
# each agent perform randome action in the environment

import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        else:
            return self.action
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0

def init_pose(env):
    # set enemy boat to far away
    actions=[[0,0,0,0],[0,0],[50,0]]
    obs, rewards, done, info = env.step(actions)
    time.sleep(5)
    actions=[[0,0,0,0],[0,0],[50,30000]]
    obs, rewards, done, info = env.step(actions)
    time.sleep(10)
    # set drone to the start angle
    #actions=[[0,0,0,180],[0,0],[0,0]]
    obs, rewards, done, info = env.step(actions)
    time.sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-Ocean_Map-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(240,240))
    env.unwrapped.agents_category=['boat', 'boat_enemy', 'drone'] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    if args.nav_agent:
        env = agents.NavAgents(env, mask_agent=False)
    episode_count = 100
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    for eps in range(1, episode_count):
        obs = env.reset()
        drone_start_loc=[5080, 15145, 2321, -0.19, 34.910, 0]
        # boat_start_loc=[-6417.9, 6440.8, 125.3, -6.0, 2.3, 0.0]
        # boat_enemy_start_loc=[20709.7, 14469.0, 80.115, -3.6, 77.9, 0.0]
        env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0],drone_start_loc[:3])
        #env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[0],drone_start_loc[3:])
        # env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[1],boat_start_loc[:3])
        # env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[1],boat_start_loc[3:])
        # env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[2],boat_enemy_start_loc[:3])
        # env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[2],boat_enemy_start_loc[3:])

        init_pose(env)
        #agents_num = len(env.action_space)
        #agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
        count_step = 0
        t0 = time.time()
        agents_num = len(obs)
        #C_rewards = np.zeros(agents_num)
        # set to init agent pos
        
        while True:
           
            # actions = [agents[i].act(obs[i]) for i in range(agents_num)]

            # actions 输入是环境中所有智能体的action list: [drone,boat, boat_enemy]
            # actions_space:
            # -drone:[x,y,z,yaw]
            # -boat:[angle,speed]
            #example:[[1,0,0,0], [0,500], [0,0]]
            #env.unwrapped.unrealcv.set_attack(env.unwrapped.player_list[1],20)
            actions=[[0,0,0,0],[0,0],[0,0]]
            obs, rewards, done, info = env.step(actions)
            #C_rewards += rewards
            count_step += 1
            cv2.imshow('drone obs',obs[0])
            cv2.imshow('boat obs',obs[1])
            cv2.imshow('boat_enemy obs',obs[2])
            cv2.waitKey(1)

    # Close the env and write monitor result info to disk
    print('Finished')
    env.close()