# place multiple agent in the environment, each agent share the same action space
# each agent perform randome action in the environment

import argparse
import gym_unrealcv
import gym
import os
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
from gym_unrealcv.envs.utils import misc
import os 
os.environ['UnrealEnv'] = r'C:\\Users\\86188\Desktop\\tasks\\unreal\\unreal_old\\gym_unrealcv\\envs\\UnrealEnv'

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

class DronePoseTracker(object):
    def __init__(self, expected_distance = 5000, expected_angle = 0):

        self.velocity_high = 0.75
        self.velocity_low = -0.75
        self.angle_high = 1
        self.angle_low = -1
        self.expected_distance = expected_distance
        self.expected_angle = expected_angle
        from simple_pid import PID
        self.angle_pid = PID(1, 0.01, 0, setpoint=1)
        self.velocity_pid = PID(3, 0.000, 0.2, setpoint=1)

    def act(self, pose, target_pose):
        delt_yaw = misc.get_direction(pose, target_pose) # get the angle between current pose and goal in x-y plane
        # angle = np.clip(self.angle_pid(-delt_yaw), self.angle_low, self.angle_high)
        angle = np.clip(self.angle_pid(self.expected_angle-delt_yaw), self.angle_low, self.angle_high)

        delt_distance = (np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2])) - self.expected_distance)
        
        velocity = np.clip(self.velocity_pid(-delt_distance/1000), self.velocity_low, self.velocity_high)

        #print('delt_distance:',delt_distance,'velocity:',velocity, 'velocity_pid:',self.velocity_pid(-delt_distance))
        return [velocity,0,0,angle]



class BoatPoseTracker(object):
    def __init__(self, expected_distance=10000, expected_angle=2):
        self.velocity_high = 600  # 根据船只的最大速度调整
        self.velocity_low = -600  # 根据船只的最小速度调整
        self.angle_high = 60  # 根据船只的最大转向角度调整
        self.angle_low = -60  # 根据船只的最小转向角度调整
        self.expected_distance = expected_distance
        self.expected_angle = expected_angle

        # 调整 PID 控制器参数以适应船只的动态特性
        from simple_pid import PID
        self.angle_pid = PID(0.5, 0.01, 0.01, setpoint=1)  # 调整角度 PID 参数
        self.velocity_pid = PID(1.0, 0.05, 0.02, setpoint=1)  # 调整速度 PID 参数

    def act(self, pose, target_pose):
        # 计算当前姿态与目标姿态之间的角度差
        delt_yaw = misc.get_direction(pose, target_pose)  # 获取当前姿态与目标姿态之间的角度差
        angle = np.clip(self.angle_pid(self.expected_angle - delt_yaw), self.angle_low, self.angle_high)

        # 计算当前姿态与目标姿态之间的距离差
        delt_distance = (np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2])) - self.expected_distance)
        velocity = np.clip(self.velocity_pid(-delt_distance), self.velocity_low, self.velocity_high)
        print(np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2])))
        if np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2])) <= 10000:
            env.unwrapped.unrealcv.set_attack(env.unwrapped.player_list[1], 20)
        # 返回控制动作
        return [angle, velocity]  

def init_pose(env):
    # set enemy boat to far away
    actions=[[0,0,0,0],[0,0],[50,0]]
    obs, rewards, done, info = env.step(actions)
    time.sleep(5)
    actions=[[0,0,0,0],[0,0],[50,22000]]
    obs, rewards, done, info = env.step(actions)
    time.sleep(10)
    # set drone to the start angle
    #actions=[[0,0,0,180],[0,0],[0,0]]
    
    # obs, rewards, done, info = env.step(actions)
    # time.sleep(5)

def update_world_camera(env, enemy_location):
    # update world camera to track the enemy
    related_pos = [-219, 6030, 7288]
    camera_rotation = [-50, -120, 0]
    cur_pos = [enemy_location[0]+related_pos[0],enemy_location[1]+related_pos[1],enemy_location[2]+related_pos[2]]
    env.unwrapped.unrealcv.set_location(0, cur_pos)
    env.unwrapped.unrealcv.set_rotation(0, camera_rotation)

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
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(640, 640))
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
        drone_start_loc=[1368, 6686, 2359, -10.269, 46.61, 0]
        # boat_start_loc=[-6417.9, 6440.8, 125.3, -6.0, 2.3, 0.0]
        # boat_enemy_start_loc=[20709.7, 14469.0, 80.115, -3.6, 77.9, 0.0]
        env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0],drone_start_loc[:3])
        #env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[0],drone_start_loc[3:])
        # env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[1],boat_start_loc[:3])
        # env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[1],boat_start_loc[3:])
        # env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[2],boat_enemy_start_loc[:3])
        # env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[2],boat_enemy_start_loc[3:])
        env.unwrapped.unrealcv.set_app(env.unwrapped.player_list[2],2)
        init_pose(env)
        drone_tracker = DronePoseTracker()   
        boat_tracker = BoatPoseTracker() 
        #agents_num = len(env.action_space)
        #agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
        count_step = 0
        t0 = time.time()
        agents_num = len(obs)
        #C_rewards = np.zeros(agents_num)
        # set to init agent pos
        enemy_destroyed = False        
        while True:
           
            # actions = [agents[i].act(obs[i]) for i in range(agents_num)]

            # actions 输入是环境中所有智能体的action list: [drone,boat, boat_enemy]
            # actions_space:
            # -drone:[x,y,z,yaw]
            # -boat:[angle,speed]
            #example:[[1,0,0,0], [0,500], [0,0]]

            try:
                enemy_boat_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.player_list[2])
            
                drone_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.player_list[0])
                boat_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.player_list[1])
                drone_action = drone_tracker.act(drone_pose,enemy_boat_pose)

                distance_drone2enemy = np.linalg.norm(np.array(drone_pose[:2]) - np.array(enemy_boat_pose[:2]))
                control_flag = False
                if distance_drone2enemy < 5000 or control_flag:
                    boat_action = boat_tracker.act(boat_pose,enemy_boat_pose)
                    control_flag = True
                else:
                    boat_action = [0,0]
                
                actions=[drone_action,boat_action,[0,300]]
                obs, rewards, done, info = env.step(actions)

                update_world_camera(env, enemy_boat_pose)
                #C_rewards += rewards
                count_step += 1
                cv2.imshow('drone obs',obs[0])
                third_view_img = env.unwrapped.unrealcv.read_image(0, 'lit')
                if count_step%1 == 0:
                    #save image
                    drone_dir = 'C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\drone\\'
                    boat_dir = 'C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\boat\\'
                    third_dir = 'C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\3\\'
                    os.makedirs(drone_dir,exist_ok=True)
                    os.makedirs(boat_dir,exist_ok=True)
                    os.makedirs(third_dir,exist_ok=True)

                    cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\drone\\'+str(count_step)+'.png',obs[0])
                    cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\boat\\'+str(count_step)+'.png',obs[1])
                    #cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\3\\'+str(count_step)+'.png',third_view_img)

                cv2.imshow('boat obs',obs[1])
                cv2.imshow('boat_enemy obs',obs[2])
                cv2.waitKey(1)
            except:
                enemy_destroyed = True
            
            if enemy_destroyed:
                print('enemy boat destroyed')
                obs = []
                obs.append(env.unwrapped.unrealcv.read_image(1, 'lit'))
                obs.append(env.unwrapped.unrealcv.read_image(2, 'lit'))
                count_step+=1
                third_view_img = env.unwrapped.unrealcv.read_image(0, 'lit')
                if count_step%1 == 0:
                    #save image
                    cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\drone\\'+str(count_step)+'.png',obs[0])
                    cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\boat\\'+str(count_step)+'.png',obs[1])
                    #cv2.imwrite('C:\\Users\\Administrator\\Desktop\\Unrealzoo\\imgs\\3\\'+str(count_step)+'.png',third_view_img)
                cv2.imshow('drone obs',obs[0])
                cv2.imshow('boat obs',obs[1])
                #cv2.imshow('third view',third_view_img)
                cv2.waitKey(1)

    # Close the env and write monitor result info to disk
    print('Finished')
    env.close()