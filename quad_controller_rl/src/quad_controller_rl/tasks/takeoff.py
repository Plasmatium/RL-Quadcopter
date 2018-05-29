"""Takeoff task."""
import pdb

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

def unit_vec(vec):
    norm = np.linalg.norm(vec)
    return np.zeros(3) if norm == 0 else vec/norm

class Takeoff(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        
        # action_max 用来配合Actor参数，实际和上面两个的意思一样的
        self.action_max = 25.0

        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        # self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        # 需要更确切的target，提供三维坐标
        self.target_point = np.array([0.0, 0.0, 10.0])
        self.last_pose = None

    def reset(self):
        # Nothing to reset; just return initial condition
        # print('update reseeeeeeeeeeeeeeeeeeeeeeeet\n'*5)
        return Pose( 
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist( # Twist is velocity
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Compute reward / penalty and check if this episode is complete
        done = False
        curr_point = np.array([pose.position.x, pose.position.y, pose.position.z])
        '''项目自带reward判定方式------------------------------------'''
        target_z = self.target_point[2]
        reward = -min(abs(target_z - pose.position.z), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        if pose.position.z >= target_z:  # agent has crossed the target height
            reward += 10.0  # bonus reward
            done = True
        elif timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
            done = True
        '''----------------------------------------------------------'''
        # 1. 当前位置和target_point越近越好
        # 2. 尽量小的翻转
        # curr_point = np.array([pose.position.x, pose.position.y, pose.position.z])
        # reward_x, reward_y, reward_z = [
        #     -min(20, abs(curr_point[i] - self.target_point[i])) for i in range(3)]
        
        # reward_z = (reward_z + 20)**3 / 200.0
        # reward_x /= 1.5
        # reward_y /= 1.5

        # reward = reward_x + reward_y + reward_z
        # dist = np.linalg.norm(curr_point - self.target_point)

        # if curr_point[2] >= self.target_point[2]:
        #     reward += 15.0
        #     done = True
        # elif timestamp > self.max_duration:
        #     reward -= 15.0
        #     done = True
        '''-----------------------------------------------------------------'''
        # 奖励设定为最小化速度向量和方位向量夹角
        dist = np.linalg.norm(curr_point - self.target_point)
        # la = linear_acceleration
        # acc_vec = np.array([la.x, la.y, la.z])
        # acc_unit_vec = unit_vec(acc_vec)

        # s_vec = self.target_point - curr_point
        # s_unit_vec = unit_vec(s_vec)

        # delta_orein = abs(s_unit_vec - acc_unit_vec)

        # rx = min(5, np.log(1/(delta_orein[0] + 1e-10))-15)/3
        # ry = min(5, np.log(1/(delta_orein[1] + 1e-10))-15)/3
        # rz = min(5, np.log(1/(delta_orein[2] + 1e-10))-15)

        # reward = rx + ry + rz

        # if curr_point[2] >= self.target_point[2]:
        #     reward += 15.0
        #     done = True
        # elif timestamp > self.max_duration:
        #     reward -= 15.0
        #     done = True        

        # print('suv {:7.3f} {:7.3f} {:7.3f} | '
        #     'accuv {:7.3f} {:7.3f} {:7.3f} | '
        #     'delor {:7.3f} {:7.3f} {:7.3f} | '
        #     'reward {:7.3f} | dist {:7.3f}'.format(s_unit_vec[0], s_unit_vec[1], s_unit_vec[2],
        #         acc_unit_vec[0], acc_unit_vec[1], acc_unit_vec[2],
        #         delta_orein[0], delta_orein[1], delta_orein[2],
        #         reward, dist),end='\r')
        '''----------------------------------------------------------------'''
        print('realtime reword: {:.3f} | dist is {:.3f}'.format(
            reward, dist),end='\r')

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
