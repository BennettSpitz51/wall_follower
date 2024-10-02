#!/usr/bin/env python
import rospy
import random
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

class WallFollower:
    def __init__(self):
        rospy.init_node('wall_follower', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=30)
        rospy.Subscriber('/scan', LaserScan, self.laser_calc)
        self.twist = Twist()
        self.dw = 0.75  
        self.alpha = 0.5  
        self.gamma = 0.9  
        self.epsilon = 1.0  
        self.epsilon_decay = 0.995  
        self.epsilon_min = 0.1
        self.q_table = {}  
        self.state = None  
        self.steps = 0
        self.episodes = 0
        self.training = True
        self.max_steps = 3000
        self.max_episodes = 5000
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_gazebo = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    def reset_simulation(self):
        rospy.loginfo("Resetting simulation...")
        self.steps = 0
        self.state = None  
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_pub.publish(self.twist)
        try:
            self.reset_gazebo()
            rospy.loginfo("Gazebo simulation reset successfully.")
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to reset Gazebo simulation: %s" % e)

    def laser_calc(self, data):
        left_dist = min(data.ranges[40:150])
        front_dist = min(min(data.ranges[340:360]), min(data.ranges[0:20]))
        next_state = self.find_state(left_dist, front_dist)
        if self.training:
            self.q_learning_step(next_state)
        else:
            self.take_action(next_state)
        self.steps += 1
        if front_dist < 0.25:
            rospy.loginfo("Collision detected! Calculating negative reward and resetting simulation...")
            self.q_learning_step('front_blocked')  
            self.reset_simulation()  
            self.episodes += 1

        elif self.steps >= self.max_steps:
            rospy.loginfo("Max steps reached, resetting simulation...")
            self.reset_simulation()
            self.episodes += 1

        if self.episodes >= self.max_episodes and self.training:
            rospy.loginfo("Training finished. Testing phase starting...")
            self.training = False
            self.episodes = 0
            self.max_episodes = 10  
            self.epsilon = 0  

    def find_state(self, left_dist, front_dist):
        if front_dist < 0.50:
            return 'front_blocked'
        elif left_dist < 0.70:
            return 'left_too_close'
        elif left_dist > 0.80:
            return 'left_too_far'
        else:
            rospy.loginfo("LEFT OK")
            return 'left_ok'

    def take_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {'move_forward': 0.0, 'turn_left': 0.0, 'turn_right': 0.0, 'wall_turn': 0.0}
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(self.q_table[state].keys()))  
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)  
        self.perform_action(action)
        return action

    def perform_action(self, action):
        if action == 'move_forward':
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0
            rospy.loginfo("MOVE FORWARD")
        elif action == 'turn_left':
            self.twist.linear.x = 0.05
            self.twist.angular.z = 0.3
            rospy.loginfo("TURN LEFT")
        elif action == 'turn_right':
            self.twist.linear.x = 0.05
            self.twist.angular.z = -0.3
            rospy.loginfo("TURN RIGHT")
        elif action == 'wall_turn':
            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.2
            rospy.loginfo("WALL TURN")
        self.cmd_pub.publish(self.twist)

    def q_learning_step(self, next_state):
        if self.state is None:
            self.state = next_state
            return
        if self.state not in self.q_table:
            self.q_table[self.state] = {'move_forward': 0.0, 'turn_left': 0.0, 'turn_right': 0.0, 'wall_turn': 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {'move_forward': 0.0, 'turn_left': 0.0, 'turn_right': 0.0, 'wall_turn': 0.0}
        action = self.take_action(self.state)
        reward = self.calculate_reward(next_state)
        old_value = self.q_table[self.state][action]
        next_max = max(self.q_table[next_state].values())
        self.q_table[self.state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.state = next_state

    def calculate_reward(self, state):
        if state == 'front_blocked':
            return -500  
        elif state == 'left_ok':
            return 50  
        elif state == 'left_too_close':
            return 1
        else:
            return -1  

if __name__ == '__main__':
    try:
        wf = WallFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

