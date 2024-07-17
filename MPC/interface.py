# -*- coding: utf-8 -*-

import sys
import numpy as np
import rospy


class Interface(object):
    def __init__(self):
        self._initialize_node('mpc_controller')

        #load parameters
        publish_rate = 1 / rospy.get_param('~sampling_time')
        self._rate = rospy.Rate(publish_rate)
        #run_mode = rospy.get_param('~run_mode')
        self._taget_speed = rospy.get_param('~target_speed')

        # Inputs
        self.inputs = dict(
            state=None,
            path=None,
            path_yaw=None,
            # speed_profile=None,
        )
        
        # if run_mode == "DV":  
        #     # Subscribers real car
        #     rospy.Subscriber('/navigation/speed_profiler/path', PlannedPath, self._cb_path)
        #     rospy.Subscriber('/slam/output/odom', Odometry, self._cb_odometry)
        #     # Publishers real car
        #     self.dv_control_target_publisher = rospy.Publisher('/navigation/stanley/dv_control_target', dv_control_target, queue_size=1)
        # elif run_mode == "FSDS":
        #     # Subscribers simulation
        #     rospy.Subscriber('/navigation/speed_profiler/path', PlannedPath, self._cb_path)
        #     rospy.Subscriber('/fsds/testing_only/odom', Odometry, self._cb_odometry)
        #     # Publishers simulation
        #     self.fsds_control_target_publisher = rospy.Publisher('/fsds/control_command', FSDS_ControlCommand, queue_size=1)
        

    def _initialize_node(self, node_name):
        myargv = rospy.myargv(argv=sys.argv)
        debug = myargv[1].lower() == 'true' if len(myargv) == 2 else False
        rospy.init_node(node_name, log_level=rospy.DEBUG if debug else rospy.ERROR)


    """ Subscribers/callbacks """
    def _cb_odometry(self, odometry):
        x = odometry.pose.pose.position.x
        y = odometry.pose.pose.position.y
        x_speed = odometry.twist.twist.linear.x
        y_speed = odometry.twist.twist.linear.y
        velocity = np.sqrt(x_speed*x_speed + y_speed*y_speed)

        orientation_q = odometry.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        euler_angles = euler_from_quaternion(orientation_list) 
        yaw = euler_angles[2] 
        self.inputs['state'] = np.array([x, y, yaw, velocity])
        

    def _cb_path(self, path):
        path_points = np.array(list(zip(path.x, path.y)))
        path_points_yaw = np.array(path.yaw)
        path_speed_profile = np.array(path.speed_profile)
        self.inputs['path'] = path_points
        self.inputs['path_yaw'] = path_points_yaw
        self.inputs['speed_profile'] = path_speed_profile


    """ Publishers """
    def publish_dv_control_target(self, steering_angle, speed):
        command_msg = dv_control_target()
        command_msg.dv_steering_angle_target = steering_angle
        command_msg.dv_speed_target = speed
        self.dv_control_target_publisher.publish(command_msg)

    def publish_fsds_control_target(self, steering_angle, throttle):
        command_msg = FSDS_ControlCommand()

        command_msg.throttle = np.maximum(throttle, 0)
        command_msg.brake = np.maximum(-throttle, 0)
        command_msg.steering = -steering_angle
        self.fsds_control_target_publisher.publish(command_msg)
        
    
    """ Auxilliary """
    def is_ready(self):
        return all(v is not None for v in self.inputs.values())


    def sleep(self):
        try:
            self._rate.sleep()
        except rospy.exceptions.ROSInterruptException as e:
            rospy.loginfo(
                '[controller] Program terminated. Exception: {}'
                .format(e)
            )
        
    def log(self, text):
        rospy.loginfo(text)

    @staticmethod
    def is_shutdown():
        return rospy.is_shutdown()
