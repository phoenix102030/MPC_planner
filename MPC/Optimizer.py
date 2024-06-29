import casadi as ca
import numpy as np
# import forcespro
# import forcespro.nlp
from Config import VehicleDynamics
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_dc.geometry.util import (chaikins_corner_cutting, compute_curvature_from_polyline, resample_polyline,
                                         compute_pathlength_from_polyline, compute_orientation_from_polyline, compute_polyline_length)
from matplotlib.animation import FuncAnimation
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle, StaticObstacle
from commonroad.geometry.shape import Shape
from Config import compute_centers_of_approximation_circles, compute_approximating_circle_radius
import sys
import time

class Optimizer(object):
    def __init__(self, configuration, init_values, predict_horizon):
        self.configuration = configuration
        # steering angles
        self.delta_min = configuration.p.steering.min  # -1.066
        self.delta_max = configuration.p.steering.max  # 1.066
        # steering velocity
        self.deltav_min = configuration.p.steering.v_min  # -0.4
        self.deltav_max = configuration.p.steering.v_max  # 0.4
        # velocity
        self.v_min = 0  # highway
        self.v_max = configuration.p.longitudinal.v_max  # 50.8
        # acceleration
        self.a_max = configuration.p.longitudinal.a_max  # 11.5

        # get some initial values
        self.init_position, self.init_velocity, self.init_acceleration, self.init_orientation = init_values[0], init_values[1], init_values[2], init_values[3]
        # get some config for optimizer
        self.iter_length = configuration.iter_length
        self.delta_t = configuration.delta_t
        self.desired_velocity = configuration.desired_velocity
        self.resampled_path_points = configuration.reference_path
        self.orientation = configuration.orientation
        self.predict_horizon = predict_horizon
        self.weights_setting = configuration.weights_setting  # store weights using a dictionary

        # define the three circles of obstacle
        self.obstacle_circles_centers_tuple = compute_centers_of_approximation_circles(configuration.static_obstacle["position_x"],
                                                                                       configuration.static_obstacle["position_y"],
                                                                                       configuration.static_obstacle["length"],
                                                                                       configuration.static_obstacle["width"],
                                                                                       configuration.static_obstacle["orientation"])

        # get approximate radius for ego and obstacle vehicles
        self.radius_obstacle, _ = compute_approximating_circle_radius(configuration.static_obstacle["length"], configuration.static_obstacle["width"])
        self.radius_ego, _ = compute_approximating_circle_radius(configuration.p.l, configuration.p.w)
    
    def equal_constraints(self, states, ref_states, controls, f):
        pass

    def inequal_constraints(self, *args, **kwargs):
        pass

    def cost_function(self, states, controls, reference_states):
        obj = 0
        # define penalty matrices
        Q = np.array([[self.weights_setting["weight_x"], 0.0, 0.0, 0.0, 0.0], 
                      [0.0, self.weights_setting["weight_y"], 0.0, 0.0, 0.0],
                      [0.0, 0.0, self.weights_setting["weight_steering_angle"], 0.0, 0.0], 
                      [0.0, 0.0, 0.0, self.weights_setting["weight_velocity"], 0.0],
                      [0.0, 0.0, 0.0, 0.0, self.weights_setting["weight_heading_angle"]]])
          
        R = np.array([[self.weights_setting["weight_velocity_steering_angle"], 0.0], [0.0, self.weights_setting["weight_long_acceleration"]]])
        
        P = np.diag([self.weights_setting["weight_x_terminate"], self.weights_setting["weight_y_terminate"], self.weights_setting["weight_steering_angle_terminate"],
                     self.weights_setting["weight_velocity_terminate"], self.weights_setting["weight_heading_angle_terminate"]])
        # cost
        for i in range(self.predict_horizon):
            # obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
            obj = obj + (states[:, i] - reference_states[:, i+1]).T @ Q @ (states[:, i] - reference_states[:, i+1]) + controls[:, i].T @ R @ controls[:, i]
            + (states[:, -1] - reference_states[:, -1]).T @ P @ (states[:, -1] - reference_states[:, -1])
        return obj

    def solver(self):
        pass

    def optimize(self):
        pass