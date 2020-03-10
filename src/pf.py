#!/usr/bin/env python
import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.random import randn
from numpy.random import uniform
from numpy.linalg import norm
import scipy.stats


class PF:
    # state_vector_size = 3
    # control_size = 2
    # measurement_size = 2
    def __init__(self, state_vector_size, control_size, measurement_size, N):
        self.state_vector_size = state_vector_size
        self.control_size = control_size
        self.beacons = []
        self.beaconsCount = 5
        self.dt = 1 # time between measurements
        self.state_vector = (np.array((5., 5., 0.)))[:,None] # x, y, theta at start based on room size
        self.velocity_vector = np.zeros((2, 1)) # v, w (vel., ang. vel.)
        # Create particles and weights
        self.numParticles = N
        self.particles = self.create_uniform_particles((0,10),(0,10),(-np.pi,np.pi), N)
        self.weights = np.ones(self.numParticles) / self.numParticles
        self.sensor_std_err = 1.4
        self.predict_std = (.2, .05) # std heading change, std velocity
        self.estimated_state_vector = self.state_vector
        self.estimated_variance = self.state_vector
        np.random.seed(42)


    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] = wrap_to_pi(particles[:, 2])
        return particles

    def create_gaussian_particles(self, mean, std, N):
        particles = np.empty((N, 3))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        particles[:, 2] = mean[2] + (randn(N) * std[2])
        particles[:, 2] = wrap_to_pi(particles[:, 2])
        return particles

    def predict(self):
        """ 
        Move according to control input (heading change, velocity)
        with noise Q (std heading change, std velocity)
        """
        # States
        x = self.state_vector[0]
        y = self.state_vector[1]
        theta = self.state_vector[2]
        
        # Velocities
        v = self.velocity_vector[0]
        w = self.velocity_vector[1]
        
        # Expected value of error
        dv = 0
        dw = 0
        
        # Non linear motion model when there's no angular velocity
        if abs(w + dw) > 1e-4:
            d = (v + dv) / (w + dw)
            # x_i+1
            f1 = x + d * (-np.sin(theta) + np.sin(theta + (w + dw) * self.dt))
            # y_i+1
            f2 = y + d * (np.cos(theta) - np.cos(theta + (w + dw) * self.dt))
            
        # Linear motion model
        else:
            dx = (v + dv) * self.dt
            # x_i+1
            f1 = x + dx * np.cos(theta)
            # y_i+1
            f2 = y + dx * np.sin(theta)
            
        # theta_i+1
        f3 = theta + (w + dw) * self.dt

        self.state_vector = np.array((f1, f2, wrap_to_pi(f3)))


        N = len(self.particles)

        # update heading
        self.particles[:, 2] += f3 + (randn(N) * self.predict_std[0])
        self.particles[:, 2] = wrap_to_pi(self.particles[:, 2])

        # move in the (noisy) commanded direction
        distx = f1 + (randn(N) * self.predict_std[1])
        disty = f2 + (randn(N) * self.predict_std[1])
        self.particles[:, 0] = distx
        self.particles[:, 1] = disty


    def set_params(self, v, w, dt):
        """
        Method for setting velocity, angular velocity and elapsed time from
        odometry
        """
        self.velocity_vector = np.array((v, w))
        self.dt = dt


    def set_beacons(self, beacons):
        """
        Sets beacons to an array and updates size of error matrix
        """
        self.beacons = beacons
        self.R = 0.5 * np.identity(2 * len(beacons))


    def update(self, particles, weights, z, R, landmarks):
        weights.fill(1.)

        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.e-300      # avoid round-off to zero
        weights /= sum(weights) # normalize

        # resample if too few effective particles
        if self.neff(weights) < self.numParticles/2:
            indexes = systematic_resample(weights)
            self.resample_from_index(particles, weights, indexes)

        self.estimate(particles, weights)


    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        pos = particles[:, 0:2]
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)
        self.estimated_state_vector[0] = mean[0]
        self.estimated_state_vector[1] = mean[1]
        #self.estimated_variance[0] = var[0]
        #self.estimated_variance[1] = var[1]
        return mean, var


    def simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random(N))
        # resample according to indexes
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)


    def resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights.fill(1.0 / len(weights))
        

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))
        

    def propagate_state(self):
        """
        Use beacons measurements
        """
        if len(self.beacons) == 0:
            return

        # distance from robot to each landmark
        beacons = [(7.3, 3), (1, 1), (9, 9), (1, 8), (5.8, 8)] # xy coordinates
        zs = (norm(beacons - self.state_vector[0:1], axis=1) + 
              (randn(self.beaconsCount) * self.sensor_std_err))

        self.update(self.particles, self.weights, z=zs, R=self.sensor_std_err, 
               landmarks=beacons)


def wrap_to_pi(angle):
    """
    Wraps the input angle between [-pi,pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi