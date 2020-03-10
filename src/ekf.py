#!/usr/bin/env python
import numpy as np

class EKF:
    # state_vector_size = 3
    # control_size = 2
    # measurement_size = 2
    def __init__(self, state_vector_size, control_size, measurement_size):
        self.state_vector_size = state_vector_size
        self.control_size = control_size
        self.beacons = []
        self.dt = 1 # time between measurements
        self.state_vector = (np.array((5., 5., 0.)))[:,None] # x, y, theta at start based on room size
        self.velocity_vector = np.zeros((2, 1)) # v, w (vel., ang. vel.)
        self.P = np.identity(state_vector_size) 
        self.q = np.array([[0.01, 0.] , [0., 0.015]]) # variance of error through testing
        self.R = 0.5 * np.identity(measurement_size) # error from datasheet etc.
        self.F = np.zeros((state_vector_size, state_vector_size)) # F
        self.G = np.zeros((state_vector_size, control_size)) # G
        self.H = np.zeros((measurement_size, state_vector_size)) # H
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.K = np.zeros((state_vector_size, control_size))


    """
    PREDICTION PART BEGINS HERE
    """    
    
    def predict(self):
        """
        Uses motion model to predict the location of the robot.
        Does not take into account the beacon measurements (for that see
        propagate_state)
        """
        # Use old state for Jacobians
        self.motion_jacobian_state_vector()        
        self.motion_jacobian_noise_components()   
        
        # Predicted state and covariance without observations       
        self.predict_state()    
        self.calculate_cov()       


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
    
    
    def predict_state(self):
        """
        Predicts state of robot solely based on motion model.
        This could be run alone and it would give out somewhat decent
        results atleast in shorter time periods
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
        f3 = wrap_to_pi(theta + (w + dw) * self.dt)      
        self.state_vector = np.array((f1, f2, f3))
    

    def calculate_cov(self):
        """
        Calculate predicted covariance matrix (P). 
        Does not take into account the beacon measurements (for that see
        propagate_state)
        """
        self.Q = np.matmul(self.G, self.q)
        self.Q = np.matmul(self.Q, self.G.transpose())
        
        self.P = np.matmul(self.F, self.P)
        self.P = np.matmul(self.P, self.F.transpose()) + self.Q


    def motion_jacobian_state_vector(self):
        """
        Calculate Jacobian matrix of motion model in relation to the state 
        vector (F)
        """
        # States
        theta = self.state_vector[2]
        
        # Velocities
        v = self.velocity_vector[0]
        w = self.velocity_vector[1]
        
        # Expected value of error
        dv = 0
        dw = 0
        
        # Identity matrix
        self.F = np.identity((self.state_vector_size))
        
        # Calculate Jacobian
        # Non linear motion model when there's no angular velocity
        if abs(w + dw) > 1e-4:      
            d = (v + dv) / (w + dw)
            self.F[0][2] = d * (-np.cos(theta) + np.cos(theta + (w + dw) * self.dt))
            self.F[1][2] = d * (-np.sin(theta) + np.sin(theta + (w + dw) * self.dt))

        # Linear motion model
        else:  
            self.F[0][2] = -np.sin(theta) * (v + dv) * self.dt
            self.F[1][2] = np.cos(theta) * (v + dv) * self.dt 
        

    def motion_jacobian_noise_components(self):
        """
        Calculate Jacobian matrix of motion model in relation to the noise 
        vector (G)
        """
        # States
        theta = self.state_vector[2]
        
        # Velocities
        v = self.velocity_vector[0]
        w = self.velocity_vector[1]
        
        # Expected value of error
        dv = 0
        dw = 0
        
        # Initialize
        self.G = np.zeros((self.state_vector_size, self.control_size))
        
        # Calculate Jacobian
        # Non linear motion model when there's no angular velocity
        if abs(w + dw) > 1e-4:
            angle = theta + (w + dw) * self.dt
            dc = np.cos(angle) - np.cos(theta)
            ds = np.sin(angle) - np.sin(theta)

            self.G[0][0] = ds / (w + dw)
            self.G[0][1] = (np.cos(angle) * (v + dv) * self.dt / (w + dw)) - (v + dv) * ds / np.square(w + dw)
            self.G[1][0] = -dc / (w + dw)
            self.G[1][1] = (v + dv) * dc / np.square(w + dw) + np.sin(angle) * (v + dv) * self.dt / (w + dw)
            self.G[2][0] = 0
            self.G[2][1] = self.dt

        # Linear motion model
        else:
            self.G[0][0] = np.sin(theta) * self.dt
            self.G[0][1] = np.sin(theta) * self.dt
        
    """
    POSTERIORI PART BEGINS HERE
    """   
            
    def propagate_state(self):
        """
        Uses Kalman filter and beacon measurements to propagate the state of 
        the robot
        """
        if len(self.beacons) == 0:
            return
        self.calculate_Kalman_gain()
        
        # New state
        self.state_vector = self.state_vector + np.matmul(self.K, self.innovation())   
        
        # New covariance
        KH = np.matmul(self.K, self.H)
        self.P = np.matmul((np.identity(self.state_vector_size) - KH), self.P)
        

    def calculate_Kalman_gain(self):
        """
        Calculates Kalman gain 
        """
        # Calculate components
        P = self.P
        
        self.observation_jacobian_state_vector()
        H = self.H
        
        D = np.matmul(np.matmul(H, P), H.transpose())

        self.K = np.matmul(np.matmul(P, H.transpose()), np.linalg.inv(D + self.R))
    
    
    def innovation(self):
        """
        Calculate the innovation term = what is measured vs what is expected to
        be measured with expected value
        """             
        z = np.zeros((2*len(self.beacons), 1))
        h = np.zeros((2*len(self.beacons), 1))
            
        i = 0
        for m in self.beacons:
            lx = m[1]
            ly = m[2]
            dx = m[3]
            dy = m[4]
            
            x = self.state_vector[0]
            y = self.state_vector[1]
            theta = self.state_vector[2]
        
            # Calculate h
            h[i][0] = np.sqrt(np.square(lx-x) + np.square(ly-y))
            h[i + 1][0] = wrap_to_pi(np.arctan2(ly-y, lx-x) - theta)
            
            # Calculate observation z
            z[i][0] = np.sqrt(np.square(dx) + np.square(dy))
            z[i + 1][0] = wrap_to_pi(np.arctan2(dy, dx))
            
            # Two rows were added
            i += 2
        
        innovation = z - h
        return innovation


    def observation_jacobian_state_vector(self):
        """
        Calculate Jacobian matrix of z (observation) in relation to the state
        vector
        """
        # Initialize
        self.H = np.zeros((2*len(self.beacons), self.state_vector_size))
        
        i = 0
        for m in self.beacons:
            lx = m[1]
            ly = m[2]
            
            x = self.state_vector[0]
            y = self.state_vector[1]
            
            d = np.sqrt(np.square(x - lx) + np.square(y - ly))
            
            # Calculate Jacobian
            self.H[i][0] = (x - lx) / d
            self.H[i][1] = (y - ly) / d
            self.H[i][2] = 0
            
            self.H[i + 1][0] = -(y - ly) / (np.square(x - lx) + np.square(y - ly))
            self.H[i + 1][1] = (np.square(x - lx) * (x + lx)) / ((np.square(x) + 1) * (np.square(x - lx) + np.square(y - ly)))
            self.H[i + 1][2] = -1
            
            # Two rows were added
            i += 2


    """
    UNRELATED STUFF BEGINS HERE
    """  

    def print_initials(self):
        print("Printing some values")
        print("\n")
        print("The initial stated is {}".format(self.state_vector))
        print("\n")
        print("The initial cov. matrix is {}".format(self.P))
        print("\n")


def wrap_to_pi(angle):
    """
    Wraps the input angle between [-pi,pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
        