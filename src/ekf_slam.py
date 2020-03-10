import numpy as np
import math


class EKF_SLAM:
    def __init__(self, num_landmarks, num_states, control_size, measurement_size):
        self.rsize = 3
        self.lmsize = 2
        self.dt = 1.

        self.mean = (np.array((5., 5., 0., -1., -1., -1., -1.,
                                      -1., -1., -1., -1., -1., -1.)))[:,None]
        self.cov = np.zeros((self.mean.size, self.mean.size))
        lm_cov = np.eye(2 * num_landmarks)
        np.fill_diagonal(
            lm_cov,
            10 ** 10
        )
        self.cov[self.rsize:, self.rsize:] = lm_cov

        self.motion_command = np.array(())
        self.observed_features = []
        self.velocity_vector = np.zeros((control_size, 1))

        self.R = 0.5 * np.identity(measurement_size)  # error from datasheet etc.
        self.Gtx = np.zeros((num_states)) # Motion jacobian
        self.printCounter = 0



    def predict(self):
        #robot_pose = self.mean[:self.rsize, :]
        #self.mean[:self.rsize, :] = self.motion_command(
         #   robot_pose,
        #    command
       # )
        # States
        x = self.mean[0]
        y = self.mean[1]
        theta = self.mean[2]

        # Velocities
        v = self.velocity_vector[0]
        w = self.velocity_vector[1]

        # Expected value of error
        dv = 0
        dw = 0

        #theta_n = self.wrap_to_pi(self.mean.item(2))
       # rot1, trans = command

        # Non linear motion model when there's no angular velocity
        f1 = 0
        f2 = 0
        d = 0
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
        ang = self.wrap_to_pi(theta + (w + dw) * self.dt)


        self.mean[0] = f1
        self.mean[1] = f2
        self.mean[2] = ang

        if len(self.observed_features) == 0:
            return


       # Gtx = np.matrix([
       #     [1, 0, -d * np.sin(ang)],
       #     [0, 1, d * np.cos(ang)],
       #     [0, 0, 1],
       # ])
        Gtx = np.identity(3)
        if abs(w + dw) > 1e-4:
            d = (v + dv) / (w + dw)
            Gtx[0][2] = d * (-np.cos(theta) + np.cos(theta + (w + dw) * self.dt))
            Gtx[1][2] = d * (-np.sin(theta) + np.sin(theta + (w + dw) * self.dt))

        # Linear motion model
        else:
            Gtx[0][2] = -np.sin(theta) * (v + dv) * self.dt
            Gtx[1][2] = np.cos(theta) * (v + dv) * self.dt

        self.Gtx = Gtx
        #Gtx[0][2] = -d * np.sin(ang)
        #Gtx[1][2] = d * np.cos(ang)


        lmsize = self.mean.shape[0] - self.rsize


        r1zeros = np.zeros((self.rsize, lmsize))
        r2zeros = np.copy(r1zeros.T)

        gr1 = np.concatenate((Gtx, r1zeros), axis=1)
        gr2 = np.concatenate((r2zeros, np.eye(lmsize)), axis=1)
        Gt = np.concatenate((gr1, gr2))

        # motion noise
    #    Rtx = np.matrix([
    #        [0.1, 0, 0],
    #        [0, 0.1, 0],
    #        [0, 0, 0.01]
    #    ])
        Rtx = np.identity(3, dtype=float)
        Rtx[0][0] = 0.1
        Rtx[1][1] = 0.1
        Rtx[2][2] = 0.01

        rr1zeros = np.zeros((self.rsize, lmsize))
        rr2zeros = np.copy(rr1zeros.T)

        rr1 = np.concatenate(
            (Rtx, rr1zeros),
            axis=1
        )
        rr2 = np.concatenate(
            (rr2zeros, np.zeros((lmsize, lmsize))),
            axis=1
        )
        Rt = np.concatenate((rr1, rr2))


        #return self.mean, self.cov

    def calculate_classical_Kalman_gain(self):
        """
        Calculates Kalman gain
        """
        # Calculate components

        HR = np.concatenate((self.Gtx, np.zeros((3, 10))), axis = 1)
        D = np.matmul(np.matmul(HR, self.cov), HR.transpose())

        K = np.matmul(np.matmul(self.cov, HR.transpose()), np.linalg.inv(D + self.R))
        return K


    def set_observed_features(self, observed_features):
        """
        Sets range, heading and id observed from beacon
        """
        self.observed_features = observed_features

    def update(self):
        if len(self.observed_features) == 0:
            return

        rx = self.mean[0]
        ry = self.mean[1]

        rtheta = self.mean[2]

        Htfull = np.matrix([])
        Zdiff = np.matrix([])

        for reading in self.observed_features:
            # TODO: Put in measurement model
            srange, sbearing, lid = reading


            # Expected observation
            if self.mean[2*lid + 1] == -1:
                lx = rx + srange * np.cos(sbearing + rtheta)
                ly = ry + srange * np.sin(sbearing + rtheta)

                self.mean[2*lid + 1] = lx
                self.mean[2*lid + 2] = ly

            lx = self.mean[2*lid + 1]
            ly = self.mean[2*lid + 2]

            dx = lx - rx
            dy = ly - ry



           # delta = np.matrix([dx, dy]).T
            delta = np.zeros((2, 1))
            delta[0] = dx
            delta[1] = dy
            q = np.matmul(delta.transpose(), delta)

           # z_expected = np.matrix([
          #      np.sqrt(q),
         #       self.wrap_to_pi(np.arctan2(dy, dx) - rtheta)
        #    ]).T
            z_measured = np.matrix([srange, self.wrap_to_pi(sbearing)]).T

            z_expected = np.zeros((2, 1))
            z_expected[0] = np.sqrt(q)
            z_expected[1] = self.wrap_to_pi(np.arctan2(dy, dx) - rtheta)

            qst = np.sqrt(q)
            # Measurement jacobian
            Htt = np.zeros((2, 5))
            Htt[0][0] = -qst*dx
            Htt[0][1] = -qst*dy
            Htt[0][2] = 0
            Htt[0][3] = qst*dx
            Htt[0][4] = qst*dy
            Htt[1][0] = dy
            Htt[1][1] = -dx
            Htt[1][2] = -q
            Htt[1][3] = -dy
            Htt[1][4] = dx



            Htt = 1.0 / q * Htt

            F = np.zeros((5, self.mean.size))
            F[:self.rsize, :self.rsize] = np.eye(self.rsize)
            F[self.rsize:, 2*lid + 1:2*lid +  3] = np.eye(self.lmsize)

            Ht = np.matmul(Htt, F)

            Htfull = np.concatenate((Htfull, Ht)) if Htfull.size else np.copy(Ht)

            diff = z_measured - z_expected
            # Important to self.wrap_to_pi()s
            diff[1] = self.wrap_to_pi(diff.item(1))

            Zdiff = np.concatenate((Zdiff, diff)) if Zdiff.size else np.copy(diff)

        # measurement noise
        Qt = np.eye(Zdiff.shape[0]) * 0.01
        D = np.matmul(np.matmul(Htfull, self.cov), Htfull.transpose()) + Qt
        Kgain = np.matmul(np.matmul(self.cov, Htfull.T), np.linalg.inv(D))

        self.mean = self.mean + np.matmul(Kgain, Zdiff)
        #self.cov = np.matmul((np.eye(self.mean.size) - np.matmul(Kgain, Htfull)), self.cov)
        self.cov = self.cov - np.matmul(np.matmul(Kgain, D), Kgain.transpose())

        self.printCounter += 1
        if self.printCounter == 10:
            print(self.mean)
            self.printCounter = 0


    def set_params(self, v, w, dt):
        """
        Method for setting velocity, angular velocity and elapsed time from
        odometry
        """
        self.velocity_vector = np.array((v, w))
        self.dt = dt
    def wrap_to_pi(self, angle):
        """
        Wraps the input angle between [-pi,pi)
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
