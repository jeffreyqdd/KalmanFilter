#!/usr/bin/env python3

import time
import numpy as np
from kalman.kalman_base import LinearKalman
import matplotlib.pyplot as plt


class Car2D:
    """Note that this car is only capable of moving in the X/Y direction. It has
    no control over heading for simplicity's sake. Assume uniform drag"""

    def __init__(self, mass, throttle_force):
        """
        mass: (kg)
        pos: m
        vel: m/s
        acc: m/s^2
        throttle_force : N
        """

        self.pos_x = 0
        self.vel_x = 0
        self.accel_x = 0

        self.pos_y = 0
        self.vel_y = 0
        self.accel_y = 0

        self.mass = mass
        self.throttle_force = throttle_force

        # these are forces I made up
        self.drag_coeff = 0.5
        self.cross_area = 10
        self.fluid_density = 1.293

        # we know the variances of our onboard sensors
        self.IMU_STD_DEV = 0.1  # 68% of all measurements fall between 0.1 m/s^2 of true value
        self.DVL_STD_DEV = 0.001  # 0.01 m/s type error

        assert throttle_force > 0

    def step(self, throttle_x, throttle_y, dt):
        assert throttle_x >= -1 and throttle_x <= 1
        assert throttle_y >= -1 and throttle_y <= 1
        assert dt > 0

        # calculate net forces
        # gas
        force_x_gas = self.throttle_force * throttle_x
        force_y_gas = self.throttle_force * throttle_y

        # drag
        force_x_drag = 0.5 * self.drag_coeff * \
            self.vel_x ** 2 * self.fluid_density * self.cross_area
        force_y_drag = 0.5 * self.drag_coeff * \
            self.vel_y ** 2 * self.fluid_density * self.cross_area

        force_x_net = force_x_gas - force_x_drag * np.sign(self.vel_x)
        force_y_net = force_y_gas - force_y_drag * np.sign(self.vel_y)

        accel_x_net = force_x_net / self.mass
        accel_y_net = force_y_net / self.mass

        # step
        self.pos_x += self.vel_x * dt + 0.5 * self.accel_x * dt ** 2
        self.vel_x += self.accel_x * dt
        self.accel_x = accel_x_net

        self.pos_y += self.vel_y * dt + 0.5 * self.accel_y * dt ** 2
        self.vel_y += self.accel_y * dt
        self.accel_y = accel_y_net

    def measure(self):
        # We know the error of our sensors, so we generate noise and add it to
        # the true value

        x_vel_DVL = self.vel_x + np.random.normal(0, self.DVL_STD_DEV, 1)[0]
        y_vel_DVL = self.vel_y + np.random.normal(0, self.DVL_STD_DEV, 1)[0]

        x_accel_IMU = self.accel_x + \
            np.random.normal(0, self.IMU_STD_DEV, 1)[0]
        y_accel_IMU = self.accel_y + \
            np.random.normal(0, self.IMU_STD_DEV, 1)[0]

        ret = np.array([x_vel_DVL, x_accel_IMU,  y_vel_DVL, y_accel_IMU])
        return ret

    def __repr__(self):
        return f"{self.pos_x:5.2f} {self.vel_x:5.2f} {self.accel_x:5.2f}"


if __name__ == '__main__':
    dt = 1/30
    vehicle = Car2D(10, 40)

    # measure still baseline

    kalman_filter = LinearKalman(
        x0=np.array([0, 0, 0, 0, 0, 0]),
        P0=np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
        # we can only measure a_x and a_y
        H=np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]),
        zn_size=4,
        R=np.array([
            [vehicle.DVL_STD_DEV ** 0.5, 0., 0., 0.],
            [0., vehicle.IMU_STD_DEV ** 0.5, 0., 0.],
            [0., 0., vehicle.DVL_STD_DEV ** 0.5, 0.],
            [0., 0., 0., vehicle.IMU_STD_DEV ** 0.5],
        ]),
        F=np.array([
            [1, dt, 0.5 * dt ** 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt ** 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]),
    )

    kalman_filter.predict()

    real_posx = []
    est_posx = []
    var_posx = []
    var_velx = []
    var_accx = []
    dt_lst = []

    for i in range(10):
        vehicle.step(0.1, 0, dt)
        m = vehicle.measure()

        kalman_filter.predict()
        kalman_filter.update(m)
        xn, Pn = kalman_filter.measure()

        real_posx.append(vehicle.pos_x)
        est_posx.append(xn[0])
        var_posx.append(Pn[0, 0])
        var_velx.append(Pn[1, 1])
        var_accx.append(Pn[2, 2])
        dt_lst.append(i)

    # 95% confidence interval
    real_posx = np.array(real_posx)
    est_posx = np.array(est_posx)
    var_posx = np.array(var_posx)

    std_deviations = var_posx ** 0.5
    ci = 1.96 * std_deviations
    print(var_posx)
    print(var_velx)
    print(var_accx)
    plt.plot(dt_lst, real_posx, label='real')
    plt.plot(dt_lst, est_posx, label='est')
    plt.fill_between(dt_lst, (est_posx-ci), (est_posx+ci), alpha=.1)
    plt.legend()
    plt.show()
