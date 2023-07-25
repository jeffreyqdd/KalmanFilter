#!/usr/bin/env python3

from kalman.kalman_base import LinearKalman

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    measurements = [49.03, 48.44, 55.21, 49.98,
                    50.6, 52.61, 45.87, 42.64, 48.26, 55.84]
    xn_hst = []
    Pn_hst = []

    kalman_filter = LinearKalman(
        xn=np.array([60.]),
        P=np.array([[225.]]),
        H=np.array([[1.]]),
        zn_size=1,
        R=np.array([[25]])
    )

    kalman_filter.predict()
    for m in measurements:
        kalman_filter.predict()
        kalman_filter.update(m)
        xn, Pn = kalman_filter.measure()
        xn_hst.append(xn)
        Pn_hst.append(Pn.flatten())

    plt.plot(range(1, len(measurements)+1), xn_hst, label='pred')
    plt.plot(range(1, len(measurements)+1), measurements, label='meas')
    plt.ylim(30, 70)
    plt.show()
