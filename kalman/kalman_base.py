import numpy as np

from typing import Union, Callable, TypeVar, Tuple
from abc import ABC, abstractclassmethod


class Generics:
    T = TypeVar("T")


class KalmanBase(ABC):
    @abstractclassmethod
    def predict(self):
        raise NotImplemented('KalmanBase.predict')

    @abstractclassmethod
    def update(self):
        raise NotImplemented('KalmanBase.update')


def _guard_none(item: Generics.T, alt: Callable[[None], Generics.T]) -> Generics.T:
    if item is None:
        return alt()
    else:
        return item


class LinearKalman(KalmanBase):
    def __init__(self,
                 x0: np.ndarray,
                 P0: np.ndarray,
                 H: np.ndarray,
                 zn_size: int,
                 R: np.ndarray,
                 F: Union[np.ndarray, None] = None,
                 G: Union[np.ndarray, None] = None,
                 un_size: Union[int, None] = None,
                 Q: Union[np.ndarray, int] = 1,
                 ):
        """ Constructor for a linear kalman filter
        # Params
            x0      : initial state of the system
            P0      : initial uncertainty in x0 (covariance)
            H       : observation matrix (maps state-space into measurement-space)
            zn_size : number of measured states 
            R       : measurement covariances
            F       : state transition matrix
            G       : control matrix (maps input-space into state-space)
            un_size : number of input states
            Q       : process noise covariance
        """

        # Assert preconditions
        # G and un must come together
        if (G is None) ^ (un_size is None):
            raise ValueError('G and un_shape must be both None or non-None')

        state_size, = x0.shape

        # Recall the algorithm starts at timestep n=0
        self.xn: np.ndarray = x0
        self.Pn: np.ndarray = P0  # process covariance
        self.H: np.ndarray = H  # observation matrix
        self.R: np.ndarray = R  # measurement covariance

        self.F: np.ndarray = _guard_none(F, lambda: np.identity(state_size))
        self.un_size: int = _guard_none(un_size, lambda: 1)
        self.G: np.ndarray = _guard_none(
            G, lambda: np.zeros(shape=(state_size, self.un_size)))
        self.zn_size: int = zn_size
        self.I = np.identity(state_size)

        if isinstance(Q, int):
            self.Q: np.ndarray = np.eye(state_size) * Q
        else:
            self.Q: np.ndarray = Q
        print(Q)

        # assert correct dimensions
        assert self.Pn.shape == (state_size, state_size)
        assert self.F.shape == (state_size, state_size)
        assert self.G.shape == (state_size, self.un_size)
        assert self.Q.shape == (state_size, state_size)
        assert self.H.shape == (zn_size, state_size)
        assert self.R.shape == (zn_size, zn_size)

    def predict(self,
                u_n: Union[np.ndarray, None] = None,
                ) -> None:
        """ The predict step of the kalman filter
        # Params
            u_n : the input vector to the system. Assumed to be the zero vector
                    if left empty. 
        """
        # fill with zeros
        u_n = _guard_none(u_n, lambda: np.zeros(shape=(self.un_size, )))

        # update estimate: x_n+1 = Fx_n + Gu_n
        self.xn = self.F @ self.xn + self.G @ u_n

        # update covariances: P_n+1  = F P_n F^T + Q
        self.Pn = self.F @ self.Pn @ self.F.T + self.Q

    def update(self, z_n) -> None:
        """The update step of the kalman filter
        # Params:
            zn : the measurement vector 
        """

        kalman_gain = self.Pn @ self.H.T @ np.linalg.inv(
            (self.H @ self.Pn @ self.H.T + self.R))
        self.xn = self.xn + kalman_gain @ (z_n - self.H @ self.xn)

        intermediate = (self.I - kalman_gain @ self.H)
        self.Pn = intermediate @ self.Pn @ intermediate.T + \
            kalman_gain @ self.R @ kalman_gain.T

    def measure(self) -> Tuple[np.ndarray, np.ndarray]:
        """ returns current state and associated covariance.
        """
        return self.xn, self.Pn
