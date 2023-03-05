import numpy as np

class SO3():
    def __init__(self, d: np.ndarray) -> None:
        self.__w = None
        self.__w_hat = None
        self.__R = None

        if d.size == 3 and d.ndim == 2:
            self.__w = d
        elif d.size == 9 and d.ndim == 2:
            self.__R = d
        else:
            raise Exception("d should be either a R3 vector or R3x3 matrix")

    @staticmethod
    def generate_ssm(w):
        wp = w.flatten()
        ssm = np.array([0.0, -wp[2], wp[1], wp[2], 0.0, -
                       wp[0], -wp[1], wp[0], 0.0]).reshape(3, 3)
        assert np.array_equal(ssm.T, -ssm)  # check skew-symmetric
        return ssm

    def exp(self):
        if self.__R is None and self.__w is None:
            raise Exception("Either R or w should be specified")

        if self.__R is None:
            if self.__w.size == 3 and np.array_equal(self.__w, np.zeros((3, 1))):
                self.__R = np.eye(3)
            else:
                # compute R, Rodrigues' Formula
                self.__w_hat = SO3.generate_ssm(self.__w)
                norm_w = np.linalg.norm(self.__w)
                self.__R = np.eye(3) \
                    + (self.__w_hat / norm_w * np.sin(norm_w)) \
                    + ((self.__w_hat @ self.__w_hat) /
                       (norm_w**2) * (1 - np.cos(norm_w)))
        return self.__R

    def log(self):
        if self.__w is None and self.__R is None:
            raise Exception("Either R or w should be specified")

        if self.__w is None:
            if np.array_equal(self.__R, np.eye(3)):
                self.__w = np.zeros((3, 1))
                self.__w_hat = SO3.generate_ssm(self.__w)
            else:
                norm_w = np.arccos((np.trace(self.__R) - 1) / 2)
                r1 = np.array([self.__R[2, 1], self.__R[0, 2], self.__R[1, 0]])
                r2 = np.array([self.__R[1, 2], self.__R[2, 0], self.__R[0, 1]])
                r_diff = (r1 - r2).reshape(3, 1)
                self.__w = 1 / (2 * np.sin(norm_w)) * r_diff * norm_w
                self.__w_hat = SO3.generate_ssm(self.__w)

        return self.__w

    @property
    def w(self):
        return self.log()

    @property
    def w_hat(self):
        return SO3.generate_ssm(self.log())

    def dist(self, R2):
        '''Angular distance
        '''
        a = SO3(self.__R.T @ R2.exp())
        return np.linalg.norm(a.log())
        

class SE3():
    def __init__(self, d: np.ndarray) -> None:
        self.__v = None
        self.__T = None
        self.__g = None
        self.__so3 = None
        self.__xi = None

        if d.size == 6 and d.ndim == 2:
          self.__xi = d
          self.__so3 = SO3(self.__xi[:3, :])
          self.__v = self.__xi[3:, :]
          self.__xi_hat = np.zeros((4, 4))
          self.__xi_hat[:3, :3] = self.__so3.w_hat
          self.__xi_hat[:3, -1] = self.__v.flatten()
          
        elif d.size == 16 and d.ndim == 2:
          self.__g = d
          self.__so3 = SO3(d[:3, :3])
          self.__T = d[:3, -1]
        else:
          raise Exception("d should be either a R6 vector or a R4x4 matrix")

    def exp(self):
        if self.__g is None and self.__xi is None:
            raise Exception("Either g or xi should be specified")
        
        if self.__g is None:
            self.__g = np.eye(4)
            self.__g[:3, :3] = self.__so3.exp()
            if np.array_equal(self.__so3.w, np.zeros((3, 1))):
                self.__T = self.__v
                self.__g[:3, -1] = self.__T.flatten()
            else:
                left_jacobian = ((np.eye(3) - self.__so3.exp()) @ self.__so3.w_hat + np.outer(self.__so3.w, self.__so3.w)) / (np.linalg.norm(self.__so3.w) ** 2)
                self.__T = left_jacobian @ self.__v 
                self.__g[:3, -1] = self.__T.flatten()
        return self.__g

    def log(self):
        if self.__g is None and self.__xi_hat is None:
            raise Exception("Either g or xi should be specified")
        
        if self.__xi is None:
            if np.array_equal(self.__so3.w, np.zeros((3, 1))):
                self.__v = self.__T.reshape(3, 1)
            else:
                left_jacobian = ((np.eye(3) - self.__so3.exp()) @ self.__so3.w_hat + np.outer(self.__so3.w, self.__so3.w)) / (np.linalg.norm(self.__so3.w) ** 2)
                self.__v = np.linalg.inv(left_jacobian) @ self.__T
                self.__v = self.__v.reshape(3, 1)
            self.__xi = np.vstack([self.__so3.w, self.__v])
        return self.__xi

    @property
    def so3(self):
        return self.__so3

    def dist(self, g):
        '''Double geodesic distance
        '''
        return np.sqrt(self.so3.dist(g.so3) ** 2 + np.linalg.norm(self.exp()[:, -1] - g.exp()[:, -1]) ** 2)

def generate_rotation_matrix():
    m = np.random.rand(3, 3)
    q, r = np.linalg.qr(m)
    assert np.allclose(q @ q.T, np.eye(3))
    assert np.abs(np.linalg.det(q) - 1.0) < 1e-5

    return q

def test_so3_no_rotation():
    r1 = SO3(np.eye(3))
    assert np.array_equal(r1.exp(), np.eye(3))
    assert np.array_equal(r1.log(), np.array([0, 0 ,0]).reshape(-1, 1))

    r2 = SO3(np.zeros((3, 1)))
    assert np.array_equal(r2.exp(), np.eye(3))
    assert np.array_equal(r2.log(), np.array([0, 0, 0]).reshape(-1, 1))

def test_so3_arbitrary_rotation():
    rm = generate_rotation_matrix()
    r1 = SO3(rm)
    r2 = SO3(r1.w)
    assert np.allclose(r2.exp(), rm)

def test_so3_repeated_convert():
    r1 = SO3(np.eye(3))
    for i in range(100):
        r2 = SO3(r1.log())
        r3 = SO3(r2.exp())
        r1 = r3
    assert np.array_equal(r1.exp(), np.eye(3))

def test_se3_identity_transform():
    g1 = SE3(np.eye(4))
    assert np.array_equal(g1.exp(), np.eye(4))
    assert np.array_equal(g1.log(), np.zeros((6, 1)))

def test_se3_aribitrary_transform():
    m = np.eye(4)
    m[:3, :3] = generate_rotation_matrix()
    m[:3, -1] = np.random.rand(3)
    g1 = SE3(m)
    g2 = SE3(g1.log())
    assert np.allclose(g2.exp(), m)

def test_so3_distance():
    r1 = SO3(np.eye(3))

    #rot_z = lambda t : np.array([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape(3, 3)
    # r2 = SO3(rot_z(np.deg2rad(90)))
    w = np.array([1.57, 1.57, 0]).reshape(3, 1)
    r2 = SO3(w)
    assert np.abs(r1.dist(r2) - np.linalg.norm(w)) < 1e-5

def test_se3_distance():
    g1 = SE3(np.eye(4))
    xi1 = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    g2 = SE3(xi1)
    assert np.abs(g1.dist(g2) - np.linalg.norm(xi1)) < 1e-5

    xi2 = np.array([1.57, 1.57, 1.57, 0, 0, 0]).reshape(-1, 1)
    g3 = SE3(xi2)
    assert np.abs(g1.dist(g3) - np.linalg.norm(xi2)) < 1e-5

if __name__== "__main__":
    pass
