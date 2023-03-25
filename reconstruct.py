import numpy as np
import cv2

GT_t = np.array([0.1, -0.1, 0.1])
GT_R = np.eye(3)


def sym_skew_m(t): return np.array(
    [0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]).reshape(3, 3)


GT_t_hat = sym_skew_m(GT_t)

GT_E = GT_t_hat @ GT_R

d = 1
n = np.array([0, 0, -1])
GT_H = GT_R + np.outer(GT_t, n)

print(f"{GT_t_hat=}")
print(f"{GT_E=}")
print(f"{GT_H=}")

# Manual labling data
CORR_PNT_4_L = np.array(
    [248.20, 152.40, 448.90, 153.40, 248.10, 235.20, 448.60, 235.40]).reshape(4, 2)
CORR_PNT_4_R = np.array(
    [206.90, 183.70, 420.80, 183.80, 207.30, 271.80, 420.80, 271.60]).reshape(4, 2)
CORR_PNT_8_L = np.array([164.20, 169.60, 436.70, 169.40, 215.70, 192.80, 398.40, 192.70,
                        247.20, 345.00, 364.40, 345.40, 227.30, 375.60, 377.20, 375.80]).reshape(8, 2)
CORR_PNT_8_R = np.array([123.40, 197.00, 411.70, 196.60, 190.60, 211.80, 380.20, 211.80,
                        222.20, 372.30, 344.80, 372.30, 194.30, 410.60, 352.00, 410.80]).reshape(8, 2)
K = np.array([530.4669406576809, 0.0, 320.5, 0.0,
             530.4669406576809, 240.5, 0.0, 0.0, 1.0]).reshape(3, 3)


def recover_P(p_l, p_r, R, t_hat):
    print(p_l, p_r)
    P_l = p_l.reshape(-1, 1)
    t = np.array([t_hat[2, 1], t_hat[0, 2], t_hat[1, 0]])
    P_r = np.linalg.inv(R) @ (p_r - t).reshape(-1, 1)
    return P_l, P_r


def triangulate(kp1, kp2, P1, P2):
    A = np.vstack([
        kp1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - kp1[0] * P1[2, :],
        kp2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - kp2[0] * P2[2, :]
    ])
    u, s, vh = np.linalg.svd(A)
    return vh.T[:, -1] / vh.T[:, -1][-1]


def check_rt(R, t, point1, point2):
    P1 = np.hstack([K, np.zeros((3, 1))])  # 3 * 4
    P2 = np.zeros((3, 4))
    P2[:, :3] = R
    P2[:, -1] = t.flatten()
    P2 = K @ P2

    # print(f"{P1=}\n{P2=}")

    o1 = np.zeros((3, 1))
    o2 = -R.T @ t

    score = 0

    p3d_c1s = []

    for p1, p2 in zip(point1, point2):
        # Check visibility constraints and low parallax
        p3d_c1 = triangulate(p1, p2, P1, P2)
        # print(p3d_c1.shape)
        normal1 = p3d_c1[:3] - o1.flatten()
        dist1 = np.linalg.norm(normal1)
        # print(f"{normal1=} {dist1=}")

        normal2 = p3d_c1[:3] - o2.flatten()
        dist2 = np.linalg.norm(normal2)
        # print(f"{dist1=} {dist2=}")

        cos_parallax = np.inner(normal1, normal2) / (dist1 * dist2)
        # print(cos_parallax)

        if p3d_c1[2] < 0 and cos_parallax < 0.99998:
            continue

        p3d_c2 = R @ p3d_c1[:3].reshape(3, -1) + t

        if p3d_c2[2] < 0 and cos_parallax < 0.99998:
            continue

        # Check reprojection error
        p3d_c1_proj = K[:2, :2] @ p3d_c1[:2] / p3d_c1[2] + K[:2, -1]
        square_error1 = np.square(np.linalg.norm(p3d_c1_proj - p1[:2]))

        p3d_c2_proj = K[:2, :2] @ p3d_c2[:2] / p3d_c2[2] + K[:2, -1]
        square_error2 = np.square(np.linalg.norm(p3d_c2_proj - p2[:2]))

        print(f"{square_error1=}, {square_error2=}")

        score += 1
        p3d_c1s.append(p3d_c1)

    return score, p3d_c1s


def eight_points_method():
    # Compute calibrated points

    homo_8_l = np.hstack([CORR_PNT_8_L, np.ones((CORR_PNT_8_L.shape[0], 1))])
    homo_8_r = np.hstack([CORR_PNT_8_R, np.ones((CORR_PNT_8_R.shape[0], 1))])
    # print(f"{homo_8_l=}, {homo_8_r=}")
    K_inv = np.linalg.inv(K)

    norm_homo_8_l = (K_inv @ homo_8_l.T).T
    norm_homo_8_r = (K_inv @ homo_8_r.T).T

    # print(f"{norm_homo_8_l=}, {norm_homo_8_r=}")

    a = np.kron(norm_homo_8_l[0, :], norm_homo_8_r[0, :])
    for i in range(1, 8):
        a = np.vstack([a, np.kron(norm_homo_8_l[i, :], norm_homo_8_r[i, :])])
    print(f"{a=}, {a.shape=}")

    # Compute essential matrix
    u, s, vh = np.linalg.svd(a)
    # print(u, s, vh)
    # print(f"{vh.shape}")
    Es = vh.T[:, -1]
    print(f"{Es=} {a @ Es=}")

    E = Es.reshape(3, 3)  # Es is stacking from the column vector of E
    print(f"{E=}")

    # Check epipolar constrain
    for i in range(8):
        print(
            f"{norm_homo_8_l[i, :].reshape(1, -1) @ E @ norm_homo_8_r[i, :].reshape(-1, 1)=}")

    # Decompose essential matrix
    u, s, vh = np.linalg.svd(E)
    print(f"{s=}")
    s = np.mean(s[:2])
    # s = 1
    sigma = np.diag([s, s, 0])
    print(f"{sigma=}")
    # _, s, _ = np.linalg.svd(E_proj)
    print(f"{u @ sigma @ vh=}")

    # for i in range(8):
    #     print(f"{norm_homo_8_l[i, :].reshape(1, -1) @ E_proj @ norm_homo_8_r[i, :].reshape(-1, 1)=}")

    def Rz(y): return np.array(
        [np.cos(y), -np.sin(y), 0, np.sin(y), np.cos(y), 0, 0, 0, 1]).reshape(3, 3)
    R1 = u @ Rz(np.deg2rad(90)).T @ vh
    R2 = u @ Rz(np.deg2rad(-90)).T @ vh
    t_hat1 = u @ Rz(np.deg2rad(90)) @ sigma @ u.T
    t_hat2 = u @ Rz(np.deg2rad(-90)) @ sigma @ u.T
    print(f"{t_hat1=}\n{t_hat2=}")
    t1 = np.array([t_hat1[2, 1], t_hat1[0, 2], t_hat1[1, 0]])
    t2 = np.array([t_hat2[2, 1], t_hat2[0, 2], t_hat2[1, 0]])
    # t1 = u[:, -1]
    # t2 = u[:, -1]
    print(f"{R1=}\n {R2=}\n {t1=}\n {t2=}\n")

    # # Choose best relative pose(R, t)
    # print(
    #     f"R1, t1: {recover_P(norm_homo_8_l[0, :], norm_homo_8_r[0, :], R1, t_hat1)}")
    # print(
    #     f"R1, t2: {recover_P(norm_homo_8_l[0, :], norm_homo_8_r[0, :], R1, t_hat2)}")
    # print(
    #     f"R2, t1: {recover_P(norm_homo_8_l[0, :], norm_homo_8_r[0, :], R2, t_hat1)}")
    # print(
    #     f"R2, t2: {recover_P(norm_homo_8_l[0, :], norm_homo_8_r[0, :], R2, t_hat2)}")
    Rs = [-R1, -R1, R2, R2]
    ts = [t1, t2, t1, t2]

    for R, t in zip(Rs, ts):
        P1 = np.hstack([K, np.zeros((3, 1))])  # 3 * 4
        P2 = np.zeros((3, 4))
        P2[:, :3] = R
        P2[:, -1] = t.flatten()
        # print(P2.shape, K.shape)
        P2 = K @ P2

        print(f"{P1=}\n{P2=}")

        for i in range(norm_homo_8_l.shape[0]):
            print(check_rt(norm_homo_8_l[i, :], norm_homo_8_r[i, :], P1, P2))


def four_points_method():
    homo_4_l = np.hstack([CORR_PNT_4_L, np.ones((CORR_PNT_4_L.shape[0], 1))])
    homo_4_r = np.hstack((CORR_PNT_4_R, np.ones((CORR_PNT_4_R.shape[0], 1))))

    # From image plane to normal plane
    norm_homo_4_l = (homo_4_l[:, :2] - K[:2, -1]) / K[0, 0]
    norm_homo_4_r = (homo_4_r[:, :2] - K[:2, -1]) / K[0, 0]
    norm_homo_4_l = np.hstack(
        [norm_homo_4_l, np.ones((CORR_PNT_4_L.shape[0], 1))])
    norm_homo_4_r = np.hstack(
        [norm_homo_4_r, np.ones((CORR_PNT_4_R.shape[0], 1))])

    # a = np.kron(norm_homo_4_l[0, :], sym_skew_m(norm_homo_4_r[0, :]).T)
    # for i in range(1, 4):
    #     a = np.vstack(
    #         [a, np.kron(norm_homo_4_l[i, :], sym_skew_m(norm_homo_4_r[i, :]).T)])
    a = np.array([])
    for i in range(4):
        x, y = norm_homo_4_l[i, :2]
        xp, yp = norm_homo_4_r[i, :2]
        a = np.append(a, [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        a = np.append(a, [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        print(a.shape)
    a = a.reshape(-1, 9)
    # assert (a.shape == (12, 9))

    print(f"{a=}")

    _, _, vh = np.linalg.svd(a)
    Hs = vh.T[:, -1]
    assert (np.sum(a @ Hs) < 1e-5)
    print(f"{a @ Hs=}")
    print(f"{norm_homo_4_l}\n{norm_homo_4_r}")

    H = Hs.T.reshape(3, 3)
    H /= H[2, 2]
    H_CV, status = cv2.findHomography(norm_homo_4_l, norm_homo_4_r)
    assert (np.sum(np.abs(H_CV - H)) < 10e-5)
    print(f"{H=}\n{H_CV=}\n{GT_H=}")

    # Check planar epipolar constraint
    for i in range(4):
        print(
            f"Planar epipolar constraint: {sym_skew_m(norm_homo_4_r[i, :]) @ H @ norm_homo_4_l[i, :].reshape(-1, 1)}")

    # Decompose homograph(Faugeras)
    u, w, vh = np.linalg.svd(H)
    s = np.linalg.det(u) * np.linalg.det(vh)
    d1 = w[0]
    d2 = w[1]
    d3 = w[2]

    assert (1.0 < d1 / d2 and 1.0 < d2 / d3 and "Ensure the order")

    Rs = []
    ts = []
    ns = []

    x1 = np.sqrt((d1 ** 2 - d2 ** 2) / (d1 ** 2 - d3 ** 2))
    x3 = np.sqrt((d2 ** 2 - d3 ** 2) / (d1 ** 2 - d3 ** 2))

    # 1. sigma1 = 1, sigma3 = 1
    # 2. sigma1 = 1, sigma3 = -1
    # 3. sigma1 = -1, sigma3 = 1
    # 4. sigam1 = -1, sigam3 = -1
    x1s = [x1, x1, -x1, -x1]
    x2 = 0
    x3s = [x3, -x3, x3, -x3]

    # dp > 0
    sin_theta = np.sqrt((d1 ** 2 - d2 ** 2) *
                        (d2 ** 2 - d3 ** 2)) / ((d1 + d3) * d2)
    cos_theta = (d2 ** 2 + d1 * d3) / ((d1 + d3) * d2)
    sin_thetas = [sin_theta, -sin_theta, -sin_theta, sin_theta]

    for i in range(4):
        Rp = np.eye(3)
        Rp[0, 0] = cos_theta
        Rp[0, 2] = -sin_thetas[i]
        Rp[2, 0] = sin_thetas[i]
        Rp[2, 2] = cos_theta

        R = s * u @ Rp @ vh
        Rs.append(R)

        tp = (d1 - d3) * np.array([x1s[i], 0, -x3s[i]]).reshape(-1, 1)
        t = u @ tp
        ts.append(t / np.linalg.norm(t))  # FIXME: normalize?

        npp = np.array([x1s[i], 0, x3s[i]]).reshape(-1, 1)
        n = vh.T @ npp
        if n[2] < 0:  # FIXME: ?
            n = -n
        ns.append(n)

    # dp < 0
    sin_phi = np.sqrt((d1**2 - d2**2) * (d2**2 - d3**2)) / ((d1 - d3) * d2)
    cos_phi = (d1 * d3 - d2**2) / ((d1 - d3) * d2)
    sin_phis = [sin_phi, -sin_phi, -sin_phi, sin_phi]

    for i in range(4):
        Rp = np.eye(3)
        Rp[0, 0] = cos_phi
        Rp[0, 2] = sin_phis[i]
        Rp[1, 1] = -1
        Rp[2, 0] = sin_phis[i]
        Rp[2, 2] = -cos_phi

        R = s * u @ Rp @ vh
        Rs.append(R)

        tp = (d1 + d3) * np.array([x1s[i], 0, x3s[i]]).reshape(-1, 1)
        t = u @ tp
        ts.append(t / np.linalg.norm(t))

        npp = np.array([x1s[i], 0, x3s[i]]).reshape(-1, 1)
        n = vh.T @ npp
        if n[2] < 0:
            n = -n
        ns.append(n)

    print("Rs: ", Rs)
    print("ts: ", ts)
    print("ns: ", ns)

    for R, t, n in zip(Rs, ts, ns):
        print(f"score: {check_rt(R, t, norm_homo_4_l, norm_homo_4_r)}")


def decompose_essential():
    gt_t = np.array([0, 2, 0])
    t_hat = sym_skew_m(gt_t)
    gt_R = np.eye(3)

    E = t_hat @ gt_R
    print(f"{gt_t=}\n {gt_R=}\n {E=}")

    u, s, vh = np.linalg.svd(E)
    print(f"{u=}\n{s=}\n{vh=}")

    def Rz(y): return np.array(
        [np.cos(y), -np.sin(y), 0, np.sin(y), np.cos(y), 0, 0, 0, 1]).reshape(3, 3)
    R1 = u @ Rz(np.deg2rad(90)).T @ vh
    R2 = u @ Rz(np.deg2rad(-90)).T @ vh
    t_hat1 = u @ Rz(np.deg2rad(90)) @ np.diag(s) @ u.T
    t_hat2 = u @ Rz(np.deg2rad(-90)) @ np.diag(s) @ u.T

    t1 = np.array([t_hat1[2, 1], t_hat1[0, 2], t_hat1[1, 0]])
    t2 = np.array([t_hat2[2, 1], t_hat2[0, 2], t_hat2[1, 0]])
    print(f"{R1=}\n{R2=}\n{t1=}\n{t2=}\n")


if __name__ == "__main__":
    # eight_points_method()
    four_points_method()
    # decompose_essential()
