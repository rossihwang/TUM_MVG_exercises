import numpy as np
import cv2
import sys

Ix = None
Iy = None
Ixx = None
Iyy = None
Ixy = None


def generate_grid_point(frame_size: tuple, grid_size: tuple):
    for x in np.linspace(0, frame_size[1] - 1, grid_size[1]):
        for y in np.linspace(0, frame_size[0] - 1, grid_size[0]):
            yield (int(x), int(y))


def compute_harris_response(frame: np.ndarray, block_size: tuple):
    global Ix, Iy
    global Ixx, Iyy, Ixy

    frame = cv2.GaussianBlur(frame, (3, 3), 10)  # denoise
    Ix = cv2.Sobel(frame, cv2.CV_16S, 1, 0)
    Iy = cv2.Sobel(frame, cv2.CV_16S, 0, 1)

    # Ixx = np.abs(Ix * Ix)
    # Iyy = np.abs(Iy * Iy)
    # Ixy = np.abs(Ix * Iy)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Ixx_g = cv2.GaussianBlur(Ixx, block_size, 7)
    Iyy_g = cv2.GaussianBlur(Iyy, block_size, 7)
    Ixy_g = cv2.GaussianBlur(Ixy, block_size, 7)

    k = 0.01
    response = (Ixx_g * Iyy_g - Ixy_g * Ixy_g) - k * \
        ((Ixx_g + Iyy_g) * (Ixx_g + Iyy_g))
    response = (response - np.min(response)) / \
        (np.max(response) - np.min(response)) * 255
    response = response.astype(np.uint8)
    print(f"{np.mean(response)=}\n{np.std(response)=}\n{np.median(response)=}\n{np.max(response)=}\n{np.min(response)=}")

    return response


def is_good_to_track(response, point):
    HANDCRAFT_THRESH = 240
    if HANDCRAFT_THRESH < response[point[1], point[0]]:
        return True
    return False


def lk_optical_flow(frame1: np.ndarray, frame2: np.ndarray, points: list, block_size: tuple):
    global Ix, Iy
    global Ixx, Iyy, Ixy
    vs = np.array([])

    # Ixx_block_sum = cv2.boxFilter(Ixx, cv2.CV_16S, block_size, normalize=True)
    # Iyy_block_sum = cv2.boxFilter(Iyy, cv2.CV_16S, block_size, normalize=True)
    # Ixy_block_sum = cv2.boxFilter(Ixy, cv2.CV_16S, block_size, normalize=True)
    # frame1 = cv2.GaussianBlur(frame1, block_size, 10)  # denoise
    # frame2 = cv2.GaussianBlur(frame2, block_size, 10)
    It = frame2.astype(np.int16) - frame1.astype(np.int16)

    # Ixt = Ix * It
    # Iyt = Iy * It
    # Ixt_block_sum = cv2.boxFilter(Ixt, cv2.CV_16S, block_size, normalize=True)
    # Iyt_block_sum = cv2.boxFilter(Iyt, cv2.CV_16S, block_size, normalize=True)

    # for p in points:
    #     M = np.array([Ixx_block_sum[p[1], p[0]], Ixy_block_sum[p[1], p[0]],
    #                   Ixy_block_sum[p[1], p[0]], Iyy_block_sum[p[1], p[0]]]).reshape(2, 2)

    #     N = np.array([Ixt_block_sum[p[1], p[0]],
    #                  Iyt_block_sum[p[1], p[0]]]).reshape(2, 1)
     
    #     v = -np.linalg.inv(M) @ N
    #     vs = np.append(vs, v.flatten() + np.array([p[0], p[1]]))

    # half_win = (block_size[0] - 1) // 2
    # eps = 1e-3
    # max_iter = 10
    # for p in points:
    #     x, y = p
    #     Ix_win = Ix[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
    #     Iy_win = Iy[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
    #     It_win = It[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]

    #     A = np.vstack([Ix_win.flatten(), Iy_win.flatten()]).T
    #     b = -It_win.flatten()
        
    #     v = np.zeros(2)
    #     for _ in range(max_iter):
    #         v_new = np.linalg.inv(A.T @ A + eps * np.eye(2)) @ A.T @ b
    #         if np.linalg.norm(v_new - v) < eps:
    #             break
    #         v = v_new
    #     vs = np.append(vs, v_new)


    # vs = vs.reshape(-1, 2)


    # print(type(p0))

    # Implement with opencv
    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None)
    
    new_pts = p1.reshape(-1, 2)
    old_pts = p0.reshape(-1, 2)
    vs = new_pts - old_pts

    return vs


def main():

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <frame1> <frame2>")
        exit(0)

    frame1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    # 640x480, 60x60
    ps = []
    vs = []
    response = compute_harris_response(frame1, (7, 7))
    # print(response.shape)
    frame_show_kp = frame1.copy()
    frame_show_of = frame1.copy()
    for point in generate_grid_point(frame1.shape, (80, 120)):
        if is_good_to_track(response, point):
            frame_show_kp = cv2.circle(
                frame_show_kp, point, radius=1, thickness=2, color=100)
            ps.append(point)
    vs = lk_optical_flow(frame1, frame2, ps, (7, 7))
    # print(vs)

    for p, v in zip(ps, vs):
        cv2.arrowedLine(frame_show_of, p, (int(
            p[0] + v[0]), int(p[1] + v[1])), 100, 2)

    # cv2.calcOpticalFlowPyrLK(frame1, frame2, np.array([[0.0, 0.], [1., 1.]]), None)

    cv2.imshow("keypoints", frame_show_kp)
    cv2.imshow("optical_flow", frame_show_of)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
