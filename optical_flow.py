
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


def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def compute_harris_response(frame: np.ndarray, block_size: tuple):
    global Ix, Iy
    global Ixx, Iyy, Ixy

    frame = cv2.GaussianBlur(frame, (3, 3), 10)  # denoise
    Ix = cv2.Sobel(frame, cv2.CV_16S, 1, 0)
    Iy = cv2.Sobel(frame, cv2.CV_16S, 0, 1)

    # Ixx = Ix * Ix
    # Iyy = Iy * Iy
    # Ixy = Ix * Iy

    # Ixx_g = cv2.GaussianBlur(Ixx, block_size, 7)
    # Iyy_g = cv2.GaussianBlur(Iyy, block_size, 7)
    # Ixy_g = cv2.GaussianBlur(Ixy, block_size, 7)

    # k = 0.01
    # response = (Ixx_g * Iyy_g - Ixy_g * Ixy_g) - k * \
    #     ((Ixx_g + Iyy_g) * (Ixx_g + Iyy_g))
    # response = (response - np.min(response)) / \
    #     (np.max(response) - np.min(response)) * 255
    # response = response.astype(np.uint8)
    # print(f"{np.mean(response)=}\n{np.std(response)=}\n{np.median(response)=}\n{np.max(response)=}\n{np.min(response)=}")



    # return response
    return np.zeros_like(Ix)

def is_good_to_track(response, point):
    HANDCRAFT_THRESH = 240
    if HANDCRAFT_THRESH < response[point[1], point[0]]:
        return True
    return False


def lk_optical_flow(frame1: np.ndarray, frame2: np.ndarray, points: list, block_size: tuple):
    global Ix, Iy
    vs = np.array([])

    It = frame2.astype(np.int16) - frame1.astype(np.int16)

    Ix_pyr = [Ix]
    Iy_pyr = [Iy]
    It_pyr = [It]
    level = 1

    for i in range(level - 1):
        prev_Ix = Ix_pyr[i]
        prev_Iy = Iy_pyr[i]
        prev_It = It_pyr[i]
        next_Ix = cv2.resize(Ix, (prev_Ix.shape[1]//2, prev_Ix.shape[0]//2))
        next_Ix = cv2.GaussianBlur(next_Ix, (3, 3), 7)
        next_Iy = cv2.resize(Iy, (prev_Iy.shape[1]//2, prev_Iy.shape[0]//2))
        next_Iy = cv2.GaussianBlur(next_Iy, (3, 3), 7)
        next_It = cv2.resize(It, (prev_It.shape[1]//2, prev_It.shape[0]//2))
        next_It = cv2.GaussianBlur(next_It, (3, 3), 7)

        Ix_pyr.append(next_Ix)
        Iy_pyr.append(next_Iy)
        It_pyr.append(next_It)

    # for i, f1, f2 in zip(range(level), frame1_pyr, frame2_pyr):
    #     cv2.imshow(f"{i}_frame1", f1)
    #     cv2.imshow(f"{i}_frame2", f2)

    # cv2.waitKey(0)

    half_win = (block_size[0] - 1) // 2
    for p in points:
        v = np.zeros(2)
        for i, Ix, Iy, It in zip(range(len(Ix_pyr))[::-1], Ix_pyr[::-1], Iy_pyr[::-1], It_pyr[::-1]):

            x, y = p[0] // (2**i), p[1] // (2**i)
            print(f"{i=} {Ix.shape} {x=} {y=}")
            Ix_win = Ix[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
            Iy_win = Iy[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]
            It_win = It[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]

            A = np.vstack((Ix_win.flatten(), Iy_win.flatten())).T # [N x 2]
            b = -It_win.flatten()

            v_new, _, _, _ = np.linalg.lstsq(A.T @ A, A.T @ b)  # [2 x 2] [2 x 1]

            if np.linalg.det(A.T @ A) < 0:
                print(f"det < 0")
                break

            # print(f"{A.T @ A @ v_new=} {A.T @ b=}")
            print(v_new)
            v += v_new * 2
            print(f"{i=} {v=}")
        vs = np.append(vs, v)

    vs = vs.reshape(-1, 2)
    # print(vs)

    ## Implement with opencv
    # p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, maxLevel=3)

    # new_pts = p1.reshape(-1, 2)
    # old_pts = p0.reshape(-1, 2)
    # vs = new_pts - old_pts

    return vs

def main():

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <frame1> <frame2>")
        exit(0)

    frame1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    # frame1 = cv2.resize(frame1, (320, 240))
    # frame2 = cv2.resize(frame2, (320, 240))

    # 640x480, 60x60
    ps = []
    vs = []
    response = compute_harris_response(frame1, (21, 21))
    # print(response.shape)
    frame_show_kp = frame1.copy()
    frame_show_of = frame2.copy()
    for point in generate_grid_point(frame1.shape, (160, 240)):
        if is_good_to_track(response, point):
            frame_show_kp = cv2.circle(
                frame_show_kp, point, radius=1, thickness=2, color=100)
            ps.append(point)
    vs = lk_optical_flow(frame1, frame2, ps, (21, 21))

    for p, v in zip(ps, vs):
        cv2.arrowedLine(frame_show_of, p, (int(
            p[0] + v[0]), int(p[1] + v[1])), 100, 2)

    cv2.imshow("keypoints", frame_show_kp)
    cv2.imshow("optical_flow", frame_show_of)
    cv2.waitKey(0)


    print(gaussian_kernel(5, 1.0))


if __name__ == "__main__":
    main()
