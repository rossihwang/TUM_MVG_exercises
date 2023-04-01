import numpy as np
import cv2
import sys

def detect_harris_corner(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 20)
    Ix = cv2.Sobel(frame, -1, 1, 0)
    Iy = cv2.Sobel(frame, -1, 0, 1)

    Ixx = np.abs(Ix * Ix)
    Iyy = np.abs(Iy * Iy)
    Ixy = np.abs(Ix * Iy)

    Ixx_g = cv2.GaussianBlur(Ixx, (5, 5), 7)
    Iyy_g = cv2.GaussianBlur(Iyy, (5, 5), 7)
    Ixy_g = cv2.GaussianBlur(Ixy, (5, 5), 7)

    k = 0.04

    response = (Ixx_g * Iyy_g - Ixy * Ixy) - k * ((Ixx_g + Iyy_g) * (Ixx_g + Iyy_g))

    print(response.shape)
    response = response.astype(float) / np.max(response) * 255
    response = response.astype(np.uint8)
    print(response)

    # Ixx_g = Ixx_g.astype(float) / np.max(Ixx_g)
    # Iyy_g = Iyy_g.astype(float) / np.max(Iyy_g)
    # Ixy_g = Ixy_g.astype(float) / np.max(Ixy_g)
    
    # Ixx_g = Ixx_g.astype(np.uint8)
    # Iyy_g = Iyy_g.astype(np.uint8)
    # Ixy_g = Ixy_g.astype(np.uint8)

    cv2.imshow("Ixx_g", Ixx_g)
    cv2.imshow("Iyy_g", Iyy_g)
    cv2.imshow("Ixy_g", Ixy_g)
    threshold = 252
    response[threshold < response] = 255
    response[response <= threshold] = 0
    # cv2.imshow("response", response)

    cv2.waitKey(0)


def lk_optical_flow(frame, n):
    pass


def coars_to_fine_optical_flow():
    pass

def main():
    
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    detect_harris_corner(frame)


if __name__ == "__main__":
    main()