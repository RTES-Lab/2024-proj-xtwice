import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import numpy as np

from csp import *
from data import *
from track import *
from pbm import *

import json

M_PI = math.pi

COLOR_RED = (0, 0, 255)     # red
COLOR_BLUE = (255, 0, 0)     # red
COLOR_GREEN = (0, 255, 0)     # red

down_size_view = 0
ret_frame = 0                # Phase substraction을 수행할 기준 프레임 인덱스 (default: 0)
# NOTE 0 <= retFrame < int(filterOrder / (frameSkipRate + 1)) + 1
skip_pyramid_level = 0    # PBM 수행시 계산하지 않을 이미지 피라미드 수 (default: 0)
# NOTE 0 이상의 값 n이 설정되면 n+1번째 피라미드부터 PBM을 수행함.
attenuate_other_frequency = False      # 영역 외 주파수 배제 여부 (default: 0)
# NOTE 1로 설정할 경우 지정한 영역 외 주파수는 배제함.
Q_mat_length = 50                # Q matrix의 크기 (default: 60)
# NOTE QmatLen이 60이면 측정 지점의 밝기 측정 결과 60개로 모드를 가시화함.

# CSP(Complex Steerable Pyramid) Parameters
# NOTE CSP는 2 + Orientation*NumberofPyramid 개 생성됨.
orientation = 2             # CSP의 방향 수 (default: 2)
n_pyramids = 5         # 사용할 이미지 피라미드 수 (default: 5)
r_values = [0.5**(x) for x in range(n_pyramids+1)]

def dummy_callback(x):
    pass

def get_roi(img):
    roi_index = cv2.selectROI('select_roi', img, False)
    cv2.destroyWindow('select_roi')

    return Location.from_list(roi_index)

def crop_roi(img, loc):
    return img[loc.y:loc.y+loc.h, loc.x:loc.x+loc.w, :]

def get_coordinates(img):
    coordinates = []

    def get_mouse_click(event, x, y, flags, param):
        nonlocal coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', get_mouse_click)

    # 40 -> 100 변경
    cv2.createTrackbar("ROI Size (px)", "Image", 5, 100, dummy_callback)

    while True:
        roi_size = cv2.getTrackbarPos("ROI Size (px)", "Image")
        disp = img.copy()
        for coord in coordinates:
            cv2.circle(
                disp,
                (coord[0], coord[1]),
                3, (0, 153, 0), -1
            )
            cv2.rectangle(
                disp,
                (coord[0] - roi_size // 2, coord[1] - roi_size // 2),
                (coord[0] + roi_size // 2, coord[1] + roi_size // 2),
                (0, 0, 153),
                1
            )
        cv2.imshow("Image", disp)
        key = cv2.waitKey(10) & 0xFF
        if key == 13:  # 엔터 키를 누르면 종료
            break

    cv2.destroyAllWindows()
    ret = [Location.from_center_side(c, roi_size) for c in coordinates]
    return ret, roi_size


def moving_average(a,n):
    N=len(a)
    return np.array([np.mean(a[i:i+n]) for i in np.arange(0,N-n+1)])

# 이미지 ROI, 추적점, 확대반경을 추출하는 함수
def region_from_ui(img):
    height, width = img.shape[0], img.shape[1]
    downsize_ratio = width // 1280

    img_disp = cv2.resize(img, dsize=(0, 0), fx=(1/downsize_ratio), fy=(1/downsize_ratio))

    print("프로세싱할 이미지의 부분을 선택하세요.")
    idx_crop = get_roi(img_disp)
    idx_crop = idx_crop * downsize_ratio

    img_crop = crop_roi(img, idx_crop)

    img_disp = cv2.resize(img_crop, dsize=(0, 0), fx=(1/downsize_ratio), fy=(1/downsize_ratio))

    print("추적을 원하는 하우징 점의 좌표를 마우스 좌클릭으로 설정하고 엔터를 누르세요.")
    coord, roi_size = get_coordinates(img_disp)
    coord = [x*downsize_ratio for x in coord]

    print("왼쪽 랙바가 움직이는 영역을 선택하세요.")
    idx_left_bar = get_roi(img_disp)
    idx_left_bar = idx_left_bar * downsize_ratio
    coord.append(idx_left_bar)

    print("오른쪽 랙바가 움직이는 영역을 선택하세요.")
    idx_right_bar = get_roi(img_disp)
    idx_right_bar = idx_right_bar * downsize_ratio
    coord.append(idx_right_bar)

    print(coord)

    print("확대할 이미지의 부분을 선택하세요.")
    idx_pbm = get_roi(img_disp)
    idx_pbm = idx_pbm * downsize_ratio

    return idx_crop, idx_pbm, coord

# 이미지 ROI, 추적점, 확대반경을 추출하여 json파일에 저장하는 함수
def region_to_json(idx_crop, idx_pbm, coordinate, filename):
    result = {
        "idx_crop": [],
        "idx_pbm": [],
        "coordinate": []
    }
    result["idx_crop"] = [idx_crop.x,
                          idx_crop.y,
                          idx_crop.w,
                          idx_crop.h]
    result["idx_pbm"] = [idx_pbm.x,
                         idx_pbm.y,
                         idx_pbm.w,
                         idx_pbm.h]
    for item in coordinate:
        result["coordinate"].append([
            item.x,
            item.y,
            item.w,
            item.h
        ])
    
    with open(filename, "w") as f:
        json.dump(result, f)
    
    return

def region_from_json(filename):
    with open(filename, "r") as f:
        json_data = json.load(f)

    idx_crop = Location.from_list(json_data["idx_crop"])
    idx_pbm = Location.from_list(json_data["idx_pbm"])

    coord = []

    for item in json_data["coordinate"]:
        coord.append(Location.from_list(item))
    
    return idx_crop, idx_pbm, coord