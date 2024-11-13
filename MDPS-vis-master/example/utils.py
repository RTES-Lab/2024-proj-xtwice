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
import yaml
from box import Box

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

    # 40 -> 100 -> 120 -> 140 변경
    cv2.createTrackbar("ROI Size (px)", "Image", 5, 300, dummy_callback)

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


def find_center(filename, video_sampling_rate, frame_skip_rate, roi_file, lb, ub):
    stream = VideoFileReader(filename, video_sampling_rate, False, frame_skip_rate)
    img = stream.read()
    idx_crop, idx_pbm, coord = region_from_json(roi_file)

    for _ in range(1):
        img = stream.read()
        img_crop = crop_roi(img, idx_crop)
        img_pbm = crop_roi(img_crop, idx_pbm)
        x = stream.transform(img_pbm)[:,:,0]

    tracker = MarkerCentroidTracker(lb, ub)
    tracker.roi = Location(x=0, y=0, w=img_crop.shape[1], h=img_crop.shape[0])
    tracker.track_region = coord
    qx = []
    qy = []

    cnt_frame = 0
    
    for _ in range(9999):
        cnt_frame = cnt_frame+1
        img = stream.read()
        if img is None:
            break
        img = crop_roi(img, idx_crop)

        mask = tracker.binarize(img)
        trackpoints = tracker.extract_trackpoint(mask)
        qy.append([item.y for item in trackpoints])
        qx.append([item.x for item in trackpoints])
    # qx와 qy를 NumPy 배열로 변환합니다.
    np_qx = np.array(qx)
    np_qy = np.array(qy)

    # 배열의 모든 요소에 대해 평균을 계산합니다.
    mean_qx = np.mean(np_qx, axis=0)
    mean_qy = np.mean(np_qy, axis=0)
    print(mean_qx, mean_qy)
    return mean_qx, mean_qy

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

    # print("왼쪽 랙바가 움직이는 영역을 선택하세요.")
    # idx_left_bar = get_roi(img_disp)
    # idx_left_bar = idx_left_bar * downsize_ratio
    # coord.append(idx_left_bar)

    # print("오른쪽 랙바가 움직이는 영역을 선택하세요.")
    # idx_right_bar = get_roi(img_disp)
    # idx_right_bar = idx_right_bar * downsize_ratio
    # coord.append(idx_right_bar)

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

def load_yaml(config_yaml_file: str):
    """
    YAML 파일을 읽어와 Box 객체로 변환하는 함수.

    Parameters
    ----------
    config_yaml_file : str
        읽을 YAML 파일의 경로.

    Returns
    ----------
    config : Box
        YAML 파일의 내용을 포함한 Box 객체
    """
    with open(config_yaml_file) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

## 예비
def moving_average(a,n):
    N=len(a)
    return np.array([np.mean(a[i:i+n]) for i in np.arange(0,N-n+1)])