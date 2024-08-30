"""
이 모듈은 영상으로부터 region-of-interest (ROI)를 설정하고, 마커가 있는 지점을 추적하는 역할을 수행한다.
현재 R-MDPS의 변위를 측정하고자 하는 지점에는 주변과 확연히 다른 색깔을 가진 마커(스티커)가 붙어 있다고 가정한다.
사용자는 해당 marker의 HSV 색공간 threshold를 수동 설정하여 마커의 위치를 추출한다.
"""
from dataclasses import dataclass
import os
import cupy as cp
import numpy as np
import cv2

@dataclass
class Location:
    """
    ROI의 속성을 다루기 위해 정의된 데이터클래스.
    OpenCV에서는 기본적으로 직사각형 형태의 영역을 x (좌상단 x좌표), y (좌상단 y좌표), h (높이), w (너비) 형태로 관리한다.
    따라서 해당 4가지 속성을 저장하고, 각 값에 대한 기본적인 연산을 할 수 있는 데이터클래스 `Location` 을 사용한다.

    Examples
    ----------
    >>> from track import Location
    >>> loc = Location(x=0, y=0, w=10, h=10) # 직접 지정하기
    >>> loc = Location.from_list([0, 0, 10, 10]) # [x, y, w, h]가 저장된 리스트로부터 불러오기
    >>> loc = Location.from_center_side((2, 4), 6) # 중점 좌표가 (2, 4), 변 길이가 6인 영역 생성하기
    >>> loc*8 # 영역 8배 확대
    """
    x: int=0
    y: int=0
    w: int=0
    h: int=0

    @classmethod
    def from_list(self, a):
        """
        (x, y, w, h)의 리스트로부터 Location 생성
        """
        return self(x=a[0], y=a[1], w=a[2], h=a[3])
    
    @classmethod
    def from_center_side(self, center, side):
        """
        중점 및 길이 정보로부터 Location 생성
        """
        return self(
            x = center[0] - side // 2,
            y = center[1] - side // 2,
            w = side,
            h = side
        )

    def __add__(self, factor):
        return Location(x=self.x + factor, y=self.y + factor, w=self.w + factor, h=self.h + factor)

    def __sub__(self, factor):
        return Location(x=self.x - factor, y=self.y - factor, w=self.w - factor, h=self.h - factor)
    
    def __mul__(self, factor):
        return Location(x=self.x * factor, y=self.y * factor, w=self.w * factor, h=self.h * factor)

class ROIBase:
    """
    ROI 및 지점 추적을 다루는 클래스들의 base 클래스이다.
    현재 코드에서는 영역을 ROI와 track region으로 나눈다.
    ROI는 전체 영상 중 마커를 추적할 영역, track region은 ROI 안에서 각 마커가 움직이는 영역이다 (아래 그림 참조).
    메인 프로그램에서는 R-MDPS영상에서 하나의 ROI와 지점 숫자만큼의 track region을 사용자가 설정하며,
    track region별로 마커의 centroid를 추적해서 변위 측정점의 위치를 추정한다.

    .. image:: img/roi.png

    향후 다른 알고리즘을 가진 지점 추적 알고리즘을 구현할 경우 이 클래스를 상속해야 하며,
    이 글래스의 메소드인 extract_roi, binarize, extract_trackpoint를 구현해야 한다.

    Attributes
    ----------
    roi: Location
        ROI
    track_region: List[Location]
        복수의 track_region이 저장된 리스트

    Methods
    -------
    clear_roi(self)
        현재 roi를 초기화한다.
    clear_track_region(self)
        현재 track_region을 초기화한다.
    binarize(self, frame)
        인자로 받은 `frame` 내의 ROI에서 마커를 추적하고 마커와 마커 외 영역을 분리한 이진 `mask` 를 리턴
    extract_trackpoint(self, mask)
        인자로 받은 `mask` 안에 있는 각 track region의 마커로부터 픽셀수준 변위를 추적하여 출력
    """
    def __init__(self) -> None:
        self.roi = Location()
        self.track_region = []

    def clear_roi(self):
        self.roi = Location()
    
    def clear_track_region(self):
        self.track_region = []
    
    def extract_roi(self, frame):
        return frame[self.roi.y:self.roi.y+self.roi.h,
                     self.roi.x:self.roi.x+self.roi.w, :].copy()
    
    def binarize(self, frame):
        pass

    def extract_trackpoint(self, mask):
        pass

class MarkerTracker(ROIBase):
    """
    HSV 색공간의 threshold로 마커의 mask를 생성하고, mask의 외접원의 중점을 추적하는 알고리즘.
    정확도가 낮기 때문에 사용하지 않고 :meth:`track.MarkerCentroidTracker` 를 사용한다.
    정확도를 높이기 위해 전처리로 입력 이미지를 Gaussian blurring한다.

    Warnings
    ----------
    측정 정확도가 낮고 잡음이 많이 발생하므로 사용하지 말 것

    Attributes
    ----------
    roi: Location
        ROI. ROIBase로부터 상속
    track_region: List[Location]
        복수의 track_region이 저장된 리스트. ROIBase로부터 상속
    lb: Tuple[int, int, int]
        HSV threshold의 lower bound. 기본값 (90, 120, 60).
    ub: Tuple[int, int, int]
        HSV threshold의 upper bound. 기본값 (115, 255, 255)
    gaussian_kernel_size: Tuple[int, int]
        가우시안 커널 크기. 기본값 (9, 9)

    Methods
    ----------
    clear_roi(self)
        현재 roi를 초기화한다.
    clear_track_region(self)
        현재 track_region을 초기화한다.
    binarize(self, frame)
        인자로 받은 `frame` 내의 ROI에서 마커를 추적하고 마커와 마커 외 영역을 분리한 이진 `mask` 를 리턴
    extract_trackpoint(self, mask)
        Track region별로 추출된 마커 영역을 포함하는 가장 작은 외접원의 중점을 계산하여 리턴
    """
    def __init__(self, lb=(0,0,0), ub=(0,0,0), gaussian_kernel_size=(9, 9)) -> None:
        super().__init__()

        if not isinstance(lb, (list, tuple)):
            print("lb argument must be list or tuple, default value is set")
            lb = (0,0,0)
        elif len(lb) != 3:
            print("Length of lb argument must be 3, default value is set")
            lb = (0,0,0)
        if not isinstance(ub, (list, tuple)):
            print("ub argument must be list or tuple, default value is set")
            ub = (0,0,0)
        elif len(ub) != 3:
            print("Length of ub argument must be 3, default value is set")
            ub = (0,0,0)
        
        self.lb = lb
        self.ub = ub
        self.gaussian_kernel_size = gaussian_kernel_size
        self.mask = None

    def binarize(self, frame):
        frame_roi = self.extract_roi(frame)
        frame_roi = cv2.GaussianBlur(frame_roi, self.gaussian_kernel_size, 0)
        frame_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_roi, self.lb, self.ub)

        return mask

    def extract_trackpoint(self, mask):
        center_points = []
        for loc in self.track_region:
            tracked_position = self._extract_trackpoint_impl(mask, loc)
            center_points.append(tracked_position)
        return center_points
    
    def _extract_trackpoint_impl(self, mask, loc):
        region = mask[loc.y:loc.y+loc.h,
                      loc.x:loc.x+loc.w]
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(cnts) > 1:
            marker = sorted(cnts, key=len)[0]
        elif len(cnts) == 1:
            marker = cnts[0]
        else:
            marker = None

        if marker is not None:
            (tracked_x, tracked_y), _ = cv2.minEnclosingCircle(marker)
            tracked_position = Location(x=tracked_x + loc.x, y=tracked_y + loc.y)
        else:
            tracked_position = Location(x=loc.x - (loc.w // 2), y=loc.y - (loc.h // 2))
        
        return tracked_position


class MarkerCentroidTracker(MarkerTracker):
    """
    HSV 색공간의 threshold로 마커의 mask를 생성하고, mask의 무게중심을 추적하는 알고리즘.
    정확도를 높이기 위해 전처리로 입력 이미지를 Gaussian blurring한다.

    Attributes
    ----------
    roi: Location
        ROI. ROIBase로부터 상속
    track_region: List[Location]
        복수의 track_region이 저장된 리스트. ROIBase로부터 상속
    lb: Tuple[int, int, int]
        HSV threshold의 lower bound. 기본값 (90, 120, 60).
    ub: Tuple[int, int, int]
        HSV threshold의 upper bound. 기본값 (115, 255, 255)
    gaussian_kernel_size: Tuple[int, int]
        가우시안 커널 크기. 기본값 (9, 9)

    Methods
    ----------
    clear_roi(self)
        현재 roi를 초기화한다.
    clear_track_region(self)
        현재 track_region을 초기화한다.
    binarize(self, frame)
        인자로 받은 `frame` 내의 ROI에서 마커를 추적하고 마커와 마커 외 영역을 분리한 이진 `mask` 를 리턴
    extract_trackpoint(self, mask)
        Track region별로 추출된 마커 영역의 무게중심을 계산하여 리턴

    Examples
    ----------
    >>> from track import MarkerCentroidTracker, Location
    >>> tracker = MarkerCentroidTracker((90, 120, 60), (115, 255, 255))
    >>> tracker.roi = Location(x=0, y=0, w=1024, h=1024)
    >>> tracker.track_region.append(Location(100, 100, 100, 100)) # 첫 번째 track region
    >>> tracker.track_region.append(Location(400, 400, 100, 100)) # 두 번째 track region
    >>> # 이미지 읽어오는 작업
    >>> mask = tracker.binarize(img)
    >>> trackpoints = tracker.extract_trackpoint(mask)
    >>> for p in trackpoints:
    >>>     # trackpoint p별로 작업 수행
    """
    def __init__(self, lb=(0, 0, 0), ub=(0, 0, 0), gaussian_kernel_size=(9, 9)) -> None:
        super().__init__(lb, ub, gaussian_kernel_size)


    def _extract_trackpoint_impl(self, mask, loc):
        region = mask[loc.y:loc.y+loc.h,
                      loc.x:loc.x+loc.w]
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(cnts) > 1:
            marker = sorted(cnts, key=len)[0]
        elif len(cnts) == 1:
            marker = cnts[0]
        else:
            marker = None

        if marker is not None:
            M = cv2.moments(marker)
            if M["m00"] != 0:
                tracked_x, tracked_y = M["m10"] / M["m00"] , M["m01"] / M["m00"]
            else:
                tracked_x, tracked_y = 0, 0
            tracked_position = Location(x=tracked_x + loc.x, y=tracked_y + loc.y)
        else:
            # tracked_position = Location(x=loc.x - (loc.w // 2), y=loc.y - (loc.h // 2))
            tracked_position = Location(x=-1, y=-1)
        
        return tracked_position