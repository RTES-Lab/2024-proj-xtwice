"""
해당 모듈은 complex steerable pyramid (CSP)를 생성하는 것을 담당한다.
CSP는 :meth:`csp.generate_csp` 함수를 통해 생성하며,
나머지 함수들은 CSP 생성에 필요한 구성요소를 계산하는 함수이다.

CSP의 핵심 개념은 Freeman and Adelson의 연구 [1]_ 에 
수록되어 있으며, 이 코드의 CSP 생성 기능은 PBM 알고리즘 원작자인 
Wadhwa et al. [2]_ 의 `MATLAB 코드 <http://people.csail.mit.edu/nwadhwa/phase-video/>`_ 를 참고하여 구현했다.

.. [1] W.T. Freeman and E.H Adelson, "The design and use of steerable filters," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 9, pp. 891-906, 1991.

.. [2] N\. Wadhwa et al., "Phase-based video motion processing," ACM Transactions on Graphics, vol. 32, no. 4, pp. 1–10, 2013.

"""
import cupy as cp

import math

from typing import Tuple, List, Union

M_PI = math.pi


def generate_csp(
    dimension: Union[List[int], Tuple[int, int]],
    n_pyramids: int,
    n_orientations: int,
    twidth: float = 1,
) -> Tuple[List[cp.ndarray], List[Tuple[int, int]]]:
    """
    CSP와 CSP에 해당하는 인덱스를 생성한다. CSP와 인덱스는 추후 :meth:`pbm.PBM` 클래스의 인자로 사용된다.

    Parameters
    ----------
    dimension: Union[List[int], Tuple[int, int]]
        CSP의 크기 (높이 x 너비). 영상의 크기와 동일하게 설정
    n_pyramid: int
        피라미드의 수. 과제 수행간의 기본값은 7.
    n_orientations: int
        피라미드의 방향 수. 과제 수행간의 기본값은 2.
    twidth: float
        CSP의 고깔 크기. 과제 수행간의 기본값은 1.

    Returns
    ----------
    filter, index: Tuple[List[cp.ndarray], List[Tuple[int, int]]]
        filter는 CSP의 내용, index는 CSP의 인덱스를 의미한다. **피라미드의 수는 n_pyramid x n_orientations + 2개** 이다.
        filter는 높이 x 너비 형태의 cp.ndarray, 인덱스는 [(h_min, h_max), (w_min, w_max)] 형태의 튜플이다.

    Examples
    ----------
    (1,024 x 1,024) 크기를 가진 영상을 처리하는 피라미드 수 3, 2방향, twidth 크기 1의 CSP는 다음과 같이 생성한다.
    
    >>> from csp import generate_csp
    >>> pyramid, idx = generate_csp((1024, 1024), 3, 2, 1)
    >>> idx[4] # 5번째 피라미드의 인덱스: [(256, 768), (256, 768)]
    >>> pyramid[4] # 5번째 피라미드의 값을 가진 2차원 인덱스 표출

    완성된 피라미드의 형상은 다음과 같다.

    .. image:: img/csp.png
    
    Warnings
    ----------
    cuPy 라이브러리 설치 후 GPU 환경에서만 동작한다.

    See Also
    ----------
    pbm.PBM

    """
    if not (isinstance(dimension, list) or isinstance(dimension, tuple)):
        raise ValueError("Dimension must be list or tuple")
    if len(dimension) != 2:
        raise ValueError("CSP dimension must be 2D")
    if (
        dimension[0] // (2 ** (n_pyramids - 1)) == 0
        or dimension[1] // (2 ** (n_pyramids - 1)) == 0
    ):
        raise ValueError("Frame size // 2^(n_pyramids - 1) must not be 0")
    for item in dimension:
        if item < 1:
            raise ValueError("Each dimension must has positive value")
    if n_pyramids < 1:
        raise ValueError("#Pyramids must be >= 1")
    if n_orientations < 1:
        raise ValueError("#Orientations must be >= 1")

    r_vals = [0.5 ** (x) for x in range(n_pyramids + 1)]
    filters = generate_csp_filter(dimension, r_vals, n_orientations, twidth)

    return crop_csp(filters, r_vals, n_orientations)


def generate_polar_grid(dimension: Tuple[int, int]) -> Tuple[cp.ndarray, cp.ndarray]:
    xlen = dimension[1]
    ylen = dimension[0]
    center_x = int(xlen / 2)
    center_y = int(ylen / 2)

    # Create rectangular grid
    xramp = cp.array(
        [[(x - int(xlen / 2)) / (xlen / 2)
          for x in range(xlen)] for _ in range(ylen)]
    )
    yramp = cp.array(
        [[(y - int(ylen / 2)) / (ylen / 2)
          for _ in range(xlen)] for y in range(ylen)]
    )
    angle = cp.arctan2(xramp, yramp) + M_PI / 2

    rad = cp.sqrt(xramp**2 + yramp**2)
    rad[center_y][center_x] = rad[center_y - 1][center_x]

    polar_grid = [angle, rad]
    return polar_grid


def calc_radial_mask_pair(
    r: float, rad: cp.ndarray, twidth: float
) -> Tuple[cp.ndarray, cp.ndarray]:
    log_rad = cp.log2(rad) - cp.log2(r)

    himask = log_rad
    himask[himask > 0] = 0
    himask[himask < -twidth] = -twidth
    himask = himask * M_PI / (2 * twidth)

    himask = cp.cos(himask)
    lomask = cp.sqrt(1 - himask**2)

    mask = [himask, lomask]
    return mask


def calc_angle_mask(b: int, orientations: int, angle: cp.ndarray) -> cp.ndarray:
    order = orientations - 1
    const = (
        (2 ** (2 * order))
        * (math.factorial(order) ** 2)
        / (orientations * math.factorial(2 * order))
    )  # Scaling constant

    angle_ = (M_PI + angle - (M_PI * (b - 1) /
              orientations)) % (2 * M_PI) - M_PI
    anglemask = (
        2 * cp.sqrt(const) * (cp.cos(angle_) ** order) *
        (abs(angle_) < (M_PI / 2))
    )  # Make falloff smooth
    return anglemask


def generate_csp_filter(
    dimension: Tuple[int, int], r_vals: List[int], n_orientations: int, twidth=1
) -> List[cp.ndarray]:
    grid = generate_polar_grid(dimension)

    angle = cp.asarray(grid[0])  # 픽셀 위치별 각도 값
    rad = cp.asarray(grid[1])  # 픽셀 위치별 거리 값

    radial_masks = calc_radial_mask_pair(r_vals[0], rad, twidth)
    himask = radial_masks[0]
    lomask_prev = radial_masks[1]

    filters = []
    filters.append(himask)

    for k in range(1, len(r_vals)):
        radial_masks = calc_radial_mask_pair(r_vals[k], rad, twidth)
        himask = radial_masks[0]
        lomask = radial_masks[1]

        rad_mask = himask * lomask_prev

        for j in range(1, n_orientations + 1):
            angle_masks = calc_angle_mask(j, n_orientations, angle)
            filters.append(rad_mask * angle_masks / 2)

        lomask_prev = lomask
    filters.append(lomask)

    for k in range(len(filters)):
        filters[k] = cp.array(filters[k])
    return filters


def crop_csp(
    filters: List[cp.ndarray], r_vals: List[int], n_orientations: int
) -> Tuple[List[cp.ndarray], List[cp.ndarray]]:
    xdim = filters[0].shape[1]
    ydim = filters[0].shape[0]
    n_filters = len(filters)
    filter_indice = [[0 for _ in range(n_orientations)]
                     for _ in range(n_filters)]
    cropped_filters = []

    filter_indice[0][0] = (0, ydim)
    filter_indice[0][1] = (0, xdim)

    cropped_filters.append(filters[0])

    for k in range(1, n_filters - 1, n_orientations):
        n = int(k / n_orientations) + 1
        lb_y = int((ydim * (sum(r_vals[0:n]) - 1)) / 2)
        ub_y = ydim - lb_y
        lb_x = int((xdim * (sum(r_vals[0:n]) - 1)) / 2)
        ub_x = xdim - lb_x

        for i in range(n_orientations):
            filter_indice[k + i][0] = (lb_y, ub_y)
            filter_indice[k + i][1] = (lb_x, ub_x)

        for i in range(n_orientations):
            cropped_filters.append(filters[k + i][lb_y:ub_y, lb_x:ub_x])

    lb_y = int((ydim * (sum(r_vals) - 1)) / 2)
    ub_y = ydim - lb_y
    lb_x = int((xdim * (sum(r_vals) - 1)) / 2)
    ub_x = xdim - lb_x

    filter_indice[n_filters - 1][0] = (lb_y, ub_y)
    filter_indice[n_filters - 1][1] = (lb_x, ub_x)
    cropped_filters.append(filters[n_filters - 1][lb_y:ub_y, lb_x:ub_x])

    result = [cropped_filters, filter_indice]
    return result