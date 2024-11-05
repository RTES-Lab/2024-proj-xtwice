"""
이 모듈은 영상을 확대하는 기능을 수행한다.
확대 알고리즘으로 phase-based magnification (PBM)을 사용하였으며,
실제 구현은 An and Lee 의 연구 [1]_ 를 참고하였다.

.. [1] J\. An and S\. Lee, "hase-Based Motion Magnification for Structural Vibration Monitoring at a Video Streaming Rate," IEEE Access, vol\. 10, pp\. 123423-123435, 2022.

.. warning::
    PBM 내 연산은 GPU에서 수행된다.
    이 파일 내 함수들은 GPU에서 동작되도록 구현되었으므로
    np.ndarray 타입 데이터를 인자로 사용하면 오동작한다.
"""
from cupy.fft import fft2, fftshift, ifft2, ifftshift

import numpy as np
import cupy as cp

from scipy.signal import firwin
import math

from typing import Tuple, List, Union

M_PI = math.pi


class PBM:
    """
    PBM을 수행하는 클래스. PBM을 사용하려면 :meth:`csp.generate_csp` 로 생성된 CSP 및 그 인덱스 정보와
    :meth:`pbm.generate_fir_bandpass` 로 생성된 bandpass filter 정보를 인자로 입력해야 한다.

    Attributes
    ----------
    fir_filter: cp.ndarray
        FIR 필터 배열. :meth:`pbm.generate_fir_bandpass` 함수로부터 생성
    pyr: List[cp.ndarray]
        CSP의 리스트. :meth:`csp.generate_csp` 함수로부터 생성
    pyr_idx: List[Tuple[Tuple[int, int], Tuple[int, int]]]
        CSP의 인덱스. :meth:`csp.generate_csp` 함수로부터 생성
    exclude_outside_freq: bool
        Bandpass 밖의 성분을 완전히 제거할지 결정하는 플래그. 기본값은 False
    skip_pyramid_level: int
        몇 개의 CSP 피라미드를 무시할지 결정하는 플래그. 기본값은 0
    alpha: float
        확대계수. 0인 경우 확대하지 않음.
    loop_range: int
        필터의 크기. fir_filter 멤버의 크기로부터 자동 계산됨.
    n_levels: int
        피라미드의 갯수. pyr 멤버의 크기로부터 자동 계산됨.
    xdim: int
        확대할 영상의 너비. pyr 멤버의 크기로부터 자동 계산되나, 입력 이미지와 크기가 다를 경우 오류가 발생하니 주의
    ydim: int
        확대할 영상의 높이. pyr 멤버의 크기로부터 자동 계산되나, 입력 이미지와 크기가 다를 경우 오류가 발생하니 주의
    vid_fft: List[cp.ndarray]
        PBM 계산을 위한 내부 버퍼
    phase_xqueue: List[cp.ndarray]
        PBM 계산을 위한 내부 버퍼

    Examples
    ----------

    >>> from data import *
    >>> from csp import generate_csp
    >>> from pbm import *
    >>> stream = VideoFileReader(\"./input/1.0_new_dark.mp4\", 24, False, 0)
    >>> # 24 FPS의 영상 \"./input/1.0_new_dark.mp4\"을 배속 다운샘플링 없이 읽어오는 스트림
    >>> img = stream.read() # 첫 한 프레임 읽기
    >>> h, w = img.shape[0], img.shape[1]
    >>> pyramid, pyr_idx = generate_csp((h, w), 5, 2)
    >>> # 이미지 크기를 가지는 피라미드 수 5, 방향 2 의 CSP 생성
    >>> fir_filter = generate_fir_bandpass(30, 24, (0.01, 1))
    >>> # 필터 크기(loop_range) 31, FPS 24에서 [0.01, 1] Hz 성분만 대역 통과하는 필터 생성
    >>> pbm = PBM(fir_filter, pyramid, pyr_idx, 5.0, False, 0)
    >>> # 확대계수 5로 영상을 확대하는 PBM 객체 생성
    >>> for _ in range(pbm.loop_range): # 초기화 작업 수행
    >>>     img = stream.read()
    >>>     x = stream.transform(img)
    >>>     pbm.run(x[:,:,0], True)
    >>> while condition: # 특정 조건을 만족할 때까지
    >>>     img = stream.read()
    >>>     x = stream.transform()
    >>>     x[:,:,0] = pbm.run(x[:,:,0])
    >>>     x = np.asarray(x.get())
    >>>     x = array2img(yiq2bgr(x))
    >>>     # 확대된 프레임 x로 할 작업 수행
    >>>     # ...

    """
    def __init__(
        self,
        fir_filter: cp.ndarray,
        pyr: List[cp.ndarray],
        pyr_idx: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        alpha: float,
        exclude_outside_freq: bool = False,
        skip_pyramid_level: int = 0,
    ):
        if not isinstance(fir_filter, cp.ndarray):
            raise ValueError("FIR filter must be passed by cuPy array")
        if not isinstance(pyr, list):
            raise ValueError("CSP must be passed by list type")
        for item in pyr:
            if not isinstance(item, cp.ndarray):
                raise ValueError("CSP array must be passed by cuPy array")
            if item.ndim != 2:
                raise ValueError("Dimension of CSP array must be 2")

        self.fir_filter = fir_filter
        self.pyr = pyr
        self.pyr_idx = pyr_idx
        self.exclue_outside_freq = bool(exclude_outside_freq)
        self.skip_pyramid_level = int(skip_pyramid_level)
        self.alpha = float(alpha)

        self.loop_range = self.fir_filter.size
        self.n_levels = len(self.pyr)
        self.ydim = self.pyr[0].shape[0]
        self.xdim = self.pyr[0].shape[1]

        if self.skip_pyramid_level >= self.n_levels:
            raise ValueError(
                "skip_pyramid_level must be less than number of CSPs")
        if self.alpha < 1:
            self.alpha = 1

        self.vid_fft = cp.zeros(
            (self.loop_range, self.ydim, self.xdim), dtype=cp.complex64
        )
        self.phase_xqueue = []

        for i in range(0, self.n_levels):
            target = cp.zeros(
                (self.loop_range, self.pyr[i].shape[0], self.pyr[i].shape[1]),
            )

            self.phase_xqueue.append(target)

    def run(self, x: cp.ndarray, init: bool = False):
        """
        입력된 프레임을 확대하는 메소드

        Parameters
        ----------
        x: cp.ndarray
            입력되는 프레임. YIQ 색공간의 Y에 해당하는 2차원 배열을 받는다.
        init: bool
            초기화에 사용되는 프레임인지를 결정하는 플래그.
            이 PBM 구현은 영상확대 이전 첫 loop_range와 같은 수의 프레임을 init=True 인 상태로 읽어와야 한다.

        Returns
        ----------
        magnified: cp.ndarray
            확대된 프레임
        
        """
        if self.pyr[0].shape != x.shape:
            raise ValueError("CSP and input array shapes are not matched")
        if init:
            return self._run_impl_init(x)
        else:
            return self._run_impl(x)

    def _run_impl_init(self, x):
        self.vid_fft[:-1, :, :] = self.vid_fft[1:, :, :]
        self.vid_fft[-1, :, :] = calc_fft2d(x)

        for level in range(1, self.n_levels - 1):
            lb_y, ub_y = self.pyr_idx[level][0]
            lb_x, ub_x = self.pyr_idx[level][1]

            self.phase_xqueue[level][:-1, :,
                                     :] = self.phase_xqueue[level][1:, :, :]
            self.phase_xqueue[level][-1, :, :] = calc_phase_angle(
                self.pyr[level] * self.vid_fft[-1, lb_y:ub_y, lb_x:ub_x]
            )

        return None

    def _run_impl(self, x):
        self.vid_fft[:-1, :, :] = self.vid_fft[1:, :, :]
        self.vid_fft[-1, :, :] = calc_fft2d(x)
        magnified_freq = cp.zeros((self.ydim, self.xdim), dtype=cp.complex64)

        for level in range(1 + self.skip_pyramid_level, self.n_levels - 1):
            lb_y, ub_y = self.pyr_idx[level][0]
            lb_x, ub_x = self.pyr_idx[level][1]

            ref_frame = calc_ifft2d(
                self.pyr[level] * self.vid_fft[0, lb_y:ub_y, lb_x:ub_x]
            )
            orig_frame = calc_ifft2d(
                self.pyr[level] * self.vid_fft[-1, lb_y:ub_y, lb_x:ub_x]
            )

            ref_frame_orig = ref_frame / cp.abs(ref_frame)
            ref_frame = cp.angle(ref_frame)

            self.phase_xqueue[level][:-1, :,
                                     :] = self.phase_xqueue[level][1:, :, :]
            self.phase_xqueue[level][-1, :, :] = calc_phase_angle(
                self.pyr[level] * self.vid_fft[-1, lb_y:ub_y, lb_x:ub_x]
            )
            delta_xqueue = calc_phase_difference(
                self.phase_xqueue[level], ref_frame)

            phase = (
                cp.sum(
                    delta_xqueue *
                    self.fir_filter.reshape(self.loop_range, 1, 1),
                    axis=0,
                )
                * self.alpha
            )

            if self.exclue_outside_freq:
                orig_frame = cp.abs(orig_frame) * ref_frame_orig

            output = cp.exp(1j * phase) * orig_frame
            magnified_freq[lb_y:ub_y, lb_x:ub_x] += (
                2 * self.pyr[level] * calc_fft2d(output)
            )

        level = self.n_levels - 1
        lb_y, ub_y = self.pyr_idx[level][0]
        lb_x, ub_x = self.pyr_idx[level][1]
        lowpass_frame = self.vid_fft[-1, lb_y:ub_y,
                                     lb_x:ub_x] * (self.pyr[level] ** 2)
        magnified_freq[lb_y:ub_y, lb_x:ub_x] += lowpass_frame
        magnified = cp.real(calc_ifft2d(magnified_freq))

        return magnified


def calc_fft2d(x: cp.ndarray) -> cp.ndarray:
    return fftshift(fft2(x))


def calc_ifft2d(x: cp.ndarray) -> cp.ndarray:
    return ifft2(ifftshift(x))


def calc_phase_angle(x: cp.ndarray) -> cp.ndarray:
    return cp.angle(calc_ifft2d(x))


def calc_phase_difference(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    return (M_PI + a - b) % (2 * M_PI) - M_PI


# Filtering


def generate_fir_bandpass(
    filter_size: int,
    sample_rate: int,
    freq_band: Union[List[float], Tuple[float, float]],
) -> cp.ndarray:
    if filter_size < 1:
        raise ValueError(f"Filter size {filter_size} must be >=1")
    if sample_rate < 1:
        raise ValueError(f"Sample rate (video FPS) {sample_rate} must be >=1")

    fir_filter = firwin(
        filter_size,
        (2 * freq_band[0] / sample_rate, 2 * freq_band[1] / sample_rate),
        pass_zero="bandpass",
    )

    return cp.asarray(fir_filter)
