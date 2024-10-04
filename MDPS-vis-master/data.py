"""
이 모듈은 데이터 입출력을 담당한다.
현재는 영상 프레임이 지정된 동영상 파일로부터 들어온다고 가정한다.

.. note::
    추후 PC에 연결한 카메라로부터 프레임을 직접 받아 처리하기 위해서는 별도로 :meth:`data.DataStream` 을
    상속받는 클래스를 추가로 구현해야 한다.

.. warning::
    PBM 내 연산을 제외한 다른 부분은 CPU에서 수행된다.
    data 모듈의 소스코드도 CPU에서 동작하도록 구현되었으므로 
    인자로 cp.ndarray 타입을 사용하면 오동작한다.
"""
from dataclasses import dataclass
import os
import cupy as cp
import numpy as np
import cv2


class DataStream:
    """
    영상 데이터를 다루는 클래스들의 base 클래스이다.
    향후 추가되는 데이터 입출력 모듈은 이 클래스를 상속해야 하며,
    이 클래스의 메소드인 read, transform, forward를 구현해야 한다.

    """
    def __init__(self):
        self.cap = None
        self.dim = None
        return

    def read(self):
        """
        스트림으로부터 프레임을 읽어오는 메소드. 파생 클래스에서 재정의하여 사용한다.

        Returns
        ----------
        np.ndarray
            OpenCV BGR 이미지 형식의 이미지 배열
        """
        return None

    def transform(self, x):
        """
        읽어온 프레임을 PBM에서 사용할 수 있는 형태로 변환하는 메소드.
        uint8형 BGR 색공간 이미지를 [0, 1] 범위의 실수형 YIQ 색공간 이미지로 변환한다.

        Parameters
        ----------
        x: np.ndarray
            OpenCV BGR 이미지 형식의 이미지 배열

        Returns
        ----------
        np.ndarray
            실수형 YIQ 색공간 배열
        """
        x = img2array(x)
        if self.downsample:
            x = cv2.pyrDown(x)
        x = bgr2yiq(x)
        x = cp.asarray(x)

        return x

    def forward(self):
        """
        read와 transform을 순차적으로 수행하여 리턴하는 편의성을 위한 함수이다.

        Returns
        ----------
        np.ndarray
            stream에서 읽어온 실수형 YIQ 색공간 배열
        """
        x = self.read()
        x = self.transform(x)
        return x


class VideoFileStream(DataStream):
    """
    비디오 파일을 반복적으로 읽어오는 클래스

    Attributes
    ----------
    filename: str
        영상 파일 경로
    downsample: bool
        영상을 1/2로 스케일링할지 여부
    frame_skip_rate: int
        영상 배속 옵션 (0인 경우 배속하지 않음, 1인 경우 2배속, 2인 경우 3배속, ...)
    fps: int
        영상의 sampling rate(초당 프레임 수)
    cap: cv2.VideoCapture
        영상을 불러오는 OpenCV 라이브러리 객체
    dim: List[int]
        영상의 크기 (H x W x C)
    
    Methods
    ----------
    read(self)
        See :meth:`data.DataStream`
    transform(self, x)
        See :meth:`data.DataStream`
    forward(self)
        See :meth:`data.DataStream`

    Examples
    ----------
    >>> from data import VideoFileStream
    >>> stream = VideoFileStream(filename=\"./input/mov.mp4\", fps=24, downsample=False, frame_skip_rate=0)
    >>> # 24FPS의 ./input/mov.mp4 파일을 다운샘플링과 배속 없이 가져오는 스트림 객체를 만든다.
    >>> img = stream.read()
    >>> if img is None: # 예외처리 ...

    """
    def __init__(
        self,
        filename: str,
        fps: int = 30,
        downsample: bool = False,
        frame_skip_rate: int = 0,
    ) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file or directory {filename}")
        if frame_skip_rate < 0:
            raise ValueError(f"Frame skip rate {frame_skip_rate} must be >=0")
        if fps < 1:
            raise ValueError(f"FPS f{fps} must be >=1")

        self.filename = str(filename)
        self.downsample = bool(downsample)
        self.frame_skip_rate = int(frame_skip_rate)
        self.fps = int(fps)

        self.cap = cv2.VideoCapture()

        self.cap.open(filename)

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("Failed to read video stream")

        self.dim = frame.shape

        if self.downsample:
            if self.dim[0] // 2 == 0 or self.dim[1] // 2 == 0:
                raise RuntimeError("Cannot downsample images")
            frame = cv2.pyrDown(frame)
            self.dim = frame.shape

    def __del__(self):
        self.cap.release()

    def read(self):
        frame = self._read_impl()

        for _ in range(self.frame_skip_rate):
            _ = self._grab_impl()

        return frame

    def _read_impl(self):
        ret, frame = self.cap.read()

        if not ret:
            self.cap.open(self.filename)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read video stream")

        return frame

    def _grab_impl(self):
        ret = self.cap.grab()

        if not ret:
            self.cap.open(self.filename)
            ret = self.cap.grab()
            if not ret:
                raise RuntimeError("Failed to read video stream")

        return True


class VideoFileReader(VideoFileStream):
    """
    :meth:`VideoFileStream` 과 동일한 기능을 수행하지만,
    이 클래스는 영상을 끝까지 읽었을 때 더이상 읽지 않고 `None` 을 리턴한다.

    Attributes
    ----------
    filename: str
        영상 파일 경로
    downsample: bool
        영상을 1/2로 스케일링할지 여부
    frame_skip_rate: int
        영상 배속 옵션 (0인 경우 배속하지 않음, 1인 경우 2배속, 2인 경우 3배속, ...)
    fps: int
        영상의 sampling rate(초당 프레임 수)
    cap: cv2.VideoCapture
        영상을 불러오는 OpenCV 라이브러리 객체
    dim: List[int]
        영상의 크기 (H x W x C)
    
    Methods
    ----------
    read(self)
        See :meth:`data.DataStream`
    transform(self, x)
        See :meth:`data.DataStream`
    forward(self)
        See :meth:`data.DataStream`

    Examples
    ----------
    >>> from data import VideoFileReader
    >>> stream = VideoFileReader(filename=\"./input/mov.mp4\", fps=24, downsample=False, frame_skip_rate=0)
    >>> # VideoFileStream과 동일한 방법으로 사용한다.
    >>> # 24FPS의 ./input/mov.mp4 파일을 다운샘플링과 배속 없이 가져오는 스트림 객체를 만든다.
    >>> img = stream.read()
    >>> if img is None: # 예외처리 ...

    See Also
    ----------
    data.VideoFileStream

    """
    def __init__(
        self,
        filename: str,
        fps: int = 30,
        downsample: bool = False,
        frame_skip_rate: int = 0,
    ) -> None:
        super().__init__(filename, fps, downsample, frame_skip_rate)

    def read(self):
        frame = self._read_impl()

        for _ in range(self.frame_skip_rate):
            if self._read_impl() is None:
                return None

        return frame

    def _read_impl(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame


# skimage 버전
# map_bgr_yiq = np.array([[ 0.114, -0.32134392,  0.31119955],
#                         [ 0.587, -0.27455667, -0.52273617],
#                         [ 0.299,  0.59590059,  0.21153661]])

# map_yiq_bgr = np.array([[ 1.0,  1.0,  1.0],
#                         [-1.10674021, -0.27201283,  0.95598634],
#                         [ 1.70423049, -0.64720424,  0.6208248 ]])

# J. An 버전
map_bgr_yiq = np.array(
    [[0.114, -0.322, 0.312], [0.587, -0.274, -0.523], [0.299, 0.596, 0.211]]
)
map_yiq_bgr = np.array(
    [[1.0, 1.0, 1.0], [-1.106, -0.272, 0.956], [1.703, -0.647, 0.621]]
)


def bgr2yiq(img):
    """
    입력 데이터의 색공간을 BGR에서 YIQ로 변환한다.
    
    Parameters
    ----------
    img: np.ndarray
        입력 이미지
    
    Returns
    ----------
    output: np.ndarray
        YIQ 색공간 이미지
    """
    return img @ map_bgr_yiq


def yiq2bgr(img):
    """
    입력 데이터의 색공간을 YIQ에서 BGR로 변환한다.
    
    Parameters
    ----------
    img: np.ndarray
        입력 이미지
    
    Returns
    ----------
    output: np.ndarray
        BGR 색공간 이미지
    """
    return img @ map_yiq_bgr


def img2array(x):
    """
    [0, 255] 범위의 `uint8` 형 이미지 배열을 [0.0, 1.0] 범위의 `double` 형 이미지 배열로 스케일링한다.
    
    Parameters
    ----------
    x: np.ndarray
        입력 이미지
    
    Returns
    ----------
    y: np.ndarray
        64비트 실수형 이미지
    """
    return x.astype(np.float64) / 255.0


def array2img(x):
    """
    [0.0, 1.0] 범위의 `double` 형 이미지 배열을 [0, 255] 범위의 `uint8` 형 이미지 배열로 스케일링한다.
    
    Parameters
    ----------
    x: np.ndarray
        입력 이미지
    
    Returns
    ----------
    y: np.ndarray
        8비트 부호 없는 정수형 이미지
    """
    return np.clip((x * 255.0), 0, 255).astype(np.uint8)
"""
이 모듈은 데이터 입출력을 담당한다.
현재는 영상 프레임이 지정된 동영상 파일로부터 들어온다고 가정한다.

.. note::
    추후 PC에 연결한 카메라로부터 프레임을 직접 받아 처리하기 위해서는 별도로 :meth:`data.DataStream` 을
    상속받는 클래스를 추가로 구현해야 한다.

.. warning::
    PBM 내 연산을 제외한 다른 부분은 CPU에서 수행된다.
    data 모듈의 소스코드도 CPU에서 동작하도록 구현되었으므로 
    인자로 cp.ndarray 타입을 사용하면 오동작한다.
"""
from dataclasses import dataclass
import os
import cupy as cp
import numpy as np
import cv2


class DataStream:
    """
    영상 데이터를 다루는 클래스들의 base 클래스이다.
    향후 추가되는 데이터 입출력 모듈은 이 클래스를 상속해야 하며,
    이 클래스의 메소드인 read, transform, forward를 구현해야 한다.

    """
    def __init__(self):
        self.cap = None
        self.dim = None
        return

    def read(self):
        """
        스트림으로부터 프레임을 읽어오는 메소드. 파생 클래스에서 재정의하여 사용한다.

        Returns
        ----------
        np.ndarray
            OpenCV BGR 이미지 형식의 이미지 배열
        """
        return None

    def transform(self, x):
        """
        읽어온 프레임을 PBM에서 사용할 수 있는 형태로 변환하는 메소드.
        uint8형 BGR 색공간 이미지를 [0, 1] 범위의 실수형 YIQ 색공간 이미지로 변환한다.

        Parameters
        ----------
        x: np.ndarray
            OpenCV BGR 이미지 형식의 이미지 배열

        Returns
        ----------
        np.ndarray
            실수형 YIQ 색공간 배열
        """
        x = img2array(x)
        if self.downsample:
            x = cv2.pyrDown(x)
        x = bgr2yiq(x)
        x = cp.asarray(x)

        return x

    def forward(self):
        """
        read와 transform을 순차적으로 수행하여 리턴하는 편의성을 위한 함수이다.

        Returns
        ----------
        np.ndarray
            stream에서 읽어온 실수형 YIQ 색공간 배열
        """
        x = self.read()
        x = self.transform(x)
        return x


class VideoFileStream(DataStream):
    """
    비디오 파일을 반복적으로 읽어오는 클래스

    Attributes
    ----------
    filename: str
        영상 파일 경로
    downsample: bool
        영상을 1/2로 스케일링할지 여부
    frame_skip_rate: int
        영상 배속 옵션 (0인 경우 배속하지 않음, 1인 경우 2배속, 2인 경우 3배속, ...)
    fps: int
        영상의 sampling rate(초당 프레임 수)
    cap: cv2.VideoCapture
        영상을 불러오는 OpenCV 라이브러리 객체
    dim: List[int]
        영상의 크기 (H x W x C)
    
    Methods
    ----------
    read(self)
        See :meth:`data.DataStream`
    transform(self, x)
        See :meth:`data.DataStream`
    forward(self)
        See :meth:`data.DataStream`

    Examples
    ----------
    >>> from data import VideoFileStream
    >>> stream = VideoFileStream(filename=\"./input/mov.mp4\", fps=24, downsample=False, frame_skip_rate=0)
    >>> # 24FPS의 ./input/mov.mp4 파일을 다운샘플링과 배속 없이 가져오는 스트림 객체를 만든다.
    >>> img = stream.read()
    >>> if img is None: # 예외처리 ...

    """
    def __init__(
        self,
        filename: str,
        fps: int = 30,
        downsample: bool = False,
        frame_skip_rate: int = 0,
    ) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file or directory {filename}")
        if frame_skip_rate < 0:
            raise ValueError(f"Frame skip rate {frame_skip_rate} must be >=0")
        if fps < 1:
            raise ValueError(f"FPS f{fps} must be >=1")

        self.filename = str(filename)
        self.downsample = bool(downsample)
        self.frame_skip_rate = int(frame_skip_rate)
        self.fps = int(fps)

        self.cap = cv2.VideoCapture()

        self.cap.open(filename)

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("Failed to read video stream")

        self.dim = frame.shape

        if self.downsample:
            if self.dim[0] // 2 == 0 or self.dim[1] // 2 == 0:
                raise RuntimeError("Cannot downsample images")
            frame = cv2.pyrDown(frame)
            self.dim = frame.shape

    def __del__(self):
        self.cap.release()

    def read(self):
        frame = self._read_impl()

        for _ in range(self.frame_skip_rate):
            _ = self._grab_impl()

        return frame

    def _read_impl(self):
        ret, frame = self.cap.read()

        if not ret:
            self.cap.open(self.filename)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read video stream")

        return frame

    def _grab_impl(self):
        ret = self.cap.grab()

        if not ret:
            self.cap.open(self.filename)
            ret = self.cap.grab()
            if not ret:
                raise RuntimeError("Failed to read video stream")

        return True


class VideoFileReader(VideoFileStream):
    """
    :meth:`VideoFileStream` 과 동일한 기능을 수행하지만,
    이 클래스는 영상을 끝까지 읽었을 때 더이상 읽지 않고 `None` 을 리턴한다.

    Attributes
    ----------
    filename: str
        영상 파일 경로
    downsample: bool
        영상을 1/2로 스케일링할지 여부
    frame_skip_rate: int
        영상 배속 옵션 (0인 경우 배속하지 않음, 1인 경우 2배속, 2인 경우 3배속, ...)
    fps: int
        영상의 sampling rate(초당 프레임 수)
    cap: cv2.VideoCapture
        영상을 불러오는 OpenCV 라이브러리 객체
    dim: List[int]
        영상의 크기 (H x W x C)
    
    Methods
    ----------
    read(self)
        See :meth:`data.DataStream`
    transform(self, x)
        See :meth:`data.DataStream`
    forward(self)
        See :meth:`data.DataStream`

    Examples
    ----------
    >>> from data import VideoFileReader
    >>> stream = VideoFileReader(filename=\"./input/mov.mp4\", fps=24, downsample=False, frame_skip_rate=0)
    >>> # VideoFileStream과 동일한 방법으로 사용한다.
    >>> # 24FPS의 ./input/mov.mp4 파일을 다운샘플링과 배속 없이 가져오는 스트림 객체를 만든다.
    >>> img = stream.read()
    >>> if img is None: # 예외처리 ...

    See Also
    ----------
    data.VideoFileStream

    """
    def __init__(
        self,
        filename: str,
        fps: int = 30,
        downsample: bool = False,
        frame_skip_rate: int = 0,
    ) -> None:
        super().__init__(filename, fps, downsample, frame_skip_rate)

    def read(self):
        frame = self._read_impl()

        for _ in range(self.frame_skip_rate):
            if self._read_impl() is None:
                return None

        return frame

    def _read_impl(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame


# skimage 버전
# map_bgr_yiq = np.array([[ 0.114, -0.32134392,  0.31119955],
#                         [ 0.587, -0.27455667, -0.52273617],
#                         [ 0.299,  0.59590059,  0.21153661]])

# map_yiq_bgr = np.array([[ 1.0,  1.0,  1.0],
#                         [-1.10674021, -0.27201283,  0.95598634],
#                         [ 1.70423049, -0.64720424,  0.6208248 ]])

# J. An 버전
map_bgr_yiq = np.array(
    [[0.114, -0.322, 0.312], [0.587, -0.274, -0.523], [0.299, 0.596, 0.211]]
)
map_yiq_bgr = np.array(
    [[1.0, 1.0, 1.0], [-1.106, -0.272, 0.956], [1.703, -0.647, 0.621]]
)


def bgr2yiq(img):
    """
    입력 데이터의 색공간을 BGR에서 YIQ로 변환한다.
    
    Parameters
    ----------
    img: np.ndarray
        입력 이미지
    
    Returns
    ----------
    output: np.ndarray
        YIQ 색공간 이미지
    """
    return img @ map_bgr_yiq


def yiq2bgr(img):
    """
    입력 데이터의 색공간을 YIQ에서 BGR로 변환한다.
    
    Parameters
    ----------
    img: np.ndarray
        입력 이미지
    
    Returns
    ----------
    output: np.ndarray
        BGR 색공간 이미지
    """
    return img @ map_yiq_bgr


def img2array(x):
    """
    [0, 255] 범위의 `uint8` 형 이미지 배열을 [0.0, 1.0] 범위의 `double` 형 이미지 배열로 스케일링한다.
    
    Parameters
    ----------
    x: np.ndarray
        입력 이미지
    
    Returns
    ----------
    y: np.ndarray
        64비트 실수형 이미지
    """
    return x.astype(np.float64) / 255.0


def array2img(x):
    """
    [0.0, 1.0] 범위의 `double` 형 이미지 배열을 [0, 255] 범위의 `uint8` 형 이미지 배열로 스케일링한다.
    
    Parameters
    ----------
    x: np.ndarray
        입력 이미지
    
    Returns
    ----------
    y: np.ndarray
        8비트 부호 없는 정수형 이미지
    """
    return np.clip((x * 255.0), 0, 255).astype(np.uint8)
