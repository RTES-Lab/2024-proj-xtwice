"""
스티커의 HSV threshold를 설정하는 데 도움을 주는 프로그램
해당 프로그램으로 얻은 threshold를 설정해 스티커의 변위를 추출한다
"""

import cv2
import numpy as np

def nothing(x):
    pass

def get_first_frame(video_path: str):
    """
    영상의 첫번째 프레임 이미지를 가져오는 함수

    Parameters
    ----------
    video_path: str
        사용할 비디오 파일 경로

    Returns
    ----------
    cap: numpy.ndarray
        (height, width, channels) 형태의 배열

    Examples
    ----------
    # 여기서 video_path 경로 설정해주기!
    >>> video_path = "./input/pbm_0724/0724_30204_H_1200.mov"
    >>> first_frame = get_first_frame(video_path)
    """

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()

    # 동영상 파일 해제
    cap.release()

    if not ret:
        print("Error: Could not read the first frame.")
        return None
    
    return frame

def main():
    video_path = "./input/0927/0828_30204_H_T/0828_30204_H_T.mov"  # 동영상 파일 경로 입력
    first_frame = get_first_frame(video_path)
    
    if first_frame is None:
        return
    
    # 창 생성
    cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Filtered', 1600, 1200)

    # 트랙바 생성
    cv2.createTrackbar('H Min', 'Filtered', 0, 179, nothing)
    cv2.createTrackbar('S Min', 'Filtered', 0, 255, nothing)
    cv2.createTrackbar('V Min', 'Filtered', 0, 255, nothing)
    cv2.createTrackbar('H Max', 'Filtered', 0, 179, nothing)
    cv2.createTrackbar('S Max', 'Filtered', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Filtered', 0, 255, nothing)

    # 초기 트랙바 값 설정
    cv2.setTrackbarPos('H Max', 'Filtered', 179)
    cv2.setTrackbarPos('S Max', 'Filtered', 255)
    cv2.setTrackbarPos('V Max', 'Filtered', 255)

    while True:
        # 트랙바 값 읽기
        h_min = cv2.getTrackbarPos('H Min', 'Filtered')
        s_min = cv2.getTrackbarPos('S Min', 'Filtered')
        v_min = cv2.getTrackbarPos('V Min', 'Filtered')
        h_max = cv2.getTrackbarPos('H Max', 'Filtered')
        s_max = cv2.getTrackbarPos('S Max', 'Filtered')
        v_max = cv2.getTrackbarPos('V Max', 'Filtered')

        # HSV 변환
        hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

        # 마스크 생성
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # 결과 적용
        result = cv2.bitwise_and(first_frame, first_frame, mask=mask)

        # 결과 이미지 표시
        cv2.imshow('Filtered', result)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 프로그램 종료 시 트랙바 값 출력
    print(f"{video_path}'s HSV parameters are:")
    print(f'H Min: {h_min}, S Min: {s_min}, V Min: {v_min}')
    print(f'H Max: {h_max}, S Max: {s_max}, V Max: {v_max}')

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
