import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# 초기 시작점과 끝점 저장 변수
start_point = None
end_point = None

# 직선 그리기 중인지 여부
drawing = False
# 다시 그린 직선인지 여부
is_modified = False

def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing, first_frame, temp_frame, is_modified

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_modified:
            cv2.destroyWindow('line')
        # 마우스 왼쪽 버튼 누르면 시작점 설정 및 drawing 활성화
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 움직임 이벤트, drawing 활성화 시 끝점 업데이트 및 이미지 갱신
        if drawing:
            end_point = (x, y)
            temp_frame = first_frame.copy()
            cv2.line(temp_frame, start_point, end_point, (0, 255, 0), 2)
            # cv2.imshow('line', temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 왼쪽 버튼 떼면 drawing 비활성화 및 직선 길이 계산
        drawing = False
        end_point = (x, y)
        cv2.line(temp_frame, start_point, end_point, (0, 255, 0), 2)
        is_modified = True

def get_first_frame(video_path: str):
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

def get_displacment():
    global first_frame, temp_frame, real_length, pixel_length
    video_path = f"../../input/{filename}.mov"  # 동영상 파일 경로 입력
    first_frame = get_first_frame(video_path)
    
    if first_frame is None:
        return

    temp_frame = first_frame.copy()
    
    # 창 생성
    cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Filtered', 1600, 1200)

    # 마우스 콜백 함수 설정
    cv2.setMouseCallback('Filtered', draw_line)

    while True:
        # 결과 이미지 표시
        cv2.imshow('Filtered', temp_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # 직선의 길이와 변위 출력
    if end_point is not None and start_point is not None:
        pixel_length = np.linalg.norm(np.array(start_point) - np.array(end_point))
        print(f'Pixel length: {pixel_length:.3f} pixels')
        print(f'mm per pixel: {real_length/pixel_length:.3f} mm')

    cv2.destroyAllWindows()
    mm_per_pixel = real_length/pixel_length  

    return mm_per_pixel


def get_graph(csv_file : str, mm_per_pixel):

    file = csv_file  # 파일 경로를 지정하세요.
    df = pd.read_csv(file, usecols=[0, 1, 2, 3, 4, 5, 6], names=['A', 'B', 'C', 'D', 'E', 'F', 'G'], header=0)  # 첫 4개 열만 불러옵니다.

    # 각 열의 평균을 구합니다.
    means = df.mean()


    # 각 열의 모든 값에서 각 열의 평균을 뺍니다.
    df_centered = df - means

    for column in df_centered.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df_centered.index/120, df_centered[column]*(mm_per_pixel))
        plt.title(f'Centered Values of {column}', size=15)
        plt.xlabel('Time[s]', size=15)
        plt.ylabel('Displacement[mm]', size=15)
        plt.ylim(-0.1, 0.1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13) 
        plt.savefig(f'{filename}_{column}_{csv_file[0].upper()}_axis.png')
        plt.close()

    print("Graphs saved successfully.")

def main():
    mm_per_pixel = get_displacment()
    get_graph('x.csv', mm_per_pixel)
    get_graph('y.csv', mm_per_pixel)

if __name__ == "__main__":
    # 현재 스크립트의 경로를 가져옴
    script_path = os.path.dirname(os.path.realpath(__file__))

    # 해당 경로로 이동
    os.chdir(script_path)

    filename = os.getcwd().split('/')[-1]
    if filename[-1] == "7":
        real_length = 30
    elif filename[-1] == "1":
        real_length = 65
    main()
