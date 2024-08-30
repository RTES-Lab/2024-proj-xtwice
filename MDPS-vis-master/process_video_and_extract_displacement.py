"""
변위 추출 파이썬 파일을 한번에 실행할 수 있는 프로그램
더불어, 지금까지 사용한 threshold를 값을 기록한다.
"""


import os

# 영상 날짜
date: str = "0814"

# 영상 위치 번호
axis: str = "X"

# 영상에 사용된 베어링 타입
bearing_fault_type: str = "B"

filename = f"{date}_30204_{bearing_fault_type}_{axis}"

os.system(f"python ./example/get_roi.py -fname ../videos/{filename}/{filename}.mov -f 120 -o ../test/{filename}.json")
os.system(f"python ./example/extract_displacement.py -fname ../videos/{filename}/{filename}.mov -f 120 -skip 0 -o ../output/{filename} -fo 14 -flb 0.01 -fub 1.0 -a 0 -roi ../test/{filename}.json ")

'''
B, 1 :     tracker = MarkerCentroidTracker((90, 75, 90), (128, 255, 255))
B, 2 :    tracker = MarkerCentroidTracker((103, 189, 0), (179, 255, 255))
B, 3 :     tracker = MarkerCentroidTracker((100, 167, 219), (179, 255, 255))
B, 4 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))
B, 5 :     tracker = MarkerCentroidTracker((25, 125, 125), (100, 255, 255))
B, 6 :     tracker = MarkerCentroidTracker((100, 200, 120), (179, 255, 255))
B, 7 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))

IR, 1 :     tracker = MarkerCentroidTracker((28, 75, 75), (121, 255, 255))
IR, 2 :     tracker = MarkerCentroidTracker((100,140, 100), (179, 255, 255))
IR, 3 :     tracker = MarkerCentroidTracker((100, 130, 170), (179, 255, 255))
IR, 4 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))
IR, 5 :     tracker = MarkerCentroidTracker((25, 125, 125), (100, 255, 255))
IR, 6 :     tracker = MarkerCentroidTracker((100, 200, 120), (179, 255, 255))
'''