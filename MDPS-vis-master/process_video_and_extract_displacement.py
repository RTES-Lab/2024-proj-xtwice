"""
변위 추출 파이썬 파일을 한번에 실행할 수 있는 파이썬 파일
"""
import os
from utils import *


config = load_yaml('./config.yaml')

input_mov_file = f"{config.date}_{config.bearing_type}_{config.fault_type}_{config.rpm}.mov"
json_file = f"{config.date}_{config.bearing_type}_{config.fault_type}_{config.rpm}.json"

get_roi_command = (
    f"python ./example/get_roi.py "
    f"-fname ./input/{input_mov_file} "
    f"-f {config.fps} "
    f"-o ./test/{json_file}"
)

extract_displacement_conmmand = (
    f"python ./example/extract_displacement.py "
    f"-fname ./input/{input_mov_file} "
    f"-f {config.fps} -skip {config.skip} "
    f"-o ./output/{config.date}/{config.date}_{config.bearing_type}_{config.fault_type}_{config.rpm} "
    f"-fo {config.fo} -flb {config.flb} -fub {config.fub} -a {config.a} "
    f"-roi ./test/{json_file} "
)


if __name__=="__main__":
    os.system(get_roi_command)
    os.system(extract_displacement_conmmand)





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