"""
변위 추출 파이썬 파일을 한번에 실행할 수 있는 파이썬 파일
"""
import os
from utils import *


config = load_yaml('./process_and_extract_config.yaml')

input_mov_file = f"{config.date}_{config.bearing_type}_{config.fault_type}_{config.axis}.mov"
json_file = f"{config.date}_{config.bearing_type}_{config.fault_type}_{config.axis}.json"


get_roi_command = (
    f"python ./example/get_roi.py "
    f"-fname ./videos/{input_mov_file} "
    f"-f {config.fps} "
    f"-o ./test/{json_file}"
)

extract_displacement_conmmand = (
    f"python ./example/extract_displacement.py "
    f"-fname ./videos/{input_mov_file} "
    f"-f {config.fps} -skip {config.skip} "
    f"-o ./output/{config.date}/{config.date}_{config.bearing_type}_{config.fault_type}_{config.axis} "
    f"-fo {config.fo} -flb {config.flb} -fub {config.fub} -a {config.a} "
    f"-roi ./test/{json_file} "
)


if __name__=="__main__":
    os.system(get_roi_command)
    os.system(extract_displacement_conmmand)

