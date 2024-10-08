'''
변위 추출을 한번에 실행할 수 있는 파이썬 파일 
변위를 추출하고자 하는 영상의 날짜, 결함 종류, axis 등을 제대로 입력해야
원하는 결과를 얻을 수 있음에 유의
'''

import sys

if sys.platform == 'win32':
    import os
    os.environ['PYTHONUTF8'] = '1'

import os
from utils import *

from get_threshold import get_first_frame, get_hsv


def get_hsv_param(input_mov_file: str):
    '''
    hsv 파라미터 구하는 함수
    '''

    input_video = f'{yaml_config.video_root}/{input_mov_file}'
    first_frame = get_first_frame(input_video)

    # hsv 파라미터(hsv_min, hsv_max) 계산
    hsv_min, hsv_max  = get_hsv(first_frame)

    print(f"{input_mov_file}'s HSV parameters are:")
    print(f'H Min: {hsv_min[0]}, S Min: {hsv_min[1]}, V Min: {hsv_min[2]}')
    print(f'H Max: {hsv_max[0]}, S Max: {hsv_max[1]}, V Max: {hsv_max[2]}')

    # hsv 파라미터 값 리턴
    return list(hsv_min), list(hsv_max)


def get_roi(input_mov_file: str, json_file: str, input_dir: str):
    '''
    get_roi.py 실행
    '''
    get_roi_command = (
    f"python ./example/get_roi.py "
    f"-fname {yaml_config.video_root}/{input_mov_file} "
    f"-f {yaml_config.fps} "
    f"-o ./input/{input_dir}/{json_file}"
    )

    os.system(get_roi_command)


def extract_displacement(input_mov_file: str, json_file: str, output_dir: str, input_dir:str, hsv_params: tuple, RPM: int):
    '''
    extract_displacement.py 실행
    이 과정에서 쓰인 hsv 값은 get_hsv_param 함수에서 리턴받은 hsv 파라미터의 값을 사용하게 됨.
    '''
    # hsv 파라미터
    hsv_min, hsv_max = hsv_params # hsv_params
    # 여기서 hsv 값을 직접 주고 싶으면 (105, 125, 194), (179, 255, 255) 이런 식으로 하면 됨

    output_dir = os.path.join(yaml_config.output_root, output_dir)

    # extract_displacement.py 실행
    extract_displacement_conmmand = (
    f"python ./example/extract_displacement.py "
    f"-fname {yaml_config.video_root}/{input_mov_file} "
    f"-f {RPM} -skip {yaml_config.skip} "
    f"-o {output_dir} "
    f"-fo {yaml_config.fo} -flb {yaml_config.flb} -fub {yaml_config.fub} -a {yaml_config.a} "
    f"-roi ./input/{input_dir}/{json_file} "
    f"-hsvmin {hsv_min[0]} {hsv_min[1]} {hsv_min[2]} "
    f"-hsvmax {hsv_max[0]} {hsv_max[1]} {hsv_max[2]}"
    )

    os.system(extract_displacement_conmmand)


def process_video_and_extract_displacement():
    '''
    hsv 값 구하기와 변위 추출을 한번에 수행할 수 있는 함수
    '''
    # 변위를 추출할 동영상 파일과 해당 영상의 정보가 담긴 json 파일, output 디렉토리 이름 명시
    # 동영상 파일 이름, json 파일 이름, output 디렉토리 이름 등의 형식은 지켜져야 함
    input_data = f"{target_config['date']}_{target_config['bearing_type']}_{target_config['RPM']}_{target_config['fault_type']}_{target_config['axis']}"
    input_dir = os.path.join(f"{target_config['date']}/{input_data}")

    input_mov_file = os.path.join(input_dir, f"{input_data}.mp4")
    json_file = f"{target_config['date']}_{target_config['bearing_type']}_{target_config['RPM']}_{target_config['fault_type']}_{target_config['axis']}.json"
    output_dir = f"{target_config['date']}/{target_config['date']}_{target_config['bearing_type']}_{target_config['RPM']}_{target_config['fault_type']}_{target_config['axis']}"

    print('input_mov_file is: ', input_mov_file)

    # hsv 파라미터 추출
    hsv_params = get_hsv_param(input_mov_file)

    # .py 코드 실행
    get_roi(input_mov_file, json_file, input_dir)
    extract_displacement(input_mov_file, json_file, output_dir, input_dir, hsv_params, target_config['RPM'])


if __name__=="__main__":

    ########################
    # 주로 수정해야 할 부분
    target_config = {
        'date': '1008',
        'bearing_type': '30204',
        'RPM': '1200',
        'fault_type': 'H',
        'axis': 'S'
    }
    ########################
    yaml_config = load_yaml('./process_and_extract_config.yaml')

    process_video_and_extract_displacement()
