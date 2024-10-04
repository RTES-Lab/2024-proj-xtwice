import os
import sys

if sys.platform == 'win32':
    os.environ['PYTHONUTF8'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse

import cv2
import numpy as np

from csp import generate_csp
from data import *
from track import *
from pbm import PBM, generate_fir_bandpass
from utils import *

n_ma = 5
# 여기서 시간 조절 가능! 9999가 5분이였음. 999(30초 정도)
# 원래 999999
max_frames = 999999
cut_frames = 0

config = load_yaml('./process_and_extract_config.yaml')

def extract_displacement(filename, roi_file, output_dir, fps, frame_skip_rate, filter_order, freq_lb, freq_ub, alpha, hsv_min, hsv_max):
    video_sampling_rate = fps // (frame_skip_rate + 1)
    loop_range = filter_order + 1

    """ 시작 변위 제거 부분, Start.mp4로 시작되는 파일의 중앙을 빼는 부분"""
    # 확장자 앞에 _Start 추가
    parts = filename.rsplit('.mp4', 1)
    axis = parts[0][-1]
    filename_with_start = parts[0] + '_Start.mp4'
    mean_qx, mean_qy = find_center(filename_with_start, video_sampling_rate, frame_skip_rate, roi_file, hsv_min, hsv_max)

    """ end """

    stream = VideoFileReader(filename, video_sampling_rate, False, frame_skip_rate)

    img = stream.read()
    for _ in range(cut_frames):
        img = stream.read()
        if img is None:
            break

    idx_crop, idx_pbm, coord = region_from_json(roi_file)

    pyramid, pyr_idx = generate_csp((idx_pbm.h, idx_pbm.w), n_pyramids, orientation)
    fir_filter = generate_fir_bandpass(loop_range, video_sampling_rate, (freq_lb, freq_ub))
    pbm = PBM(fir_filter, pyramid, pyr_idx, alpha, attenuate_other_frequency, skip_pyramid_level)

    for _ in range(pbm.loop_range):
        img = stream.read()
        img_crop = crop_roi(img, idx_crop)
        img_pbm = crop_roi(img_crop, idx_pbm)
        x = stream.transform(img_pbm)[:,:,0]
        pbm.run(x, True)
    
    # (100, 150, 50), (140, 255, 255) -> (90, 120, 60), (115, 255, 255) -> (97, 0, 114), (115, 255, 231) -> (90, 50, 60), (140, 255, 255)
    tracker = MarkerCentroidTracker(tuple(hsv_min), tuple(hsv_max))
    tracker.roi = Location(x=0, y=0, w=img_crop.shape[1], h=img_crop.shape[0])
    tracker.track_region = coord

    qx = []
    qy = []

    cnt_frame = 0
    

    """주어진 동영상 파일에서 프레임 읽고, 특정 좌표 추척하여 X,Y 좌표를 저장하는 과정"""
    for _ in range(max_frames):
        print(cnt_frame+1)
        cnt_frame = cnt_frame+1

        #프레임 읽는 부분
        img = stream.read()
        if img is None:
            break
        
        # 영역 자르기(ROI), crop_roi 함수는 이미지에서 관심 영역을 자르는 역할을 함
        # idx_crop :은 잘라낼 영역의 좌표 정보를 포함
        img = crop_roi(img, idx_crop)


        # 이미지 처리
        # 프레임에서 또 다른 관심 영역(img_pbm)을 자르고, 그 영역에 PBM 적용하여 out_frame을 얻는다.
        # 필터가 적용된 결과를 원래 이미지의 해당 영역 안에 다시 삽입하여 업데이트 한다.
        if alpha != 0:
            img_pbm = crop_roi(img, idx_pbm)
            input_pbm = stream.transform(img_pbm)
            input_pbm[:,:,0] = pbm.run(input_pbm[:,:,0])
            out_frame = np.asarray(input_pbm.get())
            out_frame = array2img(yiq2bgr(out_frame))
            img[idx_pbm.y:idx_pbm.y+idx_pbm.h,idx_pbm.x:idx_pbm.x+idx_pbm.w] = out_frame

        # 마스크 처리 및 트랙포인트 추출
        """
        tracker.binarize(img)는 이미지를 이진화하여 마스크(mask)를 만듭니다. 이진화는 이미지에서 특정 범위의 색상 값을 구분하여 흑백 이미지를 만드는 과정입니다.
        tracker.extract_trackpoint(mask)는 이 마스크를 기반으로 트랙포인트(추적할 지점들)를 추출합니다. 이 트랙포인트는 특정 물체의 중심이나 좌표를 나타냅니다.
        """
        mask = tracker.binarize(img)
        trackpoints = tracker.extract_trackpoint(mask)

        # 좌표 저장
        qy.append([item.y for item in trackpoints])
        qx.append([item.x for item in trackpoints])
    
    qx = np.array(qx)
    qy = np.array(qy)
    
    # 뒤의 부분이 주석처리 되어 있으므로 m_qx, m_qy는 사용하지 않는다!(여기 주석 처리 해도 됨)
    m_qx = np.zeros((qx.shape[0]-n_ma+1, qx.shape[1]))
    m_qy = np.zeros((qy.shape[0]-n_ma+1, qy.shape[1]))


    """
    # qx.shape[1] : 열의 개수, 즉 몇개의 객체를 추적?
    moving_average() : example/utils.py에 존재
    """

    for i in range(qx.shape[1]):
        m_qx[:,i] = moving_average(qx[:,i], n_ma)
    for i in range(qy.shape[1]):
        m_qy[:,i] = moving_average(qy[:,i], n_ma)

    # 파일명은 원하는대로 설정할것
    # np.save(f"{output_dir}/new_xmag2.npy", qx)
    # np.save(f"{output_dir}/new_ymag2.npy", qy)

    """ """
    # hide_idx = np.where(qy == -1)
    # qqy = idx_crop.h - qy
    # qqx = idx_crop.w - qx
    # qqy[hide_idx] = -1


    print(qx.shape)
    print(mean_qx.shape)
    print(mean_qx)
    print(qy.shape)
    print(mean_qy.shape)
    print(mean_qy)

    """여기서 결국 qqx, qqy가 새로운 값인듯"""
    """"""
    qqx = qx - mean_qx
    qqy = qy - mean_qy

    np.savetxt(f"{output_dir}/{config.axis_to_csv_dic[axis][0]}", qqx, fmt="%.18f", delimiter=",")
    np.savetxt(f"{output_dir}/{config.axis_to_csv_dic[axis][1]}", qqy, fmt="%.18f", delimiter=",")

    # # 동영상은 만들지 않음
    # return

    """ 붉은 원 생성 부분""" # axis = 0 은 열 방향으로 평균을 내라는 뜻
    x0 = np.mean(m_qx[:50, :], axis=0)
    y0 = np.mean(m_qy[:50, :], axis=0)

    dqx = m_qx - x0
    dqy = m_qy - y0

    stream = VideoFileReader(filename, video_sampling_rate, False, frame_skip_rate)
    for _ in range(cut_frames):
        img = stream.read()
        if img is None:
            break
    # img = stream.read()

    for _ in range(pbm.loop_range):
        _ = stream.read()
    for _ in range(n_ma-1):
        _ = stream.read()

    width, height = idx_crop.w, idx_crop.h
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(f"{output_dir}/test_output.avi", fcc, 30, (width, height))

    idx = 0
    
    for _ in range(max_frames):
        print(idx)
        img = stream.read()
        if img is None:
            break
        if idx >= m_qy.shape[0]:
            break
        
        img = crop_roi(img, idx_crop)

        if alpha != 0:
            img_pbm = crop_roi(img, idx_pbm)
            input_pbm = stream.transform(img_pbm)
            input_pbm[:,:,0] = pbm.run(input_pbm[:,:,0])
            out_frame = np.asarray(input_pbm.get())
            out_frame = array2img(yiq2bgr(out_frame))
            img[idx_pbm.y:idx_pbm.y+idx_pbm.h,idx_pbm.x:idx_pbm.x+idx_pbm.w] = out_frame

        mag_ratio = 50

        # cv2.circle(원을 그릴 이미지 배열, 원의 중심 좌표, 원의 반지름 크기, 원의 색상, 원의 테두리 두께)
        # x0, y0 : 원래 원의 중심, dqx, dqy는 변화량, 변화량에 mag_ratio 곱해서 더 잘보이게 하는 것임!
        if m_qy.shape[1] == 7:
            for i in [0, 1, 2]:
                cv2.circle(img,
                        (round(x0[i]+dqx[idx, i]*mag_ratio), round(y0[i]+dqy[idx, i]*mag_ratio)),
                        color=COLOR_RED,
                        radius=50, thickness=10)
                if i != 2:
                    cv2.line(img,
                            (round(x0[i]+dqx[idx, i]*mag_ratio), round(y0[i]+dqy[idx, i]*mag_ratio)),
                            (round(x0[i+1]+dqx[idx, i+1]*mag_ratio), round(y0[i+1]+dqy[idx, i+1]*mag_ratio)),
                            color=COLOR_GREEN,
                            thickness=10)

            for i in [3, 4]:
                cv2.circle(img,
                        (round(x0[i]+dqx[idx, i]*mag_ratio), round(y0[i]+dqy[idx, i]*mag_ratio)),
                        color=COLOR_RED,
                        radius=50, thickness=10)
                if i != 4:
                    cv2.line(img,
                            (round(x0[i]+dqx[idx, i]*mag_ratio),round(y0[i]+dqy[idx, i]*mag_ratio)),
                            (round(x0[i+1]+dqx[idx, i+1]*mag_ratio), round(y0[i+1]+dqy[idx, i+1]*mag_ratio)),
                            color=COLOR_GREEN,
                            thickness=10)
            
            for i in [5, 6]:
                cv2.circle(img,
                        (round(m_qx[idx, i]), round(y0[i]+dqy[idx, i]*mag_ratio)),
                        color=COLOR_RED,
                        radius=50, thickness=10)
                if i != 6:
                    cv2.line(img,
                            (round(m_qx[idx, i]*mag_ratio),round(y0[i]+dqy[idx, i]*mag_ratio)),
                            (round(m_qx[idx, i+1]*mag_ratio), round(y0[i+1]+dqy[idx, i+1]*mag_ratio)),
                            color=COLOR_GREEN,
                            thickness=10)
        else:
            for i in range(m_qy.shape[1]):
                if i < m_qy.shape[1] - 2:
                    cv2.circle(img,
                            (round(x0[i]+dqx[idx, i]*mag_ratio), round(y0[i]+dqy[idx, i]*mag_ratio)),
                            color=COLOR_RED,
                            radius=50, thickness=10)
                else:
                    cv2.circle(img,
                        (round(m_qx[idx, i]), round(y0[i]+dqy[idx, i]*mag_ratio)),
                        color=COLOR_RED,
                        radius=50, thickness=10)

        idx = idx + 1
        out.write(img)
    
    out.release()

    return

    """붉은 원 생성 끝"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fname", "--filename", dest="filename", required=True, type=str)
    parser.add_argument("-f", "--fps", dest="fps", required=True, type=int)
    parser.add_argument("-skip", "--frame-skip-rate", dest="frame_skip_rate", required=True, type=int)
    parser.add_argument("-o", "--output", dest="output", required=True, type=str)
    parser.add_argument("-fo", "--filter-order", dest="filter_order", required=True, type=int)
    parser.add_argument("-flb", "--freq-lb", dest="freq_lb", required=True, type=float)
    parser.add_argument("-fub", "--freq-ub", dest="freq_ub", required=True, type=float)
    parser.add_argument("-a", "--alpha", dest="alpha", required=True, type=float)
    parser.add_argument("-roi", "--roi-file", dest="roi_file", required=True, type=str)
    parser.add_argument("-hsvmin", "--hsv-min", dest="hsv_min", required=True, nargs='+', type=int)
    parser.add_argument("-hsvmax", "--hsv-mac", dest="hsv_max", required=True, nargs='+', type=int)
    
    args = parser.parse_args()

    if os.path.isdir(args.output):
        raise ValueError("Existing directory")
    if os.path.isfile(args.output):
        raise ValueError("Existing file name")
    os.makedirs(args.output)
    if not os.path.isfile(args.roi_file):
        raise ValueError("No ROI file")
    
    if args.fps <= 0:
        raise ValueError("FPS must be > 0")
    if args.filter_order <= 0:
        raise ValueError("Filter order must be > 0")
    if args.freq_lb <= 0 or args.freq_lb >= args.freq_ub:
        raise ValueError("Lowcut frequency must be > 0 and lower than highcut frequency")
    if args.freq_ub <= 0 or args.freq_ub <= args.freq_lb:
        raise ValueError("Highcut frequency must be > 0 and higher than lowcut frequency")
    if args.alpha < 0:
        raise ValueError("alpha must be >= 0")
    if args.frame_skip_rate < 0:
        raise ValueError("Frame skip rate must be >= 0")

    extract_displacement(filename=args.filename,
                         roi_file=args.roi_file,
                         output_dir=args.output,
                         fps=args.fps,
                         frame_skip_rate=args.frame_skip_rate,
                         filter_order=args.filter_order,
                         freq_lb=args.freq_lb,
                         freq_ub=args.freq_ub,
                         alpha=args.alpha,
                         hsv_min = args.hsv_min,
                         hsv_max = args.hsv_max
                        )