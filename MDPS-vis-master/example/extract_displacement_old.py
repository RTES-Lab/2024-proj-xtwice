import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse

import cv2
import numpy as np

from csp import generate_csp
from data import *
from track import *
from pbm import PBM, generate_fir_bandpass
from utils import *

# 이동평균 변수. 작을수록 추출된 영상의 ROI 변동성이 심해짐 -> csv 파일에는 영향을 주지 않고, 영상에서 빨간색 원의 가시화 정도를 위해 설정된 변수
n_ma = 5

max_frames = 9999
cut_frames = 0

def extract_displacement(filename, roi_file, output_dir, fps, frame_skip_rate, filter_order, freq_lb, freq_ub, alpha):
    video_sampling_rate = fps // (frame_skip_rate + 1)
    loop_range = filter_order + 1

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
    tracker = MarkerCentroidTracker((89, 33, 86), (179, 255, 255))
    tracker.roi = Location(x=0, y=0, w=img_crop.shape[1], h=img_crop.shape[0])
    tracker.track_region = coord

    qx = []
    qy = []

    cnt_frame = 0
    
    for _ in range(max_frames):
        print(cnt_frame+1)
        cnt_frame = cnt_frame+1
        img = stream.read()
        if img is None:
            break
        img = crop_roi(img, idx_crop)

        if alpha != 0:
            img_pbm = crop_roi(img, idx_pbm)
            input_pbm = stream.transform(img_pbm)
            input_pbm[:,:,0] = pbm.run(input_pbm[:,:,0])
            out_frame = np.asarray(input_pbm.get())
            out_frame = array2img(yiq2bgr(out_frame))
            img[idx_pbm.y:idx_pbm.y+idx_pbm.h,idx_pbm.x:idx_pbm.x+idx_pbm.w] = out_frame

        mask = tracker.binarize(img)
        trackpoints = tracker.extract_trackpoint(mask)
        qy.append([item.y for item in trackpoints])
        qx.append([item.x for item in trackpoints])
    
    qx = np.array(qx)
    qy = np.array(qy)
    
    m_qx = np.zeros((qx.shape[0]-n_ma+1, qx.shape[1]))
    m_qy = np.zeros((qy.shape[0]-n_ma+1, qy.shape[1]))

    for i in range(qx.shape[1]):
        m_qx[:,i] = moving_average(qx[:,i], n_ma)
    for i in range(qy.shape[1]):
        m_qy[:,i] = moving_average(qy[:,i], n_ma)

    # 파일명은 원하는대로 설정할것
    # np.save(f"{output_dir}/new_xmag2.npy", qx)
    # np.save(f"{output_dir}/new_ymag2.npy", qy)

    hide_idx = np.where(qy == -1)
    qqy = idx_crop.h - qy
    qqx = idx_crop.w - qx
    qqy[hide_idx] = -1

    # 여기서 파일 명을 설정할 수 있음! 영상에 따라 x, y, z 조절하기!
    np.savetxt(f"{output_dir}/x.csv", qqx, fmt="%.18f", delimiter=",")
    np.savetxt(f"{output_dir}/z.csv", qqy, fmt="%.18f", delimiter=",")


    """원 생성"""
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
                         alpha=args.alpha)