import time
import psutil
import os

import tensorflow as tf
import torch

def measure_inference_time(model, input_data, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.time()
        model(input_data)  # 또는 model(input_data)
        times.append(time.time() - start)
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times)
    }

def measure_memory_usage(model, input_data):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    model(input_data)
    mem_after = process.memory_info().rss
    return mem_after - mem_before

def compare_models(tflite_model, ptl_model, input_data):
    # 모델 파일 크기
    tflite_size = os.path.getsize('model.tflite')
    ptl_size = os.path.getsize('model.ptl')
    
    # 추론 시간
    tflite_timing = measure_inference_time(tflite_model, input_data)
    ptl_timing = measure_inference_time(ptl_model, input_data)
    
    # 메모리 사용량
    tflite_memory = measure_memory_usage(tflite_model, input_data)
    ptl_memory = measure_memory_usage(ptl_model, input_data)
    
    return {
        'file_size': {
            'tflite': tflite_size,
            'ptl': ptl_size
        },
        'inference_time': {
            'tflite': tflite_timing,
            'ptl': ptl_timing
        },
        'memory_usage': {
            'tflite': tflite_memory,
            'ptl': ptl_memory
        }
    }

def main():
    tflite_model = tf.lite.Interpreter("./saved/ANN_z_fused_features.tflite")
    ptl_model = torch.jit.load("./saved/wdcnn.ptl")
    compare_models(tflite_model, ptl_model, input_data)