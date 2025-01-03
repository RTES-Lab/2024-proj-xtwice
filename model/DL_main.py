# main.py

import os
import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.metrics import classification_report

import funs

def main(yaml_config, target_config, save_model=False, save_log=False):
    
    ##############################
    # 1. initialize              
    ##############################
    funs.set_seed(yaml_config.seed)

    ##############################
    # 2. preprocessing           
    ##############################
    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = funs.get_dir_list(directory_list, target_view=target_config['view'])

    # 데이터프레임 제작
    df = funs.make_dataframe(yaml_config, directory)

    # 데이터 증강
    augmented_df = funs.augment_dataframe(df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    statistics_df = funs.add_statistics(augmented_df, target_config['axis'], is_standardize=True)

    print("총 데이터 개수:", len(statistics_df))
    fault_type_counts = statistics_df["fault_type"].value_counts()
    print(f"결함 별 데이터 개수:\n{fault_type_counts}") 

    ##############################
    # 3. plot figs - 삭제        
    ##############################
    
    ##############################
    # 4. train                   
    ##############################
    # 데이터, 라벨 얻기
    X, Y = funs.get_data_label(statistics_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')


    # PyTorch Tensor로 변환
    data_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Conv1d 입력 맞춤
    label_tensor = torch.tensor(Y, dtype=torch.long)

    # DataLoader 생성
    dataset = TensorDataset(data_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=yaml_config.batch_size, shuffle=True)

    model = funs.WDCNN(n_classes=len(set(Y)))
    trainer = funs.DLTrainer(yaml_config)

    # 훈련 수행
    accuracies, losses, all_y_true, all_y_pred, model_instance = trainer.kfold_training(model, dataloader)
    mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval = funs.calculate_result(accuracies, losses)

    all_y_true = np.concatenate([np.concatenate(arrays) for arrays in all_y_true])
    all_y_pred = np.concatenate([np.concatenate(arrays) for arrays in all_y_pred])

    report = classification_report(all_y_true, all_y_pred, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(report)

    class_accuracies = {}
    for class_label in np.unique(all_y_true):
        correct_class_predictions = np.sum((all_y_true == class_label) & (all_y_pred == class_label))
        total_class_samples = np.sum(all_y_true == class_label)
        class_accuracies[class_label] = correct_class_predictions / total_class_samples if total_class_samples > 0 else 0

    print("클래스별 정확도:")
    for class_label, accuracy in class_accuracies.items():
        print(f"클래스 {yaml_config.class2label_dic[class_label]}: {accuracy:.4f}")


    print(model_instance)

    ##############################
    # 5. save                    #
    ##############################
    # if save_log:
    #     current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     funs.log_results(
    #         file_path = yaml_config['log_txt'],
    #         timestamp=current_time,
    #         date = target_config['date'],
    #         input_feature=target_config['input_feature'],
    #         mean_accuracy=mean_accuracy,
    #         accuracy_confidence_interval=accuracy_confidence_interval,
    #         mean_loss=mean_loss,
    #         loss_confidence_interval=loss_confidence_interval,
    #         class2label_dic = yaml_config.class2label_dic,
    #         class_accuracies = class_accuracies,
    #         report=report,
    #     )


    best_model, best_history = trainer.get_best_model(model, dataloader)
    best_model = model_instance
    best_model.eval()


    # 모델의 속성을 확인하여 문제가 되는 데이터를 찾기
    # for name, value in vars(best_model).items():
    #     print(f"Attribute: {name}, Type: {type(value)}, Value: {value}")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = yaml_config.model_dir
    save_path = os.path.join(model_dir, f'wdcnn_{current_time}.pt')

    # 모델 저장 (state_dict)
    # torch.save(best_model.state_dict(), save_path)
    # print(f"State dict saved at {save_path}")

    csv_path = '/home/kseorang/Github/X-twice/MDPS-vis-master/output/1105/1105_6204_1201_H_F/z.csv'
    model_path = './saved/wdcnn_20250103_161025.ptl'  # 최적화된 모델 경로

    # 예측 수행
    predictions = predict_with_ptl(model_path, csv_path, yaml_config)
    predictions2 = predict_with_torch(best_model, csv_path)
    print("Predictions with ptl:", predictions)
    print("Predictions with torch:", predictions2)


    # JIT 스크립트 변환
    # try:
    #     # 모델 내부 데이터 타입 검사
    #     print("Checking model attributes and types...")
    #     for name, param in best_model.named_parameters():
    #         print(f"Param: {name}, Type: {type(param)}")

    #     scripted_model = torch.jit.script(best_model)
    #     optimized_scripted_module = optimize_for_mobile(scripted_model)
    #     lite_save_path = os.path.join(model_dir, f'wdcnn_{current_time}.ptl')
    #     optimized_scripted_module._save_for_lite_interpreter(lite_save_path)

        
    #     # jit_save_path = os.path.join(model_dir, f'wdcnn_{current_time}jit.pt')
    #     # scripted_model.save(jit_save_path)
    #     # print(f"Scripted model saved at {jit_save_path}")

    #     # torch.jit.save(optimized_scripted_module, lite_save_path)

    #     # print(f"Optimized model saved at {lite_save_path}")

    # except Exception as e:
    #     print(f"Error during JIT script or optimization: {e}")


    # if save_model:
    #     # 저장 경로 생성
    #     funs.export_to_executorch(best_model, (128, 1, 2048), yaml_config.model_dir)

def load_csv_data(csv_path, segment_length=2048):
    import pandas as pd
    # CSV 파일에서 데이터 로드
    data = pd.read_csv(csv_path, skiprows=1)

    # DisplacementZ(px) 열 추출
    displacement_z = data['DisplacementZ(px)'].values

    # 연속된 segment_length 크기로 나누기
    segments = []
    for start_idx in range(0, len(displacement_z) - segment_length + 1, segment_length):
        segment = displacement_z[start_idx:start_idx + segment_length]
        segments.append(segment)
    
    return np.array(segments)

def load_csv_data2(csv_path, segment_length=2048):
    import pandas as pd
    # CSV 파일에서 데이터 로드
    data = pd.read_csv(csv_path).iloc[10240:10240+segment_length, 0].values

    data=data.reshape(1, 2048)
    
    return np.array(data, dtype=np.float32)

def predict_with_ptl(model_path, csv_path, yaml_config):
    # 모델 로드
    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()

    # 데이터 로드
    segments = load_csv_data2(csv_path)

    # PyTorch 텐서로 변환
    data_tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)

    # 예측 수행
    with torch.no_grad():
        outputs = loaded_model(data_tensor)
        print(outputs)
        predictions = torch.argmax(outputs, dim=1)

    return predictions
    
def predict_with_torch(best_model, csv_path):
    # 모델 로드
    loaded_model = best_model
    loaded_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 데이터 로드
    segments = load_csv_data2(csv_path)
    print(segments)

    means = np.mean(segments)  # 각 열의 평균
    std_devs = np.std(segments)  # 각 열의 표준편차
    
    # 표준화
    standardized_data = (segments - means) / std_devs

    # PyTorch 텐서로 변환
    data_tensor = torch.tensor(standardized_data, dtype=torch.float32).unsqueeze(1)
    
    # 예측 수행
    with torch.no_grad():
        outputs = loaded_model(data_tensor)
        outputs = outputs.to(device)
        _, predictions = torch.max(outputs, 1)

    return predictions
    

if __name__ == "__main__":
    target_config = {
        'date': ['1105', '1217'],           # 필수
        # 'date': ['1011', '1012', '1024', '1102', '1105', '1217'],         # 필수
        # 'bearing_type': '6204',   # optional
        # 'RPM': '1200',            # optional
        'view': 'F',                # optional
        'axis': ['z'],              # 필수. ['z'] or ['x'] or ['z', 'x'] 
        'input_feature': 'z'    # 필수. 모델 input feature로 사용할 데이터
    }

    yaml_config = funs.load_yaml('./model_config.yaml')

    main(yaml_config, target_config, save_model=True, save_log=False)
