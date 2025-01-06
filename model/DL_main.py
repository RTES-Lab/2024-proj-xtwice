# main.py

import os
import datetime
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.metrics import classification_report

import funs


def main(
        yaml_config, target_config, save_displacement_figs=False, 
        save_model=False, save_log=False, compare=False
        ):
    
    ##############################
    # 1. initialize              
    ##############################
    # funs.set_seed(yaml_config.seed)

    ##############################
    # 2. preprocessing           
    ##############################
    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = funs.get_dir_list(directory_list, target_view=target_config['view'])

    # 데이터프레임 제작
    df = funs.make_dataframe(yaml_config, directory)

    # 데이터 증강
    augmented_df = funs.augment_dataframe(df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    print("총 데이터 개수:", len(augmented_df))
    fault_type_counts = augmented_df["fault_type"].value_counts()
    print(f"결함 별 데이터 개수:\n{fault_type_counts}") 


    ##############################
    # 3. plot figs     
    ##############################
    # 변위 데이터 플롯
    if len(target_config['axis']) == 1:
        axis_name = target_config['axis'][0]
    elif len(target_config['axis']) == 2:
        axis_name = f"{target_config['axis'][0]}, {target_config['axis'][1]}"

    if save_displacement_figs:
        funs.get_displacement_pic(augmented_df, axis_name, target_config['date'])
    
    ##############################
    # 4. train                   
    ##############################
    # 데이터, 라벨 얻기
    X, Y = funs.get_data_label(augmented_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')


    # PyTorch Tensor로 변환
    data_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Conv1d 입력 맞춤
    label_tensor = torch.tensor(Y, dtype=torch.long)

    # DataLoader 생성
    dataset = TensorDataset(data_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=yaml_config.batch_size, shuffle=True)

    model = funs.WDCNN(n_classes=len(set(Y)))
    trainer = funs.DLTrainer(yaml_config)

    accuracies, losses, all_y_true, all_y_pred, best_model, _ = trainer.get_best_model(model, dataloader)
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

    
    ##############################
    # 5. save                    
    ##############################
    if save_log:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        funs.log_results(
            file_path = yaml_config['log_txt'],
            timestamp=current_time,
            date = target_config['date'],
            input_feature=target_config['input_feature'],
            mean_accuracy=mean_accuracy,
            accuracy_confidence_interval=accuracy_confidence_interval,
            mean_loss=mean_loss,
            loss_confidence_interval=loss_confidence_interval,
            class2label_dic = yaml_config.class2label_dic,
            class_accuracies = class_accuracies,
            report=report,
        )

    if save_model:
        # pth 모델 저장
        model_name = 'wdcnn'
        tmp_model_save_path = os.path.join(yaml_config.model_dir, f'{model_name}.pth')
        torch.save(best_model.state_dict(), tmp_model_save_path)

        best_model = model.get_model(n_classes=4)
        best_model.load_state_dict(torch.load(tmp_model_save_path, weights_only=True))

        # ptl 모델 저장
        model_save_path = os.path.join(yaml_config.model_dir, f'{model_name}.ptl')
        scripted_model = torch.jit.script(best_model)
        optimized_scripted_module = optimize_for_mobile(scripted_model)
        optimized_scripted_module._save_for_lite_interpreter(model_save_path)
        print(f"모델이 {model_save_path}에 저장되었습니다.")
        
    if compare:
        """ funs.set_seed(yaml_config.seed) 를 주석처리 할 것 """
        # ptl 모델과 torch 모델 결과 비교
        funs.compare_torch_n_ptl(augmented_df, model_save_path, best_model)

if __name__ == "__main__":
    target_config = {
        'date': ['1105', '1217'],           # 필수
        # 'date': ['1011', '1012', '1024', '1102', '1105', '1217'],         # 필수
        # 'bearing_type': '6204',   # optional
        # 'RPM': '1200',            # optional
        'view': 'F',                # optional
        'axis': ['z'],              # 필수. ['z'] or ['x'] or ['z', 'x'] 
        'input_feature': 'z'    # 필수. 딥러닝의 경우 반드시 'z'
    }

    yaml_config = funs.load_yaml('./model_config.yaml')

    main(yaml_config, target_config, save_displacement_figs=False, save_model=True, save_log=False, compare=False)
