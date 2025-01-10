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
        yaml_config, target_config,
        save_model=False, save_log=False, compare=False
        ):
    
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
    train_df, val_df, test_df = funs.split_dataframe(df, 0.7, 0.3)

    # 데이터 증강
    train_df = funs.augment_dataframe(train_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    val_df = funs.augment_dataframe(val_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    test_df = funs.augment_dataframe(test_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    print("총 데이터 개수:", len(train_df)+len(val_df)+len(test_df))
    fault_type_counts = train_df["fault_type"].value_counts()
    fault_type_counts += val_df["fault_type"].value_counts()
    fault_type_counts += test_df["fault_type"].value_counts()
    print(f"결함 별 데이터 개수:\n{fault_type_counts}") 

    
    ##############################
    # 3. train                   
    ##############################
    # 데이터, 라벨 얻기
    X_train, y_train = funs.get_data_label(train_df, target_config['input_feature'])
    X_val, y_val = funs.get_data_label(val_df, target_config['input_feature'])
    X_test, y_test = funs.get_data_label(test_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    # PyTorch Tensor로 변환
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) 
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)  
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=yaml_config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=yaml_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=yaml_config.batch_size, shuffle=False)

    model = funs.WDCNN(n_classes=4)
    trainer = funs.DLTrainer(yaml_config)

    val_accuracy_list, val_loss_list, val_true_list, val_pred_list, best_model, best_history, test_accuracy, test_loss, test_true_list, test_pred_list = trainer.get_best_model(model, train_loader, val_loader, test_loader)

    val_acc = np.mean(val_accuracy_list)
    val_loss = np.mean(val_loss_list)
    test_accuracy = np.mean(test_accuracy)
    test_loss = np.mean(test_loss)

    val_true_list = np.concatenate(val_true_list, axis=0)
    val_pred_list = np.concatenate(val_pred_list, axis=0)


    print()
    print("val 평균 정확도:", val_acc)
    print("val 평균 손실:", val_loss)
    print("테스트 성능")
    print("test 정확도:", test_accuracy)
    print("test 손실:", test_loss)

    val_report = classification_report(val_true_list, val_pred_list, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(val_report)
    class_accuracies = {}
    for class_label in np.unique(val_true_list):
        correct_class_predictions = np.sum((val_true_list == class_label) & (val_pred_list == class_label))
        total_class_samples = np.sum(val_true_list == class_label)
        class_accuracies[class_label] = correct_class_predictions / total_class_samples if total_class_samples > 0 else 0

    print("클래스별 정확도:")
    for class_label, accuracy in class_accuracies.items():
        print(f"클래스 {yaml_config.class2label_dic[class_label]}: {accuracy:.4f}")

    test_report = classification_report(test_true_list, test_pred_list, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(test_report)

    class_accuracies = {}
    for class_label in np.unique(test_true_list):
        correct_class_predictions = np.sum((test_true_list == class_label) & (test_pred_list == class_label))
        total_class_samples = np.sum(test_true_list == class_label)
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
            model_name          = "WDCNN",
            file_path           = yaml_config['log_csv'],
            timestamp           = current_time,
            date                = target_config['date'],
            input_feature       = target_config['input_feature'],
            val_accuracy        = val_acc,
            val_loss            = val_loss,
            test_accuracy       = test_accuracy,
            test_loss           = test_loss,
            class2label_dic     = yaml_config.class2label_dic,
            class_accuracies    = class_accuracies,
            val_report          = val_report,
            test_report         = test_report
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
        

if __name__ == "__main__":
    args = funs.parse_arguments()

    target_config = {
        'date': args.dates,
        'view': args.view,
        'axis': args.axis,
        'input_feature': args.input_feature
    }

    yaml_config = funs.load_yaml('./model_config.yaml')

    main(
        yaml_config, target_config, 
        save_model=args.save_model, save_log=args.save_log, compare=args.compare)
