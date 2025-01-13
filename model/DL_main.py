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
    
    # import pandas as pd
    # csv_file = pd.read_csv('./data_0109.csv')

    # correct = 0

    # # 한 행씩 읽어서 처리
    # for index, row in csv_file.iterrows():
    #     # z 열의 데이터를 문자열로 가져옴
    #     z_list_str = row['z']
        
    #     # 문자열을 공백으로 분리하여 리스트로 변환
    #     try:
    #         z_list = [float(value) for value in z_list_str.split()]
    #     except ValueError as e:
    #         print(f"Error parsing z_list at row {index}: {e}")
    #         continue  # 이 행을 건너뜁니다.
        
    #     # 모델 로드
    #     loaded_model = torch.jit.load('./saved/wdcnn.ptl')
    #     loaded_model.eval()

    #     # z_list를 텐서로 변환 (모델에 맞는 형태로 변환)
    #     data_tensor = torch.tensor(z_list, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # 배치 차원과 채널 차원 추가

    #     # Python에 이 로그 추가
    #     print("First 5 values:", z_list[:5])
    #     print("Input shape:", data_tensor.shape)

    #     # 예측 수행
    #     with torch.no_grad():
    #         print("valiadation...")
    #         outputs = loaded_model(data_tensor) 
    #         print(outputs)
    #         predictions = torch.argmax(outputs, dim=1)  # 예측된 클래스 추출
        

    #     # 예측값이 실제 라벨과 일치하는지 확인
    #     if predictions.item() == int(row['label']):  # 예측값과 실제 라벨을 비교
    #         correct += 1

    # # 최종 정확도 출력
    # print(f"Accuracy: {correct}/{len(csv_file)}")

    # return



    ##############################
    # 2. preprocessing           
    ##############################
    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = funs.get_dir_list(directory_list, target_view=target_config['view'])

    # 데이터프레임 제작
    df = funs.make_dataframe(yaml_config, directory)
    # train_df, val_df, test_df = funs.split_dataframe(df, 0.7, 0.3)

    # 데이터 증강
    df = funs.augment_dataframe(df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    # df['z'] = df['z'].apply(lambda x: ' '.join(map(str, x)))

    # output_path = './data_0109.csv'
    # df.to_csv(output_path, index=False)

    # print(f"데이터가 {output_path}에 저장되었습니다.")


    # return

    # val_df = funs.augment_dataframe(val_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    # test_df = funs.augment_dataframe(test_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    # print("총 데이터 개수:", len(train_df)+len(val_df)+len(test_df))
    # fault_type_counts = train_df["fault_type"].value_counts()
    # fault_type_counts += val_df["fault_type"].value_counts()
    # fault_type_counts += test_df["fault_type"].value_counts()
    # print(f"결함 별 데이터 개수:\n{fault_type_counts}") 

    
    ##############################
    # 3. train                   
    ##############################
    # 데이터, 라벨 얻기
    X, Y = funs.get_data_label(df, target_config['input_feature'])
    # X_val, y_val = funs.get_data_label(val_df, target_config['input_feature'])
    # X_test, y_test = funs.get_data_label(test_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    # PyTorch Tensor로 변환
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
    Y = torch.tensor(Y, dtype=torch.long)
    # X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # y_val = torch.tensor(y_val, dtype=torch.long)
    # y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoader 생성
    dataset = TensorDataset(X, Y)
    # val_dataset = TensorDataset(X_val, y_val)
    # test_dataset = TensorDataset(X_test, y_test)

    data_loader = DataLoader(dataset, batch_size=yaml_config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=yaml_config.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=yaml_config.batch_size, shuffle=False)

    model = funs.WDCNN(n_classes=4)
    trainer = funs.DLTrainer(yaml_config)

    accuracies, losses, all_y_true, all_y_pred, best_model, best_fold_history = trainer.get_best_model(model, data_loader)
    mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval = funs.calculate_result(accuracies, losses)


    all_y_true = np.concatenate([np.concatenate(arrays) for arrays in all_y_true])
    all_y_pred = np.concatenate([np.concatenate(arrays) for arrays in all_y_pred])

    # print(test_pred_list)

    print()
    print("val 평균 정확도:", mean_accuracy)
    print("val 평균 손실:", mean_loss)
    print("테스트 성능")
    # print("test 정확도:", test_accuracy)
    # print("test 손실:", test_loss)

    val_report = classification_report(all_y_true, all_y_pred, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(val_report)
    class_accuracies = {}
    for class_label in np.unique(all_y_true):
        correct_class_predictions = np.sum((all_y_true == class_label) & (all_y_pred == class_label))
        total_class_samples = np.sum(all_y_true == class_label)
        class_accuracies[class_label] = correct_class_predictions / total_class_samples if total_class_samples > 0 else 0

    print("클래스별 정확도:")
    for class_label, accuracy in class_accuracies.items():
        print(f"클래스 {yaml_config.class2label_dic[class_label]}: {accuracy:.4f}")

    # test_report = classification_report(test_true_list, test_pred_list, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    # print('클래스별 성능 보고서')
    # print(test_report)

    # class_accuracies = {}
    # for class_label in np.unique(test_true_list):
    #     correct_class_predictions = np.sum((test_true_list == class_label) & (test_pred_list == class_label))
    #     total_class_samples = np.sum(test_true_list == class_label)
    #     class_accuracies[class_label] = correct_class_predictions / total_class_samples if total_class_samples > 0 else 0

    # print("클래스별 정확도:")
    # for class_label, accuracy in class_accuracies.items():
    #     print(f"클래스 {yaml_config.class2label_dic[class_label]}: {accuracy:.4f}")

    
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
            mean_accuracy        = mean_accuracy,
            mean_loss            = mean_loss,
            class2label_dic     = yaml_config.class2label_dic,
            class_accuracies    = class_accuracies,
            val_report          = val_report,
            accuracy_confidence_interval= accuracy_confidence_interval,
            loss_confidence_interval=loss_confidence_interval
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
    

    # for _ in range(100):
    #     funs.compare_torch_n_ptl(df, "./saved/wdcnn.ptl", best_model)
        

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
