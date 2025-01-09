# main.py

import os
import datetime

import tensorflow as tf
from sklearn.metrics import classification_report

import funs

import numpy as np


def main(
        yaml_config, target_config, save_model=False, 
        save_log=False
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

    # 통계값 값 추가
    train_df, val_df, test_df = funs.add_statistics(train_df, val_df, test_df, target_config['axis'], is_standardize=True)

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

    model = funs.ANN()
    trainer = funs.Trainer(yaml_config)
    
    val_accuracy, val_loss, test_accuracy, test_loss, test_true_list, test_pred_list, val_true_list, val_pred_list = trainer.get_best_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"""
Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}
Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}
    """)

    val_true_list = np.concatenate(val_true_list, axis=0)
    val_pred_list = np.concatenate(val_pred_list, axis=0)
    test_true_list = np.concatenate(test_true_list, axis=0)
    test_pred_list = np.concatenate(test_pred_list, axis=0)

    val_report = classification_report(val_true_list, val_pred_list, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('Validation 클래스별 성능 보고서')
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
    print('Test 클래스별 성능 보고서')
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
    # if save_log:
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # funs.log_results(
        #     model_name="ANN",
        #     file_path = yaml_config['log_txt'],
        #     timestamp=current_time,
        #     date = target_config['date'],
        #     input_feature=target_config['input_feature'],
        #     mean_accuracy=accuracies,
        #     mean_loss=losses,
        #     class2label_dic = yaml_config.class2label_dic,
        #     class_accuracies = class_accuracies,
        #     report=report,
        # )


    # 모델 저장
    if save_model:
        dummy = np.random.random((667, 4))
        model = model.get_model(dummy)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_dir = yaml_config.model_dir
        model_filename = f"{'ANN'}_{target_config['input_feature']}.tflite"
        model_path = os.path.join(model_dir, model_filename)

        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)



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
        save_model=args.save_model, save_log=args.save_log
        )
