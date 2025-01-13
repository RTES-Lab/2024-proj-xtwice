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

    # 데이터 증강
    df = funs.augment_dataframe(df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    df = funs.add_statistics(df, target_config['axis'], is_standardize=False)

    ##############################
    # 3. train                   
    ##############################
    # 데이터, 라벨 얻기
    X, Y = funs.get_data_label(df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    model = funs.ANN()
    trainer = funs.Trainer(yaml_config)
    
    accuracies, losses, all_y_true, all_y_pred = trainer.get_best_model(model, X, Y)
    mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval = funs.calculate_result(accuracies, losses)

    val_true_list = np.concatenate(all_y_true, axis=0)
    val_pred_list = np.concatenate(all_y_pred, axis=0)

    val_report = classification_report(val_true_list, val_pred_list, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    
    print(f"""
Validation Accuracy: {mean_accuracy:.4f}, Validation Loss: {mean_loss:.4f}
    """)


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


    ##############################
    # 5. save                    
    ##############################
    if save_log:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        funs.log_results(
            model_name          = "ANN",
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
