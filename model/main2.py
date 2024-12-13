import os
import tensorflow as tf
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from funs.databuilder import make_dataframe, augment_dataframe, add_statistics, get_data_label
from funs.utils import load_yaml, get_dir_list

import tensorflow_decision_forests as tfdf

from sklearn.model_selection import train_test_split


# def convert_tree_to_tf_model(tree, input_dim):
#     def tree_predict(inputs):
#         # 입력 형태 확인 및 변환
#         inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

#         # 트리의 노드 정보 추출
#         children_left = tree.tree_.children_left
#         children_right = tree.tree_.children_right
#         feature = tree.tree_.feature
#         threshold = tree.tree_.threshold
#         value = tree.tree_.value

#         # 현재 노드에서 재귀적으로 예측하는 내부 함수
#         def predict_sample(node=0):
#             # 리프 노드인 경우
#             if children_left[node] == -1:
#                 # 클래스 확률 반환
#                 return tf.argmax(value[node][0])

#             # 분기 노드인 경우
#             sample_feature = inputs[feature[node]]
#             if sample_feature < threshold[node]:
#                 return predict_sample(children_left[node])
#             else:
#                 return predict_sample(children_right[node])

#         # 배치 전체에 대해 예측
#         return tf.map_fn(lambda x: predict_sample(), inputs, dtype=tf.int64)

#     # Keras 모델로 변환
#     inputs = tf.keras.Input(shape=(input_dim,))
#     outputs = tree_predict(inputs)
#     return tf.keras.Model(inputs=inputs, outputs=outputs)

# 기존 코드와 동일
target_config = {
    'date': '1105',  # 필수
    'input_feature': 'rms'  # 필수. 모델 input feature로 사용할 데이터
}
yaml_config = load_yaml('./model_config.yaml')

# 1. 데이터 준비 및 모델 학습 준비
directory = os.path.join(yaml_config.output_dir, target_config['date'])
directory = get_dir_list(directory)

# 데이터프레임 제작
df = make_dataframe(yaml_config, directory)

# 데이터 증강
augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)

# 통계값 값 추가
statistics_df = add_statistics(augmented_df, 'z')
feature_list = list(statistics_df.columns)[2:]
feature_list.remove('average')

# 데이터, 라벨 얻기
X, Y = get_data_label(statistics_df, target_config['input_feature'])
print(f'Input feature: {target_config["input_feature"]}')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def to_tf_dataset(X, Y):
    return tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)

train_ds = to_tf_dataset(X_train, y_train)
test_ds = to_tf_dataset(X_test, y_test)

# 2. 10-fold 교차검증
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X_train):
    fold_X_train, fold_X_val = X_train[train_idx], X_train[val_idx]
    fold_y_train, fold_y_val = y_train[train_idx], y_train[val_idx]
    
    fold_train_ds = to_tf_dataset(fold_X_train, fold_y_train)
    fold_val_ds = to_tf_dataset(fold_X_val, fold_y_val)

    model = tfdf.keras.RandomForestModel()
    model.fit(fold_train_ds)

    score = model.evaluate(fold_val_ds, verbose=0)
    scores.append(score)

print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {sum(scores)/len(scores)}")

# 최종 모델 학습
final_model = tfdf.keras.RandomForestModel()
final_model.fit(train_ds)

# 최종 평가
final_model.evaluate(test_ds)

# 3. TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

# TFLite 모델 저장
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite 모델이 성공적으로 저장되었습니다.")
