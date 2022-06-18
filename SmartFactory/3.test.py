from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
import keras
from efficientnet.keras import EfficientNetB0


# 모델 경로
best_model_path = r'./checkpointmodel/2021-09-22-0029/[샘플딥러닝모델은 이곳에 위치한다]_epoch-001-0.2646-0.9108.h5'

# test 데이터셋 경로
data_dir = r'./sample_dataset/casting_data_3way_splited'
test_dir = data_dir+'/test'

# config 값
batch_size = 8
resize_img = (300, 300)

# test 데이터셋 불러오기
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=resize_img,
    batch_size = batch_size,
    class_mode='binary',
    shuffle=False
)

# 모델 로드
model = load_model(best_model_path)
model.summary()

test_steps = test_generator.samples
test_generator.reset()

# 예측확률값
y_pred = model.predict_generator(test_generator,steps=test_steps//batch_size+1)

# 예측카테고리
y_pred = [1 * (x[0]>=0.5) for x in y_pred]

# 정답카테고리
y_test = test_generator.classes[test_generator.index_array]


print('===============================================================================================================')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('===============================================================================================================')


# print('===============================================================================================================')
# print('정답카테고리: ', y_test)
# print('예측카테고리: ', y_pred)
# print('===============================================================================================================')

