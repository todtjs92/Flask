from keras.models import Model
from keras.applications import densenet
import datetime
import os
import time
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import optimizers


# 딥러닝 학습 설정
my_lr = 0.0001  # 나만의 학습률
my_epochs = 20  # 학습 반복 횟수(에폭)

# train, val 데이터셋 경로
data_dir = r'./sample_dataset/casting_data_3way_splited'
train_dir = data_dir +'/train'
val_dir = data_dir +'/val'

# 현재 시각
now = time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))

# 이미지 1/255로 스케일 조정 (-> 픽셀값 노말라이즈) & 학습데이터 증강
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   )
val_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 조금씩(batch_size) 읽는놈 만든다.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=16,
    class_mode='binary')
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(300, 300),
    batch_size=16,
    class_mode='binary')

# 에폭 당 반복학습 계산
steps_per_epoch = train_generator.samples
validation_steps = val_generator.samples


# EfficientNet 사용
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from efficientnet.keras import EfficientNetB0
base_model = EfficientNetB0(input_shape=(300, 300, 3), weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid', name='sigmoid')(x)
model = Model(inputs=[base_model.input], outputs=[output])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DenseNet121 모델 사용
# input = keras.Input(shape=(300,300,3))
# model = densenet.DenseNet121(include_top=False,weights='imagenet',input_tensor = input)
# x = model.output
# x = Flatten()(x)
# x = Dense(1, activation='sigmoid',name='sigmoid')(x)
# model = Model(model.input,x)
# model.summary()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# [참고] Adam default lr = 0.001
# learning rate 조절 원한다면 아래 처럼 상세하게 객체 생성 한 뒤 -> 위 model.compile(optimizer=opt, ...) 라고 넣어준다.
# opt = optimizers.Adam(lr=my_lr)  # optimizer Adam,  learning rate : my_lr

# 모델 저장 폴더
if not os.path.exists('checkpointmodel'):
    os.mkdir('checkpointmodel')

# checkpoint model 돌릴때마다 현재 날짜 시간기준으로 폴더 생성후 그 하위에 체크포인트모델저장
if not os.path.exists('./checkpointmodel/' + now):
    os.mkdir('./checkpointmodel/' + now)

model_checkpoint = ModelCheckpoint(
    filepath='./checkpointmodel/' + now + '/_epoch-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',   # 모니터 기준 설정 (val loss)
                              mode='min',
                              patience=4,           # 4회 학습하는동안 개선되지 않는다면 종료
                              )

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


trained_model = model.fit_generator(train_generator,
                                    epochs=my_epochs,
                                    verbose=1,
                                    callbacks=[model_checkpoint,earlystopping],
                                    validation_data=val_generator
                                    )
model.save(f'best_model.h5')