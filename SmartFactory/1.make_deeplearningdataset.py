import os, shutil
import random

# 원본셋 경로
BEFORE_DATASET_DIR = r'./sample_dataset/casting_data_1way_labeled'

# 학습셋 경로
AFTER_DATASET_DIR = r'./sample_dataset/casting_data_3way_splited'

# 학습셋(train/val/test) 비율 -> 6:2:2 -> 합쳐서 무조건 10이 되야함
train_ratio = 6
val_ratio = 2
test_ratio = 2

# 읽어 온 카테고리
class_names = os.listdir(BEFORE_DATASET_DIR)
print(class_names)

# 기존에 만들어진 학습셋 있는지 확인
if os.path.exists(AFTER_DATASET_DIR):  # AFTER_DATASET_DIR 이 존재하면
    shutil.rmtree(AFTER_DATASET_DIR)   # 경로 하위의 폴더, 파일 삭제

# 새롭게 학습셋 루트 경로 만들기
os.mkdir(AFTER_DATASET_DIR)

# 학습셋 하위에 train, val, test 폴더 각각 생성
train_dir = os.path.join(AFTER_DATASET_DIR, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(AFTER_DATASET_DIR, 'val')
os.mkdir(val_dir)
test_dir = os.path.join(AFTER_DATASET_DIR, 'test')
os.mkdir(test_dir)

# 원본셋의 카테고리 수만큼 순차적으로 반복
for class_name in class_names:
    # train/카테고리1, train/카테고리2 폴더 생성
    train_class_dir = os.path.join(train_dir, class_name)
    os.mkdir(train_class_dir)
    # val/카테고리1, val/카테고리2 폴더 생성
    val_class_dir = os.path.join(val_dir,class_name)
    os.mkdir(val_class_dir)
    # test/카테고리1, test/카테고리2 폴더 생성
    test_class_dir = os.path.join(test_dir,class_name)
    os.mkdir(test_class_dir)
    
    # 원본셋의 특정 카테고리 전체 이미지 읽기  e.g. casting_data_1way_labeled\0_nondefect\~
    class_name_dir = os.path.join(BEFORE_DATASET_DIR, class_name)
    
    # 이미지파일 리스트
    fnames = os.listdir(class_name_dir)
    
    # 이미지파일 섞기 (랜덤성을 위해)
    random.shuffle(fnames)

    # 학습셋 분리 비율 (주의 : 합하면 무조건 10이어야 함)
    total_ratio = train_ratio + val_ratio + test_ratio

    if test_ratio:
        train_cnt = train_ratio * len(fnames) // total_ratio
        val_cnt = val_ratio * len(fnames) // total_ratio
        test_cnt = len(fnames) - train_cnt - val_cnt
    else:
        train_cnt = train_ratio * len(fnames) // total_ratio
        val_cnt = len(fnames) - train_cnt
        test_cnt = 0
    
    # 원본셋(랜덤성)의 파일 1개씩 읽어서 학습셋 구축
    for i, fname in enumerate(fnames):
        if i < train_cnt:
            src = os.path.join(class_name_dir, fname)
            dst = os.path.join(train_class_dir, fname)
        elif i < val_cnt+train_cnt:
            src = os.path.join(class_name_dir, fname)
            dst = os.path.join(val_class_dir,  fname)
        else:
            src = os.path.join(class_name_dir, fname)
            dst = os.path.join(test_class_dir, fname)
        
        # 원본셋이미지경로(src) 에서 학습셋이미지경로(dst)로 이미지 파일 이동
        shutil.copy2(src, dst)

    print(f'{class_name} finished')
