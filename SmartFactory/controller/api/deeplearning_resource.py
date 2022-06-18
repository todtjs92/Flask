import os
import json
import datetime

import numpy as np
import cv2

from flask import render_template, Blueprint, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import backend as K

import efficientnet.tfkeras  # efficientnet 로드하기위해 필요 -> 삭제 하면 안됨.

from models.model import Image, Result, Setting
from models.model import db

from common.mail import sendMail

bp = Blueprint('deeplearning',
               __name__,
               url_prefix='/deeplearning')

# 파이썬 글로벌 변수 설정 (어떤 함수에서도 사용 할 수 있게)
deeplearnig_model_instance = None


# 요청 URL : GET http://127.0.0.1:8888/deeplearning/modelstatus
@bp.route('/modelstatus', methods=['GET'])
def check_status():
    """
    딥러닝 모델 상태 체크
    :return: 딥러닝 모델 상태 JSON 객체. DOWN=로드안되있음, ALIVE=로드되어있음
    """
    print("딥러닝 모델 상태 체크 컨트롤러!!!")

    # 파이썬 글로벌 변수 설정 (어떤 함수에서도 사용 할 수 있게)
    global deeplearnig_model_instance

    print(f"현재 상태 = {deeplearnig_model_instance}")

    #  모델 로드 여부 확인
    if deeplearnig_model_instance == None:
        print(f"deeplearnig_model_instance={deeplearnig_model_instance}")
        return jsonify(status='DOWN')
    else:
        print(f"deeplearnig_model_instance={deeplearnig_model_instance}")
        return jsonify(status='ALIVE')


# 요청 URL : POST http://127.0.0.1:8888/deeplearning/modelload
@bp.route('/modelload', methods=['POST'])
def deeplearning_model_load():
    """
    딥러닝 모델 메로리에 로드
    :return: 딥러닝 모델 로드 결과 JSON객체
    """
    print("modelload 컨트롤러!!!")

    # 파이썬 글로벌 변수 설정 (어떤 함수에서도 사용 할 수 있게)
    global deeplearnig_model_instance
    global tf_sess
    global graph
    global deep_learning_config

    print(f"[before] 딥러닝 모델 인스턴스 = {deeplearnig_model_instance}")

    # 딥러닝 모델 설정 파일 읽어오기
    with open(r'common/deep_learning_config.json', 'r', encoding='utf-8') as config_file:
        deep_learning_config = json.load(config_file)

    # 이미 로드 된 모델 있는지 확인
    if deeplearnig_model_instance != None:
        return jsonify(result='Model is already loaded')

    # 현재 텐서플로의 세션을 get
    tf_sess = tf.keras.backend.get_session()
    # 현재 텐서플로 세션의 그래프를 저장
    graph = tf_sess.graph

    with graph.as_default():  # 저장한 그래프와 연결
        print(f'로드 할 딥러닝 모델 파일 이름 = {deep_learning_config["model_file_name"]}')

        # 딥러닝 모델 로드
        deeplearnig_model_instance = load_model(f'static/kerasmodel/{deep_learning_config["model_file_name"]}',
                                                compile=False)  # 모델을 로드하여 연결된 그래프에서 불러 저장

        # 딥러닝 모델 구조 출력
        deeplearnig_model_instance.summary()

    print(f"[after] 딥러닝 모델 인스턴스 = {deeplearnig_model_instance}")

    return jsonify(result='MODEL_LOAD_SUCCESS')


# 요청 URL : POST http://127.0.0.1:8888/deeplearning/modelkill
@bp.route('/modelkill', methods=['POST'])
def deeplearning_model_kill():
    """
    딥러닝 모델 중지
    :return: 딥러닝 모델 중지 결과 JSON객체
    """
    print("modelkill 컨트롤러!")

    global deeplearnig_model_instance

    # kill 할 객체
    print(f"Kill 할 모델 객체 = {deeplearnig_model_instance}")

    # kill
    deeplearnig_model_instance = None

    print(f"Kill 한 모델 객체 상태 = {deeplearnig_model_instance}")

    return jsonify(result='MODEL_KILL_SUCCESS')


# 요청 URL : POST http://127.0.0.1:8888/deeplearning/inference
@bp.route('/inference', methods=['POST'])
def inference_one_image():
    """
    이미지 받기, 딥러닝 모델 로드, 그래드캠 저장, 인퍼런스 결과 csv화
    :return: 딥러닝 결과 JSON 객체
    """
    if deeplearnig_model_instance == None:
        return "Check Deeplearning Model"

    # 현재 날짜 구하기 : 이미지 저장 경로 / DB Row 생성날짜 위함
    now = datetime.datetime.now()
    today_detail = now.strftime('%Y-%m-%d %H:%M:%S')
    today_simple = now.strftime('%Y-%m-%d')

    # 클라이언트 확인
    auth = "simulator-001-by-form"
    if request.headers['User-Agent'][:15] == 'python-requests':
        auth = "simulator-002-by-program"
        print(f"클라이언트 종류 : 원격프로그램")
    else:
        auth = "simulator-001-by-form"
        print(f"클라이언트 종류 : 웹페이지")

    ##------------------------
    ##  <<Image>> 행 생성
    ##------------------------
    new_image = Image(saved_original_image_file_name="fake",
                      saved_original_image_file_path="fake",
                      saved_gradcam_image_file_name="fake",
                      saved_gradcam_image_file_path="fake",
                      created_date=now)
    db.session.add(new_image)
    db.session.commit()

    ##------------------------
    ##  <<Image>> 고유번호 채번
    ##------------------------

    inference_id = new_image.id

    ##------------------------
    ##  이미지 저장 경로 생성여부 확인 및 생성
    ##------------------------
    if not os.path.exists(f'static'):
        os.mkdir(f'static')
    if not os.path.exists(f'static/clientdata'):
        os.mkdir(f'static/clientdata')
    if not os.path.exists(f'static/clientdata/{today_simple}'):
        os.mkdir(f'static/clientdata/{today_simple}')
    if not os.path.exists(f'static/clientdata/{today_simple}/{inference_id}'):
        os.mkdir(f'static/clientdata/{today_simple}/{inference_id}')
        os.mkdir(f'static/clientdata/{today_simple}/{inference_id}/img')
        os.mkdir(f'static/clientdata/{today_simple}/{inference_id}/img/original')
        os.mkdir(f'static/clientdata/{today_simple}/{inference_id}/img/gradcam')

    ##------------------------
    ##  이미지 저장 경로 (원본, 히트맵=그래드캠)
    ##------------------------
    original_path = f'static/clientdata/{today_simple}/{inference_id}/img/original'
    gradcam_path = f'static/clientdata/{today_simple}/{inference_id}/img/gradcam'

    ##------------------------
    ##  요청이미지(원본) 저장
    ##------------------------

    # 폼에 담겨 온 요청이미지(원본) get
    original_file = request.files.get('file')
    original_file_name = original_file.filename

    # 파일이름과 파일확장자 분리
    original_file_name_only = ".".join(original_file_name.split('.')[:-1])
    original_file_extension_only = '.' + original_file_name.split('.')[-1]
    print(f"파일명: {original_file_name_only}")
    print(f"확장자: {original_file_extension_only}")

    # 요청이미지(원본) 저장
    original_file.save(original_path + '/' + original_file_name)

    ##------------------------
    ##  <<Image>> 행 수정
    ##------------------------
    update_image = Image.query.get(inference_id)
    update_image.saved_original_image_file_name = original_file_name
    update_image.saved_original_image_file_path = original_path
    db.session.commit()

    ##------------------------
    ##  DeepLearning 인퍼런스 위한 이미지 준비
    ##------------------------

    # 리사이즈 값 - 딥러닝 모델 학습 시 설정한 딥러닝모델입력부분 사이즈 -> input_shape=(300, 300, 3) 와 동일 해야 함
    RESIZE_W = 300
    RESIZE_H = 300

    # 이미지 경로
    original_directory = original_path + '/' + original_file_name

    # 이미지 로드 및 리사이즈 (Pillow 라이브러리)
    img = image.load_img(original_directory, target_size=(RESIZE_H, RESIZE_W))

    # 디버깅
    img_w, img_h = img.size
    print(img_h, img_w)

    # array -> nparray
    img_tensor = image.img_to_array(img)
    # 딥러닝배치 차원 추가
    img_tensor_4d = np.expand_dims(img_tensor, axis=0)
    # 픽셀값 노말라이즈 (0~255 -> 0~1)
    img_tensor_4d /= 255.

    ##------------------------
    ##  그래드캠 추출 작업 준비
    ##------------------------

    # 그래드캠 추출 전용 이미지 로드
    img_cv2 = cv2.imread(original_directory)

    # 그래드캠 저장 경로
    gradcam_file_name = original_file_name_only + '_gradcam.' + original_file_extension_only
    gradcam_directory = gradcam_path + '/' + gradcam_file_name

    ##------------------------
    ##  DeepLearning
    ##------------------------
    global graph
    # 모델을 로드한 세션, 그래프와 연결
    with tf_sess.as_default(), graph.as_default():
        ##------------------------
        ##  인퍼런스(예측)
        ##------------------------
        pred = deeplearnig_model_instance.predict(img_tensor_4d)[0]

        ##------------------------
        ##  그래드캠 추출 - 외우는 것 아님 (오픈소스 가져다가 사용 할 뿐)
        ##------------------------
        # 딥러닝 모델마다 레이어 이름(-> layer_name)이 다름 ([참고] EfficientNet summary() 함수 출력값 확인하여 변경 해보기)
        layer_name = 'top_conv'  # 혹은 top_activation

        defect_output = np.argmax(deeplearnig_model_instance.predict(img_tensor_4d)[0])
        y_c = deeplearnig_model_instance.output[0, defect_output]
        last_conv_layer = deeplearnig_model_instance.get_layer(layer_name).output
        grads = K.gradients(y_c, last_conv_layer)[0]
        gradient_function = K.function([deeplearnig_model_instance.input], [last_conv_layer, grads])
        output, grads_val = gradient_function([img_tensor_4d])
        output, grads_val = output[0, :], grads_val[0, :, :, :]
        weights = np.mean(grads_val, axis=(0, 1))
        heatmap = np.dot(output, weights)
        heatmap = np.maximum(heatmap, 0)  # ReLU 활성화함수 통과
        heatmap /= np.max(heatmap)  # 0 ~ 1.0로 리스케일
        heatmap = cv2.resize(heatmap, (img_cv2.shape[1], img_cv2.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        ##------------------------
        ##  그래드캠 저장
        ##------------------------
        overlay_heatmap = heatmap * 0.95 + img_cv2
        cv2.imwrite(gradcam_directory, overlay_heatmap)
        print(gradcam_directory, '저장완료')

        ##------------------------
        ##  <<Image>> 행 수정
        ##------------------------
        update_image = Image.query.get(inference_id)
        update_image.saved_gradcam_image_file_name = gradcam_file_name
        update_image.saved_gradcam_image_file_path = gradcam_path
        db.session.commit()

    ##------------------------
    ##  Threshold cutting
    ##------------------------
    if deep_learning_config['task_type'] == 'binary':
        # 딥러닝 예측확률값(0~1 사이 값) 소수 둘째 자리 표현 후 문자열 변경 (db 컬럼을 String으로 했으므로)
        prob = str(round(pred[0], 4))

        # 디버깅
        print(f"prob={prob}")
        print(f"pred[0]={pred[0]}")
        print(f"pred={pred}")

        # ※ ML/DL 에서는 관례적으로 1이 찾고자 하는 카테고리를 의미함
        # ex) 양불 판정 Task에서는 불량을 찾아내므로
        #     양품이미지는 0이라는 클래스(카테고리, 레이블) 로 학습/평가/추론 하며
        #     불량품이미지는 1이라는 클래스(카테고리, 레이블) 로 학습/평가/추론 하는 식으로 명명/관리 한다.
        if pred[0] >= float(deep_learning_config['JUDGE_TH']):
            # 예측 카테고리를 1번 카테고리라고 하겠다
            inference_result = 1
        else:
            # 예측 카테고리를 0번 카테고리라고 하겠다
            inference_result = 0

        ##------------------------
        ##  <<Result>> 행 생성
        ##------------------------
        new_result = Result(prob=prob, pred_class=inference_result, created_date=now, image_id=inference_id)
        db.session.add(new_result)
        db.session.commit()
        
        
        ##------------------------
        ##  <<Setting>> 메일 알람 사용 여부 설정 가져오기
        ##------------------------
        update_setting_1 = Setting.query.filter(Setting.setting_name == "MAIL_USE_YN").first()
        if update_setting_1.setting_value == "True":
            # 양품 불량품
            # 0 ~ 1
            # > 0.5 -> 불량 카테고리로 예측
            #     > 0.9 그때 메일을 보내 겠다
            #
            # < 0.5 -> 정상 카테고리로 예측
            ##------------------------
            ##  <<Setting>> 메일 알람 기준값 설정 가져오기
            ##------------------------
            update_setting_2 = Setting.query.filter(Setting.setting_name == "MAIL_TH").first()
            db_mail_th = float(update_setting_2.setting_value)

            # TODO: "보안 수준이 낮은 앱 허용"을 "사용안함" 에서 "사용함"으로 변경해주면 된다. -> https://myaccount.google.com/lesssecureapps
            # TODO: "캡챠 해제" -> https://accounts.google.com/b/0/DisplayUnlockCaptcha
            if pred[0] >= db_mail_th:
                print("메일보내는 조건 해당 !")
                print(f"메일보내는 기준값(db)={db_mail_th}")

                # 메일내용
                content = "AI Model은 " + str(new_result.id)+"번 이미지에 대해서 Category-" + str(inference_result) + " 라고 예측 했습니다"

                # 보내는메일주소, 받는메일주소, 메일내용
                sendMail('aicoding.class101@gmail.com', 'aicoding.class101@gmail.com', content)

                print("메일전송완료 !")

    ##------------------------
    ##  응답 만들기
    ##------------------------
    # json 구조
    inference_info = {
        "ID": inference_id,
        "Origin": {
            "OriginImgPath": original_path,
            "OriginImgName": original_file_name
        },
        "GradCam": {
            "GradCAMImgPath": gradcam_path,
            "GradCAMImgName": gradcam_file_name
        },
        "Inference": {
            "Pred": prob,
            "InferenceResult": str(inference_result)
        },
        "Auth": auth,
        "RequestTime": today_detail
    }

    # 디버깅
    print(inference_info)

    # 응답만들기 -> 데이터만 전송. JSON구조를 파이썬 리스트로 감싸줘서 JSON 객체로 변경. ajax 요청 부분 dataType: 'json' 로 변경해야함
    response = json.dumps([inference_info], default=str, ensure_ascii=False)

    return response
