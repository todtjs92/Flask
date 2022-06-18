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

import efficientnet.tfkeras  # efficientnet 로드하기위해 필요

from models.model import Image, Result
from models.model import db

bp = Blueprint('history',
               __name__,
               url_prefix='/api/history')

# 요청 URL : GET http://127.0.0.1:8888/history/
@bp.route('/', methods=['POST'])
def get_history():
    """
    딥러닝 결과 이력 조회
    :return: 딥러닝결과이력
    """
    print("딥러닝 결과 컨트롤러!!!")


    ##------------------------
    ##  응답 만들기
    ##------------------------


    # 비지니스로직
    image_list = Image.query.all()
    result_list = Result.query.all()

    convert_row_list = []
    for i, image in enumerate(image_list):
        # json 구조
        inference_info = {
            "ID": image.id,
            "Origin": {
                "OriginImgPath": image.saved_original_image_file_path,
                "OriginImgName": image.saved_original_image_file_name
            },
            "GradCam": {
                "GradCAMImgPath": image.saved_gradcam_image_file_path,
                "GradCAMImgName": image.saved_gradcam_image_file_name
            },
            "Inference": {
                "Pred": result_list[i].prob,
                "InferenceResult": str(result_list[i].pred_class)
            },
            "RequestTime": image.created_date
        }
        convert_row_list.append(inference_info)

    # 응답만들기 -> 데이터만 전송. JSON 객체. ajax 요청의 dataType: 'json' 로 변경해야함
    response = json.dumps([convert_row_list], default=str, ensure_ascii=False)

    return response

