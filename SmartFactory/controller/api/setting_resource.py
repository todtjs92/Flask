
from flask import render_template, Blueprint, request

from models.model import Setting
from models.model import db

bp = Blueprint('setting',
               __name__,
               url_prefix='/api/setting')


# 요청 URL : POST http://127.0.0.1:8888/api/setting/update
@bp.route('/update', methods=['POST'])
def setting_update():
    print("관리자 세팅값 업데이트 컨트롤러 !!!")

    # 요청데이터 파싱 (json객체로 보내 줬을 때)
    client_data_json = request.get_json(silent=True)
    print(f"client_data_json => {client_data_json}")
    print(f"client_data_json['mail_use_yn']={client_data_json['update_mail_use_yn']}")
    print(f"client_data_json['mail_th']={client_data_json['update_mail_th']}")

    # 비지니스로직
    try:
        ##------------------------
        ##  <<Setting>> 첫번째 행 수정
        ##------------------------
        update_setting_1 = Setting.query.filter(Setting.setting_name == "MAIL_USE_YN").first()
        update_setting_1.setting_value = client_data_json['update_mail_use_yn']
        db.session.commit()

        ##------------------------
        ##  <<Setting>> 두번째 행 수정
        ##------------------------
        update_setting_2 = Setting.query.filter(Setting.setting_name == "MAIL_TH").first()
        update_setting_2.setting_value = client_data_json['update_mail_th']
        db.session.commit()

        # 응답만들기 -> 데이터만 전송. TEXT 객체. ajax 요청의 dataType: 'text' 로 변경해야함
        response = "SUCCESS"
    except Exception as e:
        print(f"e={e}")

        # 응답만들기 -> 데이터만 전송. TEXT 객체. ajax 요청의 dataType: 'text' 로 변경해야함
        response = "FAIL"

    return response
