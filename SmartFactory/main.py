import os
import datetime

from flask import Flask
from models import model
from common import config
from models.model import db
from models.model import Setting


# 플라스크에서 플라스크객체 생성 권장하는 방식
# [주의] 함수 명 바꾸면 안됨
def create_app():
    # 플라스크 인스턴스 생성
    app = Flask(__name__)

    # 컨트롤러 임포트
    from controller import frontend_controller
    from controller import admin_controller

    # 컨트롤러(api) 임포트
    from controller.api import deeplearning_resource
    from controller.api import history_resource
    from controller.api import setting_resource

    # 블루프린트 등록
    app.register_blueprint(frontend_controller.bp)
    app.register_blueprint(admin_controller.bp)

    # 블루프린트(api) 등록
    app.register_blueprint(deeplearning_resource.bp)
    app.register_blueprint(history_resource.bp)
    app.register_blueprint(setting_resource.bp)

    return app


# 데이터베이스 연동 및 테이블 생성
def set_database(app):
    # 플라스크앱에 등록 (시스템 공통 설정 파일 - DB 설정 등 들어있음)
    app.config.from_object(config)

    # ORM(sqlalchemy) 객체가 담긴 변수(전역변수) 가져오기
    db = model.db       # models디렉토리의 model 파이썬 파일에서 db 변수 가져오기

    # DB 생성
    db.init_app(app)    # sqlalchemy를 app에 적용

    # DB 와 플라스크앱 연동
    db.app = app        # sqlalchemy의 엔진 연결 정보, 테이블 정보 불러옴

    # DB에 Model 기반으로 테이블 생성
    db.create_all()     # DB 없으면 새로 생성

    return app


def init_database():
    ##------------------------
    ##  <<Setting>> 행 2개 생성
    ##------------------------

    # 값 있으면
    if Setting.query.all():
        pass
    else:

        # 현재 날짜 구하기
        now = datetime.datetime.now()

        ##------------------------
        ##  메일 알람 사용 여부 초기값 지정
        ##------------------------
        new_setting_1 = Setting(setting_name='MAIL_USE_YN', setting_value='False', created_date=now)
        db.session.add(new_setting_1)
        db.session.commit()

        ##------------------------
        ##  메일 알람 기준값 초기값 지정
        ##------------------------
        new_setting_2 = Setting(setting_name='MAIL_TH', setting_value='0.5', created_date=now)
        db.session.add(new_setting_2)
        db.session.commit()


if __name__ == '__main__':
    ### 플라스크앱(객체) 생성
    app = create_app()

    app.app_context().push()
    app.secret_key = os.urandom(24)

    ### DB 세팅 된 플라스크앱(객체)
    app = set_database(app)
    
    ### DB에 기본값 넣기
    init_database()

    ### 플라스크앱(객체) 실행
    app.run(host='0.0.0.0', port=8888, debug=False)  # debug=True : 소스코드를 변경 자동으로 감지 Flask서버 재시작
