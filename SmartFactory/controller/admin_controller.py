from flask import render_template, request, session, Blueprint
from models.model import Setting
from models.model import db

bp = Blueprint('admin',
               __name__,
               url_prefix='/admin')


# 요청 URL : POST http://127.0.0.1:8888/admin
@bp.route('/', methods=['POST', 'GET'])
def show_adminpage_by_login():
    print("어드민 컨트롤러!!! + 로그인 검사 까지")

    setting_obj_1 = Setting.query.filter(Setting.setting_name == "MAIL_USE_YN").first()
    setting_obj_2 = Setting.query.filter(Setting.setting_name == "MAIL_TH").first()
    print(f"mail_use_yn={setting_obj_1.setting_value}")
    print(f"mail_th={setting_obj_2.setting_value}")
    mail_use_yn = setting_obj_1.setting_value
    mail_th = setting_obj_2.setting_value

    # 로그인 되어 있으면
    if session.get('login') is True:
        print("로그인 되어 있음")
        print(f"로그인 유저={session.get('AUTH')}")
        print(f"로그인 상태={session.get('login')}")

        return render_template('admin.html', mail_use_yn=mail_use_yn, mail_th=mail_th)

    if request.method == 'POST':
        print("POST방식 요청")

        # 요청데이터 파싱 (폼/폼데이터 파싱)
        result = request.form
        password = result['fake_password']

        my_password = "1234"
        if not (my_password == password):
            print("관리자 비밀번호 일치 X")
            return render_template('frontend.html')
        else:
            print("관리자 비밀번호 일치 O")
            print(f"my_password={my_password}")
            print(f"password={password}")

            # 플라스크 세션에 로그인 정보 저장 (기억하기)
            session['login'] = True
            session['AUTH'] = "[ROLE_ADMIN]"

            return render_template('admin.html', mail_use_yn=mail_use_yn, mail_th=mail_th)
    else:
        print("GET방식 요청")

        # 로그인 되어 있으면
        if session.get('login') is True:
            print("로그인 되어 있음")
            print(f"로그인 유저={session.get('AUTH')}")
            print(f"로그인 상태={session.get('login')}")

            return render_template('admin.html', mail_use_yn=mail_use_yn, mail_th=mail_th)
        else:
            print("로그인 안되어 있음")
            return render_template('frontend.html')


# 요청 URL : POST http://127.0.0.1:8888/admin/logout
@bp.route('/logout')
def logout():
    session.pop('login', None)
    session.pop('AUTH', None)
    print("관리자 계정 로그아웃")

    return render_template('frontend.html')
