from flask import render_template, Blueprint


bp = Blueprint('frontend',
               __name__,
               url_prefix='/')

# 요청 URL : http://127.0.0.1:8888/
@bp.route('/')
def main():
    print("인덱스 컨트롤러!!!")

    return render_template('frontend.html')  # 이로써 뷰단(V)과 컨트롤러(C) 분리 -> MVC 패턴
