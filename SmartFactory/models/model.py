# '''
# Image 먼저 저장되고 나중에 Result 저장 됨.
# 이미지와 딥러닝결과는 1:1 관계 (One to One 관계)
# 문법 : db.relationship('관계를 형성할 테이블명',backref='백 레퍼런스로 사용할 이름') -> 내부테이블로 자동 관리 된다. (눈에 보이지 X)
# '''


from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# 전역변수로 선언한 이유 -> 다른 파일(모듈)에서 불러다가 사용 하기 위해
db = SQLAlchemy()


# 테이블 객체 정의
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    saved_original_image_file_name = db.Column(db.String(256))
    saved_original_image_file_path = db.Column(db.String(256))
    saved_gradcam_image_file_name = db.Column(db.String(256))
    saved_gradcam_image_file_path = db.Column(db.String(256))

    created_date = db.Column(db.DateTime(), nullable=False)

    result = db.relationship('Result',
                             backref='result', # 백 레퍼런스, Image.result 로 대응 된 result의 row 데이터 확인 가능
                             uselist=False)    # 1 대 1 관계 매핑

    def __init__(self, saved_original_image_file_name, saved_original_image_file_path, saved_gradcam_image_file_name, saved_gradcam_image_file_path, created_date):
        self.saved_image_file_name = saved_original_image_file_name
        self.saved_image_file_path = saved_original_image_file_path
        self.saved_gradcam_image_file_name = saved_gradcam_image_file_name
        self.saved_gradcam_image_file_path = saved_gradcam_image_file_path
        self.created_date = created_date


    @property
    def serialize(self):
        return {
            'id': self.id,
            'saved_original_image_file_name': self.saved_original_image_file_name,
            'saved_original_image_file_path': self.saved_original_image_file_path,
            'saved_gradcam_image_file_name': self.saved_gradcam_image_file_name,
            'saved_gradcam_image_file_path': self.saved_gradcam_image_file_path,
            'created_date': self.created_date
        }


# 테이블 객체 정의 : Image 테이블의 id값(File_id)을 갖고있겠다
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prob = db.Column(db.String(50))
    pred_class = db.Column(db.String(50))
    created_date = db.Column(db.DateTime(), nullable=False)

    image_id = db.Column(db.Integer,
                     db.ForeignKey(Image.id, ondelete='CASCADE'))  # 자동 삭제 연동 설정 적용

    def __init__(self, prob, pred_class, created_date, image_id):
        self.prob = prob
        self.pred_class = pred_class
        self.created_date = created_date
        self.image_id = image_id

    @property
    def serialize(self):
        return {
            'id': self.id,
            'prob': self.prob,
            'pred_class': self.pred_class,
            'created_date': self.created_date,
            'image_id': self.image_id
        }


# 테이블 객체 정의 : 관리자 설정
class Setting(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    setting_name = db.Column(db.String(50))
    setting_value = db.Column(db.String(50))
    created_date = db.Column(db.DateTime(), nullable=False)

    def __init__(self, setting_name, setting_value, created_date):
        self.setting_name = setting_name
        self.setting_value = setting_value
        self.created_date = created_date

    @property
    def serialize(self):
        return {
            'id': self.id,
            'setting_name': self.setting_name,
            'setting_value': self.setting_value,
            'created_date': self.created_date
        }
