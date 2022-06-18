import smtplib
from email.mime.text import MIMEText

def sendMail(me, you, msg):
    """
    메일 보내는 함수
    :param me: 자신의 구글이메일 주소
    :param you: 보낼 메일 (관리자 이메일 계정). me와 똑같아도 됨
    :param msg: 메일의 내용
    :return: 없음.
    """
    smtp = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp.login(me, '.............me 의 비밀번호 적는 자리...............')                # 자신의 구글이메일 패스워드 입력
    msg = MIMEText(msg)
    msg['Subject'] = '[AI SmartFactory] Defect Product'    # 메일 제목
    smtp.sendmail(me, you, msg.as_string())
    smtp.quit()


# 테스트 코드 입니다.
# TODO: "보안 수준이 낮은 앱 허용"을 "사용안함" 에서 "사용함"으로 변경해주면 된다.
# TODO: https://myaccount.google.com/lesssecureapps
if __name__ == "__main__":
    sendMail('aicoding.class101@gmail.com', '받는메일주소', '메일보내기')
