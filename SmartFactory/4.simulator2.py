from glob import glob
import requests
import time


'''
[[ 클라이언트 2. ]]
    폴더 내 이미지 파일 모두 불러와 AI예측시스템(웹서버) 에 전송
    @Config AI_PREDICTION_USE : 기존 검사가 아닌 AI검사 사용 여부 설정
            True  : AI검사 사용 O
            False : AI검사 사용 X 
    @Config CONVEYOR_BELT_CAMERA : 이미지를 읽어 올 폴더 위치
    @Config AI_SYSTEM_WEBSERVER_URL : AI예측시스템(웹서버) URL
            개발 시: http://127.0.0.1:8888/deeplearning/inference
            운영 시: http://AWS인스턴스IP주소:8888/deeplearning/inference
    @Config PREDICTION_FREQUENCY : 이미지 보내는 빈도 시간 (단위 : 초)
'''


AI_PREDICTION_USE = True
CONVEYOR_BELT_CAMERA_OUTPUT_DIRECTORY = './camera_output'
AI_SYSTEM_WEBSERVER_URL = 'http://127.0.0.1:8888/deeplearning/inference'  # AI예측시스템(웹서버 주소)
PREDICTION_FREQUENCY = 3


if AI_PREDICTION_USE:
    for file in glob(CONVEYOR_BELT_CAMERA_OUTPUT_DIRECTORY+'/*'):
        # 파일이름과 파일확장자 분리
        original_file_name_only = ".".join(file.split('.')[:-1])
        original_file_extension_only = '.' + file.split('.')[-1]

        print(f"* 파일명: {original_file_name_only}")
        print(f"* 확장자: {original_file_extension_only}")

        # 이미지 파일 확장자 유효성 체크
        if original_file_extension_only.lower() in ['.jpg', '.bmp', '.png', '.jpeg']:
            files = {'file': open(file, 'rb')}                              # 이미지 파일 열어 딕셔너리에 넣음

            response = requests.post(AI_SYSTEM_WEBSERVER_URL, files=files)  # 이미지 파일 전송 (HTTP request 에 담아서 보냄)

            # AI 예측 결과 (AI-시스템=AI-API 에서 응답해준 딥러닝 예측 결과)
            print(f"* 딥러닝 예측 결과(AI 시스템 API 결과) = {response.text}")
            print(f"* 딥러닝 예측 요청 상태 코드(AI 시스템 API HTTP 통신 결과) = {response.status_code}")
            print("\n")

            print("Sleep 5 seconds from now on...")
            time.sleep(PREDICTION_FREQUENCY)                                                       # 대기 시간
            print("wake up!")

            print("\n")
            print("===============================================================================" * 2)
        else:
            print(f"[Error]")
            print(file, '은 이미지 파일이 아닙니다.')
            print("\n")
            print("===============================================================================" * 2)

            print("Sleep 5 seconds from now on...")
            time.sleep(PREDICTION_FREQUENCY)                                                       # 대기 시간
            print("wake up!")

print("\n\n\n-----------------------------------\n\n  \t\t 시뮬레이터2 종료  \n\n\n-----------------------------------")