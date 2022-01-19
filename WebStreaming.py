"""
* webcam을 통해 내부 ip 등으로 스트리밍하고, 이를 mlops의 과정에 포함시켜 데이터 수집 및 추론을 가능하게 하는 것이 목적
* 이 코드는 내부 ip 등으로 웹캠 영상을 스트리밍하고, 상황에 따라 capture 페이지에 GET 명령을 보내, 로컬 영역 안의 capture.jpg 파일을
  바꿔가며 추가적인 메모리의 할당 및 사용없이 작업하도록 구성 - 추후 백앤드 영역에서 연결해 사용할 때에도 메모리 할당의 최적화를 목적으로 함

/ : default page
/streaming : video를 출력
/capture : video 해당 장면을 capture해서 보여주고, 해당 이미지를 현재 작업 폴더의 capture.jpg로 저장
/cat : capture의 테스트 페이지로 현재 폴더의 cat.jpg를 출력해서 보여줌

- 각 페이지는 templates 안의 html 파일과 연동되어 있음
"""

from flask import Flask, render_template, Response
import cv2, time, os
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# 웹캠 스트리밍
def generate_frames() :
    cnt = 0

    while True :            
        ## read the camera frame
        success, frame = camera.read()
        if not success :
            break
        else :
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cnt < 1 :
                print("return value : ", ret)
                print("buffer :", buffer)
                # print("frame :", frame)  
                cnt += 1
            # time.sleep(1)


# 웹캠의 특정 장면을 잘라서 보여주고, 해당 이미지를 capture.jpg로 저장
def generate_image() :  

    while True :
            
        ## read the camera frame
        success, frame = camera.read()
        if not success :
            continue
        else :
            ret, buffer = cv2.imencode('.jpg', frame)

            encoded_img = np.frombuffer(buffer, dtype = np.uint8)
            img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)            
            cv2.imwrite("capture.jpg", img)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            break


@app.route('/') #root page
def index() :
    return render_template('index.html')

@app.route('/cat')
def test_cat() :
    return render_template('cat.html', image_file='./cat.jpg')

@app.route('/stream')
def stream() :
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture() :    
    return Response(generate_image(), mimetype='multipart/x-mixed-replace; boundary=frame'), render_template('capture.html', image_file='./capture.jpg')


if __name__=="__main__" :
    """
    app.run() - local에서만 작동
    app.run(ip 번호) - 내부 로컬 ip에서만
    app.run(host='0.0.0.0') - 어떤 호스트에서도 연결 가능하도록, 외부에서 접근 가능하도록 하려면 방화벽 차단해야함
                              같은 wifi면 접속됨
    """
    print("="*80)
    print()

    print(f"capture 이미지 저장 위치 : {os.path.join(os.getcwd(), 'capture.jpg')}")

    print()
    print("="*80)
    print()

    app.run(host="172.30.1.54", port=5000, debug=False)
    # app.run(host="0.0.0.0", debug=False)