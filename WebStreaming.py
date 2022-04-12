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

from flask import Flask, render_template, Response, request
import cv2, time, os
import numpy as np

import time
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

# Inference in Flask server
def generate_inference_frames(FLAGS) :
    FLAGS_framework = "tflite"
    FLAGS_weights = "./ckpt_data/yolov4-custom.tflite"   
    FLAGS_size = 416
    FLAGS_tiny = False
    FLAGS_model = 'yolov4'
    FLAGS_iou = 0.45
    FLAGS_score = 0.25 

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    STRIDES = np.array(cfg.YOLO.STRIDES)
    if FLAGS_model == 'yolov4':
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS_tiny)
    elif FLAGS_model == 'yolov3':
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS_tiny)
    XYSCALE = cfg.YOLO.XYSCALE if FLAGS_model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    input_size = FLAGS_size

    # print("Video from: ", video_path )
    # vid = cv2.VideoCapture(0)

    interpreter = tf.lite.Interpreter(model_path=FLAGS_weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    frame_id = 0
    while True:
        return_value, frame = camera.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == camera.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                        input_shape=tf.constant([input_size, input_size]))


        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS_iou,
            score_threshold=FLAGS_score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        # print(info)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        frame = result
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()
        # frame = result.tobytes()

        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        frame_id += 1



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
                # print("return value : ", ret)
                # print("buffer :", buffer)
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
            img_dir = os.path.join(os.path.expanduser("~"), "capture.jpg")        
            cv2.imwrite(img_dir, img)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            break


@app.route('/') #root page
def index() :
    print(f"ip address is... {request.remote_addr}:5000/")
    return render_template('index.html')

@app.route('/cat')
def test_cat() :
    return render_template('cat.html', image_file='./cat.jpg')

@app.route('/stream')
def stream() :
    print(f"ip address is... {request.environ['REMOTE_ADDR']}:5000/stream")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture() :    
    return Response(generate_image(), mimetype='multipart/x-mixed-replace; boundary=frame'), render_template('capture.html', image_file='./capture.jpg')

@app.route('/inference')
def inference() :
    return Response(generate_inference_frames(FLAGS), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__" :
    """
    app.run() - local에서만 작동
    app.run(ip 번호) - 내부 로컬 ip에서만
    app.run(host='0.0.0.0') - 어떤 호스트에서도 연결 가능하도록
    """
    print("="*80)
    print()

    # /mnt/c/Users/dqkim/capture.jpg
    print(f'capture 이미지 저장 위치 : {os.path.join(os.path.expanduser("~"), "capture.jpg")}')
    
    print()
    print("="*80)
    print()

    # app.run(host="172.30.1.54", port=5000, debug=False)
    app.run(debug=False)
    # app.run(port=5000, debug=False)
    