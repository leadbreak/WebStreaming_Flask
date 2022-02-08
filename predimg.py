import io, os, sys
import cv2
import time, signal
import json, base64
import requests, socketserver
import threading
import numpy as np
import core.utils as utils
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from ctypes import *
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import filter_boxes
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from threading import Lock
from http import server
from urllib.parse import urlparse, parse_qs

def predict(FLAGS_framework='tflite',
        FLAGS_weights='./ckpt_data/yolov4-custom.tflite',
        FLAGS_size=416,
        FLAGS_tiny=True,
        FLAGS_model='yolov4',
        FLAGS_dir='./data/img/',
        FLAGS_image='capture.jpg',
        FLAGS_output='result.jpg',
        FLAGS_iou=0.45,
        FLAGS_score=0.25):
        input_size = FLAGS_size
        #file_path = FLAGS_dir + FLAGS_image
        file_path = FLAGS_image
        print('------------------------------------------------------------------------')
        print('\t{}({}*{})'.format(file_path, input_size, input_size))
        print('------------------------------------------------------------------------')
        
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        if FLAGS_framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS_weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS_model == 'yolov3' and FLAGS_tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            saved_model_loaded = tf.saved_model.load(FLAGS_weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS_iou,
            score_threshold=FLAGS_score
        )
        annotation_path =  os.path.dirname(os.path.realpath(__file__)) + "/" + file_path.replace('./', '/').split('.')[0] + '.txt'
        if os.path.isfile(annotation_path) or os.path.isdir(annotation_path):
            with open(annotation_path, 'r+') as f:
                f.truncate(0)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(original_image, pred_bbox, annotation_path=annotation_path)
        #image = utils.draw_bbox(image_data*255, pred_bbox, annotation_path=annotation_path)
        image = Image.fromarray(image.astype(np.uint8))
        #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(FLAGS_output, image)
        '''
        predict_img_path =  os.path.dirname(os.path.realpath(__file__)) + file_path.replace('./', '/').split('.')[0] + '_label.jpg'
        print('predict_img_path =', predict_img_path)
        cv2.imwrite(predict_img_path, image)
        '''
        return image

if __name__=="__main__" :
    predict()