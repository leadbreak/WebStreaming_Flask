# WebStreaming_Flask
This is for the streaming images from webcam by Flask.


* webcam을 통해 내부 ip 등으로 스트리밍하고, 이를 mlops의 과정에 포함시켜 데이터 수집 및 추론을 가능하게 하는 것이 목적
* 이 코드는 내부 ip 등으로 웹캠 영상을 스트리밍하고, 상황에 따라 capture 페이지에 GET 명령을 보내, 로컬 영역 안의 capture.jpg 파일을
  바꿔가며 추가적인 메모리의 할당 및 사용없이 작업하도록 구성 - 추후 백앤드 영역에서 연결해 사용할 때에도 메모리 할당의 최적화를 목적으로 함

* app router는 아래와 같음 \
/ : default page
/streaming : video를 출력
/capture : video 해당 장면을 capture해서 보여주고, 해당 이미지를 현재 작업 폴더의 capture.jpg로 저장
/cat : capture의 테스트 페이지로 현재 폴더의 cat.jpg를 출력해서 보여줌

- 각 페이지는 templates 안의 html 파일과 연동되어 있음

