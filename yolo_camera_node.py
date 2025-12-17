#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloNode:
    def __init__(self):
        # 노드 이름은 launch 파일에서 설정되므로 anonymous=True 제거
        rospy.init_node('yolo_node', anonymous=False)

        # ---------------------------------------------------------
        # 1. 파라미터로 설정 받기 (Launch 파일에서 입력받음)
        # ---------------------------------------------------------
        # 기본값은 /front_camera/compressed 로 설정
        self.input_topic = rospy.get_param("~input_topic", "/front_camera/compressed")
        self.output_topic = rospy.get_param("~output_topic", "/yolo/front/image")
        self.conf_thres = rospy.get_param("~conf", 0.5) # 감지 정확도 임계값

        rospy.loginfo(f"🚀 YOLOv8 Node Start! Target: {self.input_topic}")

        # ---------------------------------------------------------
        # 2. 모델 및 브릿지 초기화
        # ---------------------------------------------------------
        # yolov8n.pt (nano) 모델 사용 - 4개를 돌려야 하므로 가벼운 모델 추천
        self.model = YOLO("yolov8n.pt") 
        self.bridge = CvBridge()
        
        # ---------------------------------------------------------
        # 3. Publisher & Subscriber
        # ---------------------------------------------------------
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        
        # CompressedImage 토픽 구독
        self.sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback)

    def image_callback(self, msg):
        try:
            # -----------------------------------------------------
            # 4. 디코딩 (Compressed -> OpenCV)
            # -----------------------------------------------------
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # -----------------------------------------------------
            # 5. YOLOv8 추론
            # -----------------------------------------------------
            results = self.model(frame, conf=self.conf_thres, verbose=False)
            
            # 결과 시각화 (이미지에 박스 그리기)
            annotated_frame = results[0].plot()

            # -----------------------------------------------------
            # 6. 결과 발행 (Rviz용)
            # -----------------------------------------------------
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pub.publish(img_msg)

        except Exception as e:
            rospy.logerr(f"이미지 처리 오류: {e}")

if __name__ == '__main__':
    try:
        YoloNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
