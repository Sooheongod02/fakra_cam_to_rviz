#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('gmsl_camera_node', anonymous=True)
    
    # [수정] 런치 파일에서 설정값 받아오기 (기본값 설정)
    device_id = rospy.get_param('~device_id', 0)
    topic_name = rospy.get_param('~camera_name', 'camera')
    
    # 각 카메라별 해상도와 FPS를 다르게 설정 가능  
    width = rospy.get_param('~width', 1920)
    height = rospy.get_param('~height', 1536)
    #fps = rospy.get_param('~fps', 30)  # <-- 핵심: FPS를 변수로 받음

    pub = rospy.Publisher(f'/{topic_name}/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    rospy.loginfo(f"[{topic_name}] Starting /dev/video{device_id} ({width}x{height} - FPS Auto")

    # GStreamer 파이프라인 (변수 적용)
    gst_str = (
        f"v4l2src device=/dev/video{device_id} ! "
        f"video/x-raw,format=UYVY,width={width},height={height} ! " 
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false drop=1"
       
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        rospy.logerr(f"[{topic_name}] 열기 실패! FPS({fps})나 해상도({width}x{height})가 맞는지 확인하세요.")
        return

    rate = rospy.Rate(30) # 루프 속도도 FPS에 맞춤
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = f"{topic_name}_link"
            pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
