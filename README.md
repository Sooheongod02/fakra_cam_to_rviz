cd catkin_ws/src
catkin_create_pkg my_camera_driver rospy sensor_msgs cv_bridge std_msgs

cd my_camera_driver
mkdir scripts

cd scripts
gedit camera_pub.py

chmod +x camera_pub.py
######

cd catkin_ws/src/my_camera_driver
mkdir launch
cd launch

gedit all_cam.launch
