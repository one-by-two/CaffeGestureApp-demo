# CaffeGestureApp-demo


This project uses Caffe mobile depth learning algorithm and Haar feature classifier to accurately 
locate fingertip and eye coordinates in real time, and realizes a new human-computer interaction mode based on 
gesture recognition and eye tracking according to the spatial relationship between intelligent devices and users.

1. Responsible for the transplantation and deployment of Caffe deep learning framework in mobile terminals.

2. Realize gesture recognition and Haar feature face detection in complex background.

3. According to the relationship between finger coordinates, human eye coordinates and the relative position of the screen, desige a screen localization algorithm to determine the mouse position based on the user pointing to the screen and perform mouse-like processing.

该项目通过Caffe移动端深度学习算法与Haar特征分类器实时精确定位指尖与人眼坐标，并根据智能设备与用户的空间关系，
实现基于手势识别与视线追踪的新的人机交互方式。

1、负责Caffe深度学习框架在移动端的移植与部署。

2、实现复杂背景下的手势识别与Haar特征人脸检测。

3、根据手指坐标、人眼坐标与屏幕的相对位置关系，设计屏幕定位算法，确定用户指向屏幕的鼠标位置并进行类鼠标处理。

# Notice
    
将CaffeGestureApp-demo/caffe_model/bvlc_alexnet/中的bvlc_alexnet.caffemodel与deployH.prototxt
拷贝至手机外部存储 sdcard.getAbsolutePath() + "/caffe_mobile/bvlc_alexnet"中;

    
    
