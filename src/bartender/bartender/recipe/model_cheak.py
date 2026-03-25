#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class YOLORealSenseNode(Node):
    def __init__(self):
        super().__init__('yolo_realsense_node')

        # ROS2 Publisher
        self.pub = self.create_publisher(Image, '/yolo/image', 10)
        self.br = CvBridge()

        # YOLOv8 모델 로드
        self.model = YOLO("recipe/best.pt")

        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # 타이머 루프 (실시간)
        self.timer = self.create_timer(0.03, self.timer_callback)  # 약 30ms 마다

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # RealSense -> OpenCV 이미지
        img = np.asanyarray(color_frame.get_data())

        # YOLO 추론
        results = self.model(img)

        # 이미지에 bbox 그리기
        annotated_frame = results[0].plot()

        # ROS2 topic publish
        self.pub.publish(self.br.cv2_to_imgmsg(annotated_frame, encoding="bgr8"))

        # Optional: 노드 화면 확인
        cv2.imshow("YOLOv8n RealSense", annotated_frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YOLORealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
