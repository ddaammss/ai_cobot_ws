import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
from sklearn.decomposition import PCA

def main():
    # ---------------------------------------------------------
    # 1. 설정 및 초기화
    # ---------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'shake.pt')  # 같은 폴더의 best.pt 사용
    print(f"모델 로딩 중: {model_path}...")
    model = YOLO(model_path)

    # RealSense 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 640x480 해상도 추천 (YOLO 학습 해상도와 일치)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 스트리밍 시작
    profile = pipeline.start(config)

    # Depth와 Color를 정확히 겹치기 위한 Align 객체 생성 (필수!)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 카메라 내부 파라미터(Intrinsics) 가져오기 (3D 변환용)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

    print("카메라 시작! (종료하려면 'q' 키를 누르세요)")

    try:
        while True:
            # ---------------------------------------------------------
            # 2. 프레임 획득 및 정렬
            # ---------------------------------------------------------
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames) # Depth를 Color에 맞춤

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # numpy 배열로 변환
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # ---------------------------------------------------------
            # 3. YOLOv8 추론 (Segmentation)
            # ---------------------------------------------------------
            results = model(color_image, verbose=False)
            
            # 객체가 감지되었는지 확인
            if results[0].masks is not None:
                # 첫 번째 감지된 객체(머들러)만 처리
                mask = results[0].masks.data[0].cpu().numpy() # GPU 텐서를 numpy로 변환
                
                # 마스크 크기를 이미지 크기에 맞게 리사이징 (가끔 다를 수 있음)
                mask = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]))
                
                # 마스크가 있는 영역(True)의 픽셀 좌표(u, v) 추출
                # mask > 0.5 는 확률 임계값
                ys, xs = np.where(mask > 0.5)

                if len(xs) > 0:
                    # ---------------------------------------------------------
                    # 4. 3D Point Cloud 변환 (벡터화 연산)
                    # ---------------------------------------------------------
                    # 해당 픽셀들의 Depth 값 가져오기 (단위: mm -> 미터 변환)
                    z_vals = depth_image[ys, xs] * 0.001 
                    
                    # 유효한 깊이값(0이 아닌 것)만 필터링
                    valid_indices = z_vals > 0
                    z_vals = z_vals[valid_indices]
                    xs_valid = xs[valid_indices]
                    ys_valid = ys[valid_indices]

                    if len(z_vals) > 50: # 점이 너무 적으면 계산 불가
                        # 2D 픽셀 -> 3D 좌표 공식 (Pinhole Camera Model)
                        x_vals = (xs_valid - ppx) * z_vals / fx
                        y_vals = (ys_valid - ppy) * z_vals / fy
                        
                        # (N, 3) 형태의 포인트 클라우드 생성
                        points_3d = np.stack((x_vals, y_vals, z_vals), axis=-1)

                        # ---------------------------------------------------------
                        # 5. PCA로 기울기 및 끝점 계산
                        # ---------------------------------------------------------
                        pca = PCA(n_components=3)
                        pca.fit(points_3d)

                        center_point = pca.mean_       # 머들러의 중심 (x, y, z)
                        direction_vector = pca.components_[0] # 주성분 벡터 (기울기 방향)

                        # [중요] 벡터 방향 정렬 (위쪽을 향하도록)
                        # 카메라 좌표계에서 Y는 아래쪽이 양수(+)임. 
                        # 따라서 머들러 위쪽(화면 상단)은 Y값이 작아야 함.
                        if direction_vector[1] > 0: 
                            direction_vector = -direction_vector

                        # 시각화를 위한 2D 투영 (벡터 그리기용)
                        # 중심점에서 벡터 방향으로 10cm 이동한 점 계산
                        p_start = center_point
                        p_end = center_point + direction_vector * 0.1 # 10cm 길이

                        # 3D -> 2D 픽셀로 다시 변환 (그리기 위해)
                        def project_point(p):
                            u = int(p[0] * fx / p[2] + ppx)
                            v = int(p[1] * fy / p[2] + ppy)
                            return (u, v)

                        u_start, v_start = project_point(p_start)
                        u_end, v_end = project_point(p_end)

                        # "끝점(Top Tip)" 찾기 로직:
                        # 포인트들 중 Y값이 가장 작은 점 (화면상 가장 위)
                        top_idx = np.argmin(ys_valid) 
                        top_u, top_v = xs_valid[top_idx], ys_valid[top_idx]


                        # ---------------------------------------------------------
                        # 6. 시각화 (Drawing)
                        # ---------------------------------------------------------
                        # 마스크 영역 초록색으로 칠하기
                        color_image[mask > 0.5] = color_image[mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
                        
                        # 중심점 (노란색)
                        cv2.circle(color_image, (u_start, v_start), 5, (0, 255, 255), -1)
                        
                        # 기울기 벡터 (파란 선)
                        cv2.line(color_image, (u_start, v_start), (u_end, v_end), (255, 0, 0), 3)
                        
                        # 잡아야 할 끝점 (빨간 점) - 여기가 그리퍼 목표 지점!
                        cv2.circle(color_image, (top_u, top_v), 8, (0, 0, 255), -1)

                        # 정보 텍스트
                        text = f"Vec: ({direction_vector[0]:.2f}, {direction_vector[1]:.2f}, {direction_vector[2]:.2f})"
                        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 화면 출력
            cv2.imshow('RealSense YOLOv8 Grasping', color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()