import numpy as np

def estimate_depth_from_window(
    depth_image,
    center_xy,
    window_radius=5,
    std_threshold=0.03,
    k=1.0,
    min_inliers=8,
    depth_scale=0.001,
    min_depth=0.1,
    max_depth=1.2,
    reducer="median",
    fallback_reducer="median",
    prefer_near_cluster=True,
):
    """
    지정된 픽셀(center_xy) 주변 윈도우 내의 깊이 값을 통계적으로 분석하여
    대표 깊이(거리)를 추정합니다.
    """
    cx, cy = center_xy
    h, w = depth_image.shape

    # 1. ROI(Region of Interest) 추출
    x1 = max(0, int(cx - window_radius))
    x2 = min(w, int(cx + window_radius + 1))
    y1 = max(0, int(cy - window_radius))
    y2 = min(h, int(cy + window_radius + 1))

    roi = depth_image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, {'mean': 0.0, 'std': 0.0, 'num_samples': 0, 'num_inliers': 0}

    # 2. 미터 단위 변환
    roi_m = roi.astype(np.float32) * depth_scale

    # 3. 유효 범위 필터링 (Min/Max Depth)
    valid_mask = (roi_m >= min_depth) & (roi_m <= max_depth)
    valid_pixels = roi_m[valid_mask]

    num_samples = valid_pixels.size
    stats = {
        'mean': 0.0,
        'std': 0.0,
        'num_samples': num_samples,
        'num_inliers': 0
    }

    if num_samples < min_inliers:
        return 0.0, stats

    # 4. 대표값 선정 (Outlier 제거 전 기준점)
    # prefer_near_cluster가 True면, 물체 경계면에서 배경보다 물체(가까운 쪽)를 선호하도록
    # 중앙값(50%) 대신 하위 25% 지점을 기준점으로 잡습니다.
    if prefer_near_cluster:
        center_est = np.percentile(valid_pixels, 25)
    else:
        center_est = np.median(valid_pixels)

    # 5. Outlier 제거 (Inlier Filtering)
    # 기준점(center_est)에서 ±(k * std_threshold) 범위 내의 값만 유효한 값으로 인정
    margin = k * std_threshold
    inlier_mask = np.abs(valid_pixels - center_est) <= margin
    inliers = valid_pixels[inlier_mask]

    # Inlier가 너무 적으면 Fallback
    if inliers.size < min_inliers:
        if fallback_reducer == "mean":
            result = np.mean(valid_pixels)
        else:
            result = np.median(valid_pixels)
        
        stats['mean'] = float(np.mean(valid_pixels))
        stats['std'] = float(np.std(valid_pixels))
        return float(result), stats

    # 6. 최종 결과 계산
    if reducer == "mean":
        result = np.mean(inliers)
    else:
        result = np.median(inliers)

    stats['mean'] = float(np.mean(inliers))
    stats['std'] = float(np.std(inliers))
    stats['num_inliers'] = inliers.size

    return float(result), stats