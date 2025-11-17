import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ================= 配置区 =================自动补全太好用了你知道吗
SHAPE_WEIGHTS = r"/home/detect/train25/weights/best.pt"  # 形状模型权重路径

# =========================================================
# 1. 颜色识别（HSV）
# =========================================================
def dominant_color_bgr(
    bgr_img,
    sat_thresh=40,        # 灰/白过滤阈值（S 通道）
    min_ratio=0.03,       # 颜色占比太少就认为没颜色
    redfam_min_ratio=0.7, # 红系像素占比至少多少才算红
    pink_v_thresh=110,    # 判定粉色的亮度阈值（V 通道）
    pink_s_thresh=190     # 判定粉色的饱和度上限（S 通道）
):

    if bgr_img is None or bgr_img.size == 0:
        return None

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 过滤掉太灰/太白的区域
    valid_mask = s > sat_thresh
    if valid_mask.sum() == 0:
        return None

    h_valid = h[valid_mask]
    s_valid = s[valid_mask]
    v_valid = v[valid_mask]

    # 颜色区间（大类）
    COLOR_RANGES = {
        "red":    [(0, 15)],
        "yellow": [(18, 35)],
        "green":  [(40, 85)],
        "blue":   [(90, 120)],
        "purple": [(125, 149)],
        # pink 不在这里写，后面从 red 家族细分
    }

    def count_range(h_vals, ranges):
        total = 0
        for lo, hi in ranges:
            if lo <= hi:
                total += ((h_vals >= lo) & (h_vals <= hi)).sum()
            else:
                # 环绕情况（这里其实用不到，留着以防以后扩展）
                total += ((h_vals >= lo) | (h_vals <= hi)).sum()
        return total

    total_pixels = h_valid.size

    # 1) 先选出大类颜色
    color_scores = {
        name: count_range(h_valid, ranges)
        for name, ranges in COLOR_RANGES.items()
    }
    best_color, best_count = max(color_scores.items(), key=lambda x: x[1])

    if best_count / total_pixels < min_ratio:
        # 没有明显主色
        return None

    # 2) 大类不是 red，直接返回
    if best_color != "red":
        return best_color

    # 3) red 家族内部拆 red / pink
    redfam_mask = (
        ((h_valid >= 0) & (h_valid <= 15)) |
        ((h_valid >= 150) & (h_valid <= 179))
    )
    redfam_count = redfam_mask.sum()
    if redfam_count == 0 or redfam_count / total_pixels < redfam_min_ratio:
        # 红系像素比例太小，就当普通 red
        return "red"

    s_red = s_valid[redfam_mask].astype(np.float32)
    v_red = v_valid[redfam_mask].astype(np.float32)

    mean_s = float(s_red.mean())
    mean_v = float(v_red.mean())

    # 经验规则：亮度较高 + 饱和度相对没那么高 → pink
    if mean_v >= pink_v_thresh and mean_s <= pink_s_thresh:
        return "pink"
    else:
        return "red"


# =========================================================
# 2. YOLO + RealSense 实时推理
# =========================================================


IMG_SIZE = 736
DEVICE = "cpu"  # "0" = CUDA:0, "cpu" = 只用 CPU


def main():
    if not SHAPE_WEIGHTS:
        raise RuntimeError("best.pt needed ！")

    # 载入 YOLOv8 形状模型
    print(f"[INFO] loading YOLOv8 shape model: {SHAPE_WEIGHTS}")
    model = YOLO(SHAPE_WEIGHTS)
    shape_names = model.names  # id -> 'cuboid'/'cylinder'/...

    # RealSense 管线
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # 深度尺度
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth scale: {depth_scale} m/unit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # ---------------- YOLOv8 检测形状 ----------------
            results = model(
                source=color_image,
                imgsz=IMG_SIZE,
                conf=0.25,
                device=DEVICE,
                verbose=False,
                stream=False,
            )
            res = results[0]
            boxes = res.boxes

            if boxes is None or boxes.xyxy.shape[0] == 0:
                # 没检测到东西
                cv2.imshow("RealSense Color+Shape", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            h_img, w_img = color_image.shape[:2]

            # 遍历每个检测框（可以只用第一个，也可以全用；这里全用）
            for i in range(boxes.xyxy.shape[0]):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                shape_name = shape_names.get(cls_id, str(cls_id))

                # 坐标裁剪到图像范围
                x1 = max(0, min(x1, w_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                y2 = max(0, min(y2, h_img - 1))
                if x2 <= x1 or y2 <= y1:
                    roi = color_image
                else:
                    roi = color_image[y1:y2, x1:x2]

                # ---------------- HSV 识别颜色（主导） ----------------
                pred_color = dominant_color_bgr(roi)
                if pred_color is None:
                    pred_color = "unknown"

                # 深度 & 3D 坐标：用框中心像素
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                depth = depth_image[y_center, x_center] * depth_scale
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [x_center, y_center], depth
                )

                # 画框 + 文本
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 第一行：颜色+形状（颜色优先）
                label = f"{pred_color} {shape_name} {conf:.2f}"
                cv2.putText(
                    color_image,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # 第二行：深度
                depth_text = f"{depth:.2f}m"
                cv2.putText(
                    color_image,
                    depth_text,
                    (x1, y2 + 20 if y2 + 20 < h_img else y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # 控制台打印一份详细 3D 信息
                print(
                    f"[DETECT] {pred_color} {shape_name} | "
                    f"conf={conf:.2f} | depth={depth:.3f}m | "
                    f"X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}"
                )

            cv2.imshow("RealSense Color+Shape", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
