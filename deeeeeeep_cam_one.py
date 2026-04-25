import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except Exception:  # RealSense is required only for live camera mode.
    rs = None


TARGET_COLORS = ("red", "yellow", "purple")
DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("best_cube.pt")


@dataclass
class DetectionConfig:
    weights: Path = DEFAULT_WEIGHTS
    imgsz: int = 736
    conf: float = 0.25
    device: str = "cpu"
    cube_threshold_m: float = 0.10


def clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def median_depth_m(
    depth_image: np.ndarray,
    x: int,
    y: int,
    depth_scale: float = 1.0,
    radius: int = 4,
) -> Optional[float]:
    h, w = depth_image.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    vals = patch[patch > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals) * depth_scale)


def hsv_masks(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s > 45) & (v > 45)
    masks = {
        "red": valid & (((h >= 0) & (h <= 12)) | ((h >= 165) & (h <= 179))),
        "yellow": valid & (h >= 18) & (h <= 40),
        "purple": valid & (h >= 122) & (h <= 163),
    }
    kernel = np.ones((5, 5), np.uint8)
    out = {}
    for color, mask in masks.items():
        m = (mask.astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        out[color] = m
    return out


def force_color_ryp(bgr: np.ndarray) -> Tuple[str, Dict[str, int]]:
    masks = hsv_masks(bgr)
    scores = {color: int(cv2.countNonZero(mask)) for color, mask in masks.items()}
    if max(scores.values(), default=0) > 0:
        return max(scores, key=scores.get), scores

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s > 25) & (v > 30)
    if valid.sum() == 0:
        return "red", scores
    hue = float(np.median(h[valid]))
    distances = {
        "red": min(abs(hue - 0), abs(hue - 179)),
        "yellow": abs(hue - 29),
        "purple": abs(hue - 142),
    }
    return min(distances, key=distances.get), scores


def refine_bbox_by_color(
    image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[str, Tuple[int, int, int, int], Optional[np.ndarray], Dict[str, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(bbox, width, height)
    roi = image[y1:y2, x1:x2]
    color, scores = force_color_ryp(roi)
    mask = hsv_masks(roi)[color]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80.0, roi.shape[0] * roi.shape[1] * 0.003)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return color, (x1, y1, x2, y2), None, scores

    contour = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(contour)
    pad = max(3, int(max(rw, rh) * 0.035))
    refined = clamp_box((x1 + rx - pad, y1 + ry - pad, x1 + rx + rw + pad, y1 + ry + rh + pad), width, height)

    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect)
    corners[:, 0] += x1
    corners[:, 1] += y1
    return color, refined, corners.astype(np.float32), scores


def deproject_point(intrinsics: Any, pixel: Tuple[float, float], depth_m: float) -> Optional[np.ndarray]:
    if rs is None or depth_m is None or depth_m <= 0:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
    return np.array(point, dtype=np.float32)


def estimate_side_m(
    depth_image: np.ndarray,
    depth_intrin: Any,
    bbox: Tuple[int, int, int, int],
    corners: Optional[np.ndarray],
    depth_scale: float,
) -> Tuple[Optional[float], Optional[float], str]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    center_depth = median_depth_m(depth_image, cx, cy, depth_scale, radius=8)
    if center_depth is None:
        return None, None, "no_depth"

    points_2d: List[Tuple[float, float]]
    if corners is not None and len(corners) == 4:
        points_2d = [(float(x), float(y)) for x, y in corners]
    else:
        points_2d = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    points_3d: List[np.ndarray] = []
    for px, py in points_2d:
        local_depth = median_depth_m(depth_image, int(px), int(py), depth_scale, radius=5)
        depth = local_depth if local_depth is not None else center_depth
        point = deproject_point(depth_intrin, (px, py), depth)
        if point is not None:
            points_3d.append(point)

    if len(points_3d) == 4:
        edges = [
            float(np.linalg.norm(points_3d[(i + 1) % 4] - points_3d[i]))
            for i in range(4)
        ]
        sane_edges = [e for e in edges if 0.01 <= e <= 1.5]
        if sane_edges:
            return float(np.median(sane_edges)), center_depth, "corner_deproject"

    fx = float(getattr(depth_intrin, "fx", 0.0) or 0.0)
    fy = float(getattr(depth_intrin, "fy", 0.0) or 0.0)
    if fx > 0 and fy > 0:
        width_m = abs(x2 - x1) * center_depth / fx
        height_m = abs(y2 - y1) * center_depth / fy
        side = float(np.median([width_m, height_m]))
        return side, center_depth, "bbox_pinhole"
    return None, center_depth, "no_intrinsics"


class CubeFilterDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        weights = Path(config.weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.weights = weights
        self.model = YOLO(str(weights))

    def detect_frame(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        depth_intrin: Any = None,
        depth_scale: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        results = self.model(
            source=color_image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            device=self.config.device,
            verbose=False,
            stream=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy.shape[0] == 0:
            return None

        height, width = color_image.shape[:2]
        candidates: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        for i in range(boxes.xyxy.shape[0]):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            bbox = clamp_box(tuple(xyxy), width, height)
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            if depth_image is not None:
                depth = median_depth_m(depth_image, cx, cy, depth_scale, radius=8)
                depth_sort = depth if depth is not None else math.inf
            else:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                depth_sort = -float(area)
            candidates.append((depth_sort, i, bbox))

        candidates.sort(key=lambda item: item[0])
        _sort_depth, best_i, bbox = candidates[0]
        conf = float(boxes.conf[best_i].item())
        color, refined_bbox, corners, color_scores = refine_bbox_by_color(color_image, bbox)

        side_m = None
        depth_m = None
        center_xyz = None
        size_method = "no_depth"
        if depth_image is not None and depth_intrin is not None:
            side_m, depth_m, size_method = estimate_side_m(
                depth_image, depth_intrin, refined_bbox, corners, depth_scale
            )
            if depth_m is not None and rs is not None:
                cx = int((refined_bbox[0] + refined_bbox[2]) / 2)
                cy = int((refined_bbox[1] + refined_bbox[3]) / 2)
                p = deproject_point(depth_intrin, (cx, cy), depth_m)
                if p is not None:
                    center_xyz = [float(v) for v in p.tolist()]

        object_type = None
        if side_m is not None:
            object_type = "cube" if side_m < self.config.cube_threshold_m else "filter"

        return {
            "color": color,
            "object_type": object_type,
            "side_m": side_m,
            "depth_m": depth_m,
            "center_xyz_m": center_xyz,
            "bbox_xyxy": [int(v) for v in refined_bbox],
            "confidence": conf,
            "color_scores": color_scores,
            "size_method": size_method,
            "weights": str(self.weights),
        }


def draw_detection(image: np.ndarray, det: Dict[str, Any]) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = det["bbox_xyxy"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
    side = det.get("side_m")
    side_text = f"{side * 100:.1f}cm" if side is not None else "size=?"
    obj = det.get("object_type") or "unknown"
    depth = det.get("depth_m")
    depth_text = f"{depth:.2f}m" if depth is not None else "depth=?"
    label = f"{det['color']} {obj} {side_text} {det['confidence']:.2f}"
    cv2.putText(out, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(out, depth_text, (x1, min(out.shape[0] - 8, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    return out


def run_camera(args: argparse.Namespace) -> None:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed; install it on the Ubuntu RealSense runtime.")

    detector = CubeFilterDetector(
        DetectionConfig(
            weights=Path(args.weights),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            cube_threshold_m=args.cube_threshold_m,
        )
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth scale: {depth_scale} m/unit")
    print(f"[INFO] weights: {detector.weights}")

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            det = detector.detect_frame(color_image, depth_image, depth_intrin, depth_scale)

            shown = color_image
            if det is not None:
                shown = draw_detection(color_image, det)
                if args.json_lines:
                    print(json.dumps(det, ensure_ascii=False))
                else:
                    side = det["side_m"]
                    size = f"{side:.3f}m" if side is not None else "?"
                    depth = det["depth_m"]
                    depth_s = f"{depth:.3f}m" if depth is not None else "?"
                    print(
                        f"[DETECT] {det['color']} {det['object_type']} "
                        f"side={size} depth={depth_s} conf={det['confidence']:.2f}"
                    )

            if not args.no_window:
                cv2.imshow("RealSense Cube/Filter", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RealSense red/yellow/purple cube/filter detector")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cube-threshold-m", type=float, default=0.10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--json-lines", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_camera(parse_args())


if __name__ == "__main__":
    main()
