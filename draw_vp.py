# File: draw_vp.py
# Visual prompts visualization utilities

import re
import json
import cv2
import numpy as np
import torch
import math
from pathlib import Path
from torchvision import transforms

# --- Configuration ---
OUTPUT_DIR = Path("./inference_visualizations")
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
COLOR_RED = (0, 0, 255)  # BGR format for Red
FIXED_ROTATION_RADIUS = 14


def _draw_star(image, center_pt):
    """Draws a solid red 5-pointed star."""
    radius = 10
    points = []
    center_x, center_y = center_pt
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        current_radius = radius if i % 2 == 0 else radius / 2.5
        x = center_x + current_radius * math.cos(angle)
        y = center_y + current_radius * math.sin(angle)
        points.append((int(x), int(y)))
    pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts_np], COLOR_RED)


def _draw_fixed_axis_rotation(image, center_pt, axis, is_cw):
    """Draws the fixed-size rotation symbol in red."""
    cx, cy = center_pt
    r = FIXED_ROTATION_RADIUS
    color = COLOR_RED
    start_angle, end_angle = 0, -270 if not is_cw else 270
    cv2.ellipse(image, (cx, cy), (r, r), 0, start_angle, end_angle, color, THICKNESS, cv2.LINE_AA)

    end_rad = math.radians(end_angle)
    ex = int(cx + r * math.cos(end_rad))
    ey = int(cy + r * math.sin(end_rad))
    tangent = end_rad + math.pi / 2 if not is_cw else end_rad - math.pi / 2
    arrow_size = r * 0.4
    p1 = (int(ex - arrow_size * math.cos(tangent + math.pi / 6)), int(ey - arrow_size * math.sin(tangent + math.pi / 6)))
    p2 = (int(ex - arrow_size * math.cos(tangent - math.pi / 6)), int(ey - arrow_size * math.sin(tangent - math.pi / 6)))
    cv2.line(image, (ex, ey), p1, color, THICKNESS, cv2.LINE_AA)
    cv2.line(image, (ex, ey), p2, color, THICKNESS, cv2.LINE_AA)
    
    # Draw axis symbol
    if axis == "Z": cv2.circle(image, (cx, cy), 2, (0, 0, 0), -1)
    elif axis == "Y": cv2.line(image, (cx, int(cy - 1.2 * r)), (cx, int(cy + 1.2 * r)), color, 1, cv2.LINE_AA)
    elif axis == "X": cv2.line(image, (int(cx - 1.2 * r), cy), (int(cx + 1.2 * r), cy), color, 1, cv2.LINE_AA)
    
    cv2.putText(image, axis, (cx + 4, cy - 4), FONT, FONT_SCALE, (0,0,0), 2, cv2.LINE_AA)


def _draw_jagged_arrow(image, x1, y1, x2, y2):
    """Draws a broken straight line with an arrowhead in red."""
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1: return
    udx, udy = dx / dist, dy / dist
    
    segment_length, gap_length = 10, 5
    current_dist = 0
    while current_dist < dist - (segment_length + gap_length):
        start_pt = (int(x1 + current_dist * udx), int(y1 + current_dist * udy))
        end_pt = (int(x1 + (current_dist + segment_length) * udx), int(y1 + (current_dist + segment_length) * udy))
        cv2.line(image, start_pt, end_pt, COLOR_RED, THICKNESS)
        current_dist += segment_length + gap_length

    angle = math.atan2(dy, dx)
    arrow_size = 10
    p1 = (int(x2 - arrow_size * math.cos(angle - math.pi / 6)), int(y2 - arrow_size * math.sin(angle - math.pi / 6)))
    p2 = (int(x2 - arrow_size * math.cos(angle + math.pi / 6)), int(y2 - arrow_size * math.sin(angle + math.pi / 6)))
    cv2.line(image, (x2, y2), p1, COLOR_RED, THICKNESS)
    cv2.line(image, (x2, y2), p2, COLOR_RED, THICKNESS)


def _extract_second_answer(text: str) -> str:
    """Finds all <answer> blocks and returns the content of the second one."""
    if not isinstance(text, str): return ""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[1].strip() if len(matches) >= 2 else ""


def _convert_tensor_to_drawable_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverses transformations to get a drawable OpenCV image from a tensor."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    pil_image = transforms.ToPILImage()(tensor.cpu())
    opencv_image_rgb = np.array(pil_image)
    return cv2.cvtColor(opencv_image_rgb, cv2.COLOR_RGB2BGR)


def _draw_prompts(image: np.ndarray, prompts: dict) -> np.ndarray:
    """Draws all available visual prompts onto an image."""
    if not isinstance(prompts, dict): return image
    
    rotation_keys = ["rotation_x_cw", "rotation_x_ccw", "rotation_y_cw", "rotation_y_ccw", "rotation_z_cw", "rotation_z_ccw"]
    
    try:
        for pts in prompts.get("bbox", []):
            cv2.rectangle(image, (int(pts[0]), int(pts[1])), (int(pts[2]), int(pts[3])), COLOR_RED, THICKNESS)
        for pts in prompts.get("arrow", []):
            cv2.arrowedLine(image, (int(pts[0]), int(pts[1])), (int(pts[2]), int(pts[3])), COLOR_RED, THICKNESS, tipLength=0.2)
        for pts in prompts.get("point", []):
            cv2.circle(image, (int(pts[0]), int(pts[1])), 5, COLOR_RED, -1)
        for pts in prompts.get("star_point", []):
            _draw_star(image, (int(pts[0]), int(pts[1])))
        for pts in prompts.get("jagged_arrow", []):
            _draw_jagged_arrow(image, int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
        
        for rot_key in rotation_keys:
            for center_coords in prompts.get(rot_key, []):
                parts = rot_key.split("_")
                axis, direction = parts[1].upper(), parts[2]
                _draw_fixed_axis_rotation(image, (int(center_coords[0]), int(center_coords[1])), axis, is_cw=(direction == "cw"))
                
    except (TypeError, IndexError, ValueError) as e:
        print(f"Warning: Could not draw prompts due to invalid coordinate data: {e}")
    
    return image


def visualize_generated_prompts(image_tensor: torch.Tensor, generated_text: str, output_filename: str):
    """
    Parses generated text for visual prompts, draws them on the corresponding
    image tensor, and saves the result.
    
    Args:
        image_tensor: Image tensor in CHW format
        generated_text: Text containing <answer> blocks with visual prompts JSON
        output_filename: Filename to save the visualization
        
    Returns:
        RGB numpy array of the annotated image, or None if failed
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    json_str = _extract_second_answer(generated_text)
    if not json_str: return None

    try:
        visual_prompts = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse visual prompts from text: {json_str}")
        return None

    drawable_image = _convert_tensor_to_drawable_image(image_tensor)
    image_with_prompts = _draw_prompts(drawable_image, visual_prompts)
    
    output_path = OUTPUT_DIR / output_filename
    cv2.imwrite(str(output_path), image_with_prompts)
    print(f"Visualization saved to: {output_path}")
    return cv2.cvtColor(image_with_prompts, cv2.COLOR_BGR2RGB)
