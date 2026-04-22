import json
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointData:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def to_dict(self):
        return {"x": self.x, "y": self.y}

class ContourExtractor:
    def __init__(self):
        self.contour_points: List[PointData] = []
        self.hole_points_list: List[List[PointData]] = []
        self.step = 10.0
        self.scale_factor = 500.0

    def process_image(self, image_path: str, output_path: str = "points.json"):
        """
        Main pipeline: Load -> Resize -> Extract -> Save
        """
        try:
            # 1. Load and Resize
            original_img = Image.open(image_path).convert('RGB')
            w, h = original_img.size
            aspect_ratio = h / w
            new_width = 1000
            new_height = int(new_width * aspect_ratio)
            
            resized_img = original_img.resize((new_width, new_height))
            logger.info(f"Image resized to: {new_width} x {new_height}")
            
            # 2. Extract Contour
            # self.extract_contour_custom_original(resized_img) # 元のアルゴリズム
            self.extract_contour_opencv(resized_img) # OpenCV版
            
            # 3. Save to JSON
            self.save_to_json(output_path)
            return True
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return False

    def extract_contour_opencv(self, img: Image.Image):
        """
        OpenCVを使用して輪郭を抽出する高速版。
        元のextract_contour_custom_originalと同じく外周を辿るアルゴリズム（鈴木のアルゴリズム）を使用します。
        """
        import cv2
        w, h = img.size
        pixels = np.array(img)
        
        detect_dark_object = True
        threshold_val = 128
        
        # グレースケール変換
        if len(pixels.shape) == 3:
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        else:
            gray = pixels
            
        # 2値化マスクの作成
        if detect_dark_object:
            _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
            
        # 輪郭の検索 (階層構造を含めて取得)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        self.contour_points = []
        self.hole_points_list = []
        if contours:
            # 最も面積の大きい輪郭のインデックスを選択（外形）
            largest_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
            largest_contour = contours[largest_contour_idx]
            
            # OpenCVは (N, 1, 2) の形式で座標を返すため PointData のリストに変換
            for pt in largest_contour:
                x, y = pt[0]
                self.contour_points.append(PointData(int(x), int(y)))
                
            # 穴（子輪郭）の検索: 親が largest_contour_idx である輪郭を探す
            if hierarchy is not None:
                for i, cnt in enumerate(contours):
                    parent_idx = hierarchy[0][i][3]
                    if parent_idx == largest_contour_idx:
                        # 小さすぎるノイズの穴を除外
                        if cv2.contourArea(cnt) > 20: 
                            hole_pts = []
                            for pt in cnt:
                                x, y = pt[0]
                                hole_pts.append(PointData(int(x), int(y)))
                            self.hole_points_list.append(hole_pts)
                
            logger.info(f"OpenCV found main contour with {len(self.contour_points)} points and {len(self.hole_points_list)} holes.")
        else:
            logger.error("No contour found by OpenCV!")

    def extract_contour_custom_original(self, img: Image.Image):
        w, h = img.size
        # Convert to numpy array for fast pixel access
        pixels = np.array(img) # Shape (h, w, 3)
        
        detect_dark_object = True
        threshold = 128
        
        # Calculate average per pixel
        # Axis 2 is color channel (R, G, B)
        avg_pixels = np.mean(pixels, axis=2) # Shape (h, w)
        
        if detect_dark_object:
            is_object = avg_pixels < threshold
        else:
            is_object = avg_pixels >= threshold
            
        # is_object is a boolean mask (h, w)
        # Flattened version for logic parity if needed, but 2D access is easier in Python
        # The C++ code uses 1D vector isObject[y*w + x]
        
        visited = np.zeros((h, w), dtype=bool)
        
        # 8-neighbor definition (CCW)
        # 0:Right, 1:Right-Up, 2:Up, 3:Left-Up, 4:Left, 5:Left-Down, 6:Down, 7:Right-Down
        # Note: In C++ OF, Y is down (positive).
        # dx:  1,  1,  0, -1, -1, -1,  0,  1
        # dy:  0, -1, -1, -1,  0,  1,  1,  1
        dx = [1,  1,  0, -1, -1, -1,  0,  1]
        dy = [0, -1, -1, -1,  0,  1,  1,  1]
        
        self.contour_points = []
        found_big_contour = False
        
        for y in range(h):
            for x in range(w):
                if is_object[y, x] and not visited[y, x]:
                    
                    temp_contour = []
                    start_x, start_y = x, y
                    
                    temp_contour.append(PointData(start_x, start_y))
                    visited[y, x] = True
                    
                    current_x, current_y = start_x, start_y
                    # enterFrom means where did we come from? 
                    # 4 means we entered from LEFT? No, 4 is Left direction.
                    # C++: enterFrom = 4; (which is Left direction)
                    enter_from = 4 
                    
                    max_steps = w * h * 2
                    steps = 0
                    closed_loop = False
                    
                    while steps < max_steps:
                        moved = False
                        
                        # Logic: checkStartDir = (enterFrom + 6) % 8
                        # "Slightly behind where we came from"
                        check_start_dir = (enter_from + 6) % 8
                        
                        # C++: Pre-check "previous" to implement "edge following" logic correctly?
                        # The C++ code checks prevIsObj and currIsObj transitions.
                        
                        prev_dir = (check_start_dir + 7) % 8
                        prev_nx = current_x + dx[prev_dir]
                        prev_ny = current_y + dy[prev_dir]
                        
                        prev_is_obj = False
                        if 0 <= prev_nx < w and 0 <= prev_ny < h:
                            prev_is_obj = is_object[prev_ny, prev_nx]
                            
                        for i in range(8):
                            curr_dir = (check_start_dir + i) % 8
                            curr_nx = current_x + dx[curr_dir]
                            curr_ny = current_y + dy[curr_dir]
                            
                            curr_is_obj = False
                            if 0 <= curr_nx < w and 0 <= curr_ny < h:
                                curr_is_obj = is_object[curr_ny, curr_nx]
                                
                            if not prev_is_obj and curr_is_obj:
                                # Found boundary transition
                                current_x = curr_nx
                                current_y = curr_ny
                                temp_contour.append(PointData(current_x, current_y))
                                
                                # Current direction is curr_dir via dx/dy
                                # Enter from is opposite? 
                                # C++: enterFrom = (currDir + 4) % 8
                                enter_from = (curr_dir + 4) % 8
                                moved = True
                                break
                            
                            prev_is_obj = curr_is_obj
                        
                        if not moved:
                            break
                        
                        if current_x == start_x and current_y == start_y:
                            closed_loop = True
                            break
                        
                        steps += 1
                    
                    if closed_loop and len(temp_contour) > 50:
                        self.contour_points = temp_contour
                        logger.info(f"Main object found (CCW)! Points: {len(self.contour_points)}")
                        found_big_contour = True
                        break
            
            if found_big_contour:
                break
                
        if not self.contour_points:
            logger.error("No contour found!")

    def save_to_json(self, output_path: str):
        if not self.contour_points:
            return
            
        def sample_points(points):
            sampled = []
            limit = len(points)
            i = 0.0
            while i < limit:
                idx = int(i)
                if idx >= limit:
                    break
                p = points[idx]
                sampled.append({
                    "x": float(p.x) / self.scale_factor,
                    "y": float(p.y) / self.scale_factor
                })
                i += self.step
            return sampled

        json_data = {}
        json_data["points"] = sample_points(self.contour_points)
        json_data["name"] = "SampleData"
        json_data["holes"] = [sample_points(hole) for hole in self.hole_points_list]
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        logger.info("Saved JSON.")

if __name__ == "__main__":
    import sys
    # Example usage: python contour_logic.py image.png
    if len(sys.argv) > 1:
        extractor = ContourExtractor()
        extractor.process_image(sys.argv[1])
