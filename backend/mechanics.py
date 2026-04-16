
import math
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageOps
import io
import numpy as np
# --- Constants ---
TARGET_PIXEL_WIDTH = 1000.0
REAL_WORLD_WIDTH = 2.0  # meters
SCALE_FACTOR = REAL_WORLD_WIDTH / TARGET_PIXEL_WIDTH
F_FORCE = 10.0  # Newtons
PI = math.pi

class Block:
    def __init__(self, mass: float, x: float, y: float):
        self.mass = mass
        self.x = x
        self.y = y

    def dist_sq(self, other_x: float, other_y: float) -> float:
        return (self.x - other_x)**2 + (self.y - other_y)**2

def calculate_center_of_mass(blocks: List[Block]) -> Tuple[float, float]:
    total_mass = sum(b.mass for b in blocks)
    if total_mass == 0:
        return 0.0, 0.0
    
    sum_x = sum(b.x * b.mass for b in blocks)
    sum_y = sum(b.y * b.mass for b in blocks)
    return sum_x / total_mass, sum_y / total_mass

def calculate_total_mass(blocks: List[Block]) -> float:
    return sum(b.mass for b in blocks)

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_distance_sq(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def calculate_moment_of_inertia(center: Tuple[float, float], blocks: List[Block]) -> float:
    cx, cy = center
    total_inertia = 0.0
    for b in blocks:
        dist_sq = (b.x - cx)**2 + (b.y - cy)**2
        total_inertia += b.mass * dist_sq
    return total_inertia

def calculate_composite_center_of_mass(
    m_rifle: float, p_G_rifle: Tuple[float, float],
    m_ball: float, p_G_ball: Tuple[float, float]
) -> Tuple[float, float]:
    total_m = m_rifle + m_ball
    if total_m == 0:
        return 0.0, 0.0
    
    nx = p_G_rifle[0] * m_rifle + p_G_ball[0] * m_ball
    ny = p_G_rifle[1] * m_rifle + p_G_ball[1] * m_ball
    return nx / total_m, ny / total_m

def calculate_total_composite_moment_of_inertia(
    I_rifle_G: float, m_rifle: float, p_G_rifle: Tuple[float, float],
    I_ball_G: float, m_ball: float, p_G_ball: Tuple[float, float],
    p_G_composite: Tuple[float, float]
) -> float:
    d_rifle_sq = calculate_distance_sq(p_G_rifle, p_G_composite)
    I_Rifle = I_rifle_G + m_rifle * d_rifle_sq
    
    d_ball_sq = calculate_distance_sq(p_G_ball, p_G_composite)
    I_ball_prime = I_ball_G + m_ball * d_ball_sq
    
    return I_Rifle + I_ball_prime

# --- Energy & Kinematics ---
def calculate_total_energy(x: float) -> float:
    # Polynomial approximation from C++ source
    return 6.96 + (-266.0 * x) + (3037.0 * x**2) + (-8386.0 * x**3) + (7041.0 * x**4)

def calculate_translational_ratio(x: float) -> float:
    # Polynomial approximation from C++ source
    return 0.993 + (-3.47 * x) + (2.45 * x**2) + (29.7 * x**3) + (-42.3 * x**4)

def calculate_spin(omega: float, t: float) -> float:
    return omega * t / (2.0 * PI)

def generate_blocks_from_image_original(
    image_bytes: bytes,
    threshold: int = 100,
    block_size: int = 10, # Note: Not strictly used in pixel-scan loop logic as ported, effectively 1
    target_mass_rifle: float = 0.8
) -> Tuple[List[Block], int, int]:
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize logic
    w, h = img.size
    original_ratio = h / w
    new_w = int(TARGET_PIXEL_WIDTH)
    new_h = int(TARGET_PIXEL_WIDTH * original_ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    pixels = img.load()
    temp_blocks = []
    total_mass_pixels = 0
    
    # Analyze pixels
    # C++ logic: for (int by = 0; by < h; by++) ...
    # And inverted Y coordinate: float inverted_y = (float)h - by;
    
    for y in range(new_h):
        for x in range(new_w):
            r, g, b = pixels[x, y]
            brightness = (r + g + b) / 3.0
            
            if brightness < threshold:
                mass_val = 1.0 # simplified from blockMassCount
                total_mass_pixels += mass_val
                
                # Invert Y for physics coordinate system (Y-up)
                inverted_y = float(new_h - y)
                
                b_obj = Block(
                    mass=mass_val,
                    x=x * SCALE_FACTOR,
                    y=inverted_y * SCALE_FACTOR
                )
                temp_blocks.append(b_obj)
                
    # Normalize mass
    final_blocks = []
    if total_mass_pixels > 0:
        mass_per_pixel = target_mass_rifle / total_mass_pixels
        for b in temp_blocks:
            b.mass *= mass_per_pixel
            final_blocks.append(b)
            
    return final_blocks, new_w, new_h

def run_optimization_original(
    image_bytes: bytes,
    handle_x_px: float,
    handle_y_px: float, 
    target_mass_rifle: float = 0.8,
    m_ball: float = 0.2
) -> Dict:
    
    # 1. Generate blocks
    # Note: Optimization could be slow with full pixel resolution (1000xWidth).
    # For a web response, we might want to downsample further if it times out, 
    # but let's try faithfulness to C++ first.
    
    blocks, width, height = generate_blocks_from_image_original(
        image_bytes, 
        target_mass_rifle=target_mass_rifle
    )
    
    if not blocks:
        return {"error": "No blocks generated"}

    # Convert normalized handle to physics coordinates
    # C++: p_handle.x = x * SCALE_FACTOR;
    #      p_handle.y = (h - y) * SCALE_FACTOR;
    
    # Input handle_x, handle_y are expected to be 0.0-1.0
    px_x = handle_x_px * width
    px_y = handle_y_px * height
    
    p_handle = (
        px_x * SCALE_FACTOR,
        (height - px_y) * SCALE_FACTOR
    )
    
    # 2. Setup Physics Constants
    m_rifle = calculate_total_mass(blocks)
    p_G_rifle = calculate_center_of_mass(blocks)
    I_rifle_G = calculate_moment_of_inertia(p_G_rifle, blocks)
    
    # m_ball passed as argument
    r_ball = 0.03
    I_ball_G = 0.4 * m_ball * (r_ball**2) # 2/5 * m * r^2
    m_composite = m_rifle + m_ball
    
    # ===============================================
    # 1. Original Analysis (Rifle Only, scaled to 1.0kg)
    # ===============================================
    original_mass_target = m_rifle + m_ball # 0.9 + 0.1 = 1.0 (approx)
    scale_ratio = original_mass_target / m_rifle if m_rifle > 0 else 1.0
    
    m_rifle_original = m_rifle * scale_ratio
    I_rifle_original = I_rifle_G * scale_ratio
    
    original_d = calculate_distance(p_handle, p_G_rifle)
    original_E_total = calculate_total_energy(original_d)
    original_ratio_trans = calculate_translational_ratio(original_d)
    
    E_trans_orig = original_E_total * original_ratio_trans
    E_rot_orig = original_E_total - E_trans_orig
    
    orig_v0 = 0.0
    orig_t = 0.0
    if m_rifle_original > 0:
        orig_v0 = math.sqrt(2.0 * max(0.0, E_trans_orig) / m_rifle_original)
        orig_t = 2.0 * orig_v0 / 9.81
        
    orig_omega = 0.0
    if I_rifle_original > 0:
        orig_omega = math.sqrt(2.0 * max(0.0, E_rot_orig) / I_rifle_original)
        
    original_spin = calculate_spin(orig_omega, orig_t)
    
    # ===============================================
    # 2. Optimization (Full Search)
    # ===============================================
    max_spin = -1.0
    optimal_stats = {}
    
    # optimization: iterate with a step to avoid O(N) where N is pixels (could be 500k+)
    # We can skip blocks for speed. Step of 10-20 is reasonable.
    step = 5 # Reduced from 20 for better precision
    
    print(f"DEBUG: Handle Pixels: ({px_x}, {px_y})")
    print(f"DEBUG: Physics Handle: {p_handle}")
    print(f"DEBUG: Rifle Mass: {m_rifle}, Ball Mass: {m_ball}")
    
    for i in range(0, len(blocks), step):
        candidate = blocks[i]
        
        current_p_ball = (candidate.x, candidate.y)
        
        current_p_composite = calculate_composite_center_of_mass(
            m_rifle, p_G_rifle, m_ball, current_p_ball
        )
        
        current_I_star = calculate_total_composite_moment_of_inertia(
            I_rifle_G, m_rifle, p_G_rifle, 
            I_ball_G, m_ball, current_p_ball, 
            current_p_composite
        )
        
        d = calculate_distance(p_handle, current_p_composite)
        
        E_total = calculate_total_energy(d)
        ratio_trans = calculate_translational_ratio(d)
        
        E_trans = E_total * ratio_trans
        E_rot = E_total - E_trans
        
        curr_v0 = 0.0
        curr_t = 0.0
        if m_composite > 0:
            curr_v0 = math.sqrt(2.0 * max(0.0, E_trans) / m_composite)
            curr_t = 2.0 * curr_v0 / 9.81
            
        curr_w = 0.0
        if current_I_star > 0:
            curr_w = math.sqrt(2.0 * max(0.0, E_rot) / current_I_star)
            
        curr_spin = calculate_spin(curr_w, curr_t)
        
        if curr_spin > max_spin:
            max_spin = curr_spin
            
            # Convert back to pixel coordinates (normalized 0-1) for response
            # x = p.x / SCALE_FACTOR
            # y = p.y / SCALE_FACTOR (inverted)
            
            ball_px_x = current_p_ball[0] / SCALE_FACTOR
            ball_px_y = height - (current_p_ball[1] / SCALE_FACTOR)
            
            comp_px_x = current_p_composite[0] / SCALE_FACTOR
            comp_px_y = height - (current_p_composite[1] / SCALE_FACTOR)
            
            optimal_stats = {
                "max_spin": max_spin,
                "ball_pos": {"x": ball_px_x / width, "y": ball_px_y / height},
                "com_pos": {"x": comp_px_x / width, "y": comp_px_y / height},
                "v0": curr_v0,
                "omega": curr_w,
                "E_total": E_total,
                "ratio_trans": ratio_trans,
                "inertia": current_I_star,
                "dist": d
            }
            
    # Rifle COM pixel coords
    rifle_px_x = p_G_rifle[0] / SCALE_FACTOR
    rifle_px_y = height - (p_G_rifle[1] / SCALE_FACTOR)
    
    result = {
        "original": {
            "spin": original_spin,
            "v0": orig_v0,
            "omega": orig_omega,
            "E_total": original_E_total,
            "dist": original_d,
            "com_pos": {"x": rifle_px_x / width, "y": rifle_px_y / height}
        },
        "optimized": optimal_stats,
        "meta": {
            "width": width,
            "height": height,
            "real_width": REAL_WORLD_WIDTH
        }
    }
    
    return result

def generate_blocks_from_image(
    image_bytes: bytes,
    threshold: int = 100,
    block_size: int = 10,
    target_mass_rifle: float = 0.8
) -> Tuple[np.ndarray, int, int]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    original_ratio = h / w
    new_w = int(TARGET_PIXEL_WIDTH)
    new_h = int(TARGET_PIXEL_WIDTH * original_ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    pixels = np.array(img, dtype=float)
    brightness = np.mean(pixels, axis=2)
    
    y_indices, x_indices = np.where(brightness < threshold)
    inverted_y = float(new_h) - y_indices
    
    x_coords = x_indices * SCALE_FACTOR
    y_coords = inverted_y * SCALE_FACTOR
    
    num_blocks = len(x_coords)
    if num_blocks > 0:
        mass_per_pixel = target_mass_rifle / num_blocks
        masses = np.full(num_blocks, mass_per_pixel)
        blocks_np = np.column_stack((masses, x_coords, y_coords))
    else:
        blocks_np = np.empty((0, 3))
        
    return blocks_np, new_w, new_h

def run_optimization(
    image_bytes: bytes,
    handle_x_px: float,
    handle_y_px: float, 
    target_mass_rifle: float = 0.8,
    m_ball: float = 0.2
) -> Dict:
    
    blocks_np, width, height = generate_blocks_from_image(
        image_bytes, 
        target_mass_rifle=target_mass_rifle
    )
    
    if len(blocks_np) == 0:
        return {"error": "No blocks generated"}

    px_x = handle_x_px * width
    px_y = handle_y_px * height
    p_handle = (px_x * SCALE_FACTOR, (height - px_y) * SCALE_FACTOR)
    
    masses = blocks_np[:, 0]
    xs = blocks_np[:, 1]
    ys = blocks_np[:, 2]
    
    m_rifle = np.sum(masses)
    if m_rifle == 0:
        return {"error": "Rifle mass is zero"}
        
    p_G_rifle_x = np.sum(xs * masses) / m_rifle
    p_G_rifle_y = np.sum(ys * masses) / m_rifle
    p_G_rifle = (p_G_rifle_x, p_G_rifle_y)
    
    dist_sq = (xs - p_G_rifle_x)**2 + (ys - p_G_rifle_y)**2
    I_rifle_G = np.sum(masses * dist_sq)
    
    r_ball = 0.03
    I_ball_G = 0.4 * m_ball * (r_ball**2)
    m_composite = m_rifle + m_ball
    
    original_mass_target = m_rifle + m_ball
    scale_ratio = original_mass_target / m_rifle if m_rifle > 0 else 1.0
    m_rifle_original = m_rifle * scale_ratio
    I_rifle_original = I_rifle_G * scale_ratio
    
    original_d = math.sqrt((p_handle[0] - p_G_rifle_x)**2 + (p_handle[1] - p_G_rifle_y)**2)
    original_E_total = calculate_total_energy(original_d)
    original_ratio_trans = calculate_translational_ratio(original_d)
    
    E_trans_orig = original_E_total * original_ratio_trans
    E_rot_orig = original_E_total - E_trans_orig
    
    orig_v0 = math.sqrt(2.0 * max(0.0, E_trans_orig) / m_rifle_original) if m_rifle_original > 0 else 0.0
    orig_t = 2.0 * orig_v0 / 9.81
    orig_omega = math.sqrt(2.0 * max(0.0, E_rot_orig) / I_rifle_original) if I_rifle_original > 0 else 0.0
    original_spin = calculate_spin(orig_omega, orig_t)
    
    step = 5
    cand_xs = xs[::step]
    cand_ys = ys[::step]
    
    comp_x = (m_rifle * p_G_rifle_x + m_ball * cand_xs) / m_composite
    comp_y = (m_rifle * p_G_rifle_y + m_ball * cand_ys) / m_composite
    
    d_rifle_sq = (p_G_rifle_x - comp_x)**2 + (p_G_rifle_y - comp_y)**2
    I_Rifle = I_rifle_G + m_rifle * d_rifle_sq
    
    d_ball_sq = (cand_xs - comp_x)**2 + (cand_ys - comp_y)**2
    I_ball_prime = I_ball_G + m_ball * d_ball_sq
    I_star = I_Rifle + I_ball_prime
    
    dist_handle = np.sqrt((p_handle[0] - comp_x)**2 + (p_handle[1] - comp_y)**2)
    
    E_total = 6.96 + dist_handle * (-266.0 + dist_handle * (3037.0 + dist_handle * (-8386.0 + dist_handle * 7041.0)))
    ratio_trans = 0.993 + dist_handle * (-3.47 + dist_handle * (2.45 + dist_handle * (29.7 + dist_handle * -42.3)))
    
    E_trans = E_total * ratio_trans
    E_rot = E_total - E_trans
    
    E_trans_safe = np.maximum(0.0, E_trans)
    E_rot_safe = np.maximum(0.0, E_rot)
    
    curr_v0 = np.sqrt(2.0 * E_trans_safe / m_composite)
    curr_t = 2.0 * curr_v0 / 9.81
    curr_w = np.sqrt(2.0 * E_rot_safe / I_star)
    
    curr_spin = curr_w * curr_t / (2.0 * PI)
    
    best_idx = np.argmax(curr_spin)
    max_spin = float(curr_spin[best_idx])
    
    ball_px_x = float(cand_xs[best_idx]) / SCALE_FACTOR
    ball_px_y = height - (float(cand_ys[best_idx]) / SCALE_FACTOR)
    
    comp_px_x = float(comp_x[best_idx]) / SCALE_FACTOR
    comp_px_y = height - (float(comp_y[best_idx]) / SCALE_FACTOR)
    
    optimal_stats = {
        "max_spin": max_spin,
        "ball_pos": {"x": ball_px_x / width, "y": ball_px_y / height},
        "com_pos": {"x": comp_px_x / width, "y": comp_px_y / height},
        "v0": float(curr_v0[best_idx]),
        "omega": float(curr_w[best_idx]),
        "E_total": float(E_total[best_idx]),
        "ratio_trans": float(ratio_trans[best_idx]),
        "inertia": float(I_star[best_idx]),
        "dist": float(dist_handle[best_idx])
    }
    
    rifle_px_x = p_G_rifle_x / SCALE_FACTOR
    rifle_px_y = height - (p_G_rifle_y / SCALE_FACTOR)
    
    result = {
        "original": {
            "spin": original_spin,
            "v0": orig_v0,
            "omega": orig_omega,
            "E_total": original_E_total,
            "dist": original_d,
            "com_pos": {"x": rifle_px_x / width, "y": rifle_px_y / height}
        },
        "optimized": optimal_stats,
        "meta": {
            "width": width,
            "height": height,
            "real_width": REAL_WORLD_WIDTH
        }
    }
    
    return result
