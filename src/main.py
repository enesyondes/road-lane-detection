import cv2, time, argparse
import numpy as np

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = 255 if len(image.shape) == 2 else (255,) * image.shape[2]
    cv2.fillPoly(mask, [vertices], match_mask_color)
    return cv2.bitwise_and(image, mask)

def draw_poly(img, poly, y_min, y_max, color=(0,255,0), thickness=6):
    if poly is None: return img
    ys = np.linspace(y_min, y_max, 20)
    xs = np.polyval(poly, ys)
    points = np.array([(int(x), int(y)) for x, y in zip(xs, ys)], np.int32)
    cv2.polylines(img, [points], False, color, thickness)
    return img

def split_and_fit(lines, img_h, img_w, fit_degree=1):
    left, right = [], []
    if lines is None: return None, None
    
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1: 
            continue
        
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        abs_slope = abs(slope)
        
        # Gürültü filtresi (yatay ve dikey gürültüleri çıkar)
        if abs_slope < 0.3 or abs_slope > 0.9:
            continue
            
        midpoint_x = (x1 + x2) / 2
        (left if midpoint_x < img_w/2 else right).append((x1, y1, x2, y2))
    
    def fit_lane(segments):
        if not segments: return None
        x_vals, y_vals = [], []
        for seg in segments:
            x_vals.extend([seg[0], seg[2]])
            y_vals.extend([seg[1], seg[3]])
        if len(x_vals) < 3:
            return None
        try:
            return np.polyfit(y_vals, x_vals, fit_degree)
        except:
            return None

    return fit_lane(left), fit_lane(right)

class EMALane:
    def __init__(self, alpha=0.3):
        self.left_poly = None
        self.right_poly = None
        self.alpha = alpha
        self.miss_count = 0
    
    def _ema_poly(self, prev, curr):
        if prev is None:
            return np.array(curr)
        return (1 - self.alpha) * np.array(prev) + self.alpha * np.array(curr)
    
    def update(self, left_poly, right_poly):
        if left_poly is not None:
            self.left_poly = self._ema_poly(self.left_poly, left_poly)
            self.miss_count = 0
        elif self.miss_count < 5:
            self.miss_count += 1
        else:
            self.left_poly = None
            
        if right_poly is not None:
            self.right_poly = self._ema_poly(self.right_poly, right_poly)
            self.miss_count = 0
        elif self.miss_count < 5:
            self.miss_count += 1
        else:
            self.right_poly = None
            
        return self.left_poly, self.right_poly

def process(frame, 
            fit_degree=1, 
            canny_low=50, 
            canny_high=150, 
            hough_thresh=30, 
            min_len=25, 
            max_gap=100):
    h, w = frame.shape[:2]
    
    roi_top_width = w * 0.7
    roi_bottom_width = w * 0.9
    roi_vertices = np.array([[
        (int((w - roi_bottom_width)/2), h),
        (int((w - roi_top_width)/2), int(h*0.6)),  # Daha aşağıdan başlatıldı
        (int((w + roi_top_width)/2), int(h*0.6)),
        (int((w + roi_bottom_width)/2), h)
    ]], dtype=np.int32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,40,255]))
    yellow_mask = cv2.inRange(hsv, np.array([15,80,80]), np.array([35,255,255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    edges = cv2.bitwise_and(edges, color_mask)
    
    cropped = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(
        cropped, 
        rho=1, 
        theta=np.pi/180, 
        threshold=hough_thresh,
        minLineLength=min_len, 
        maxLineGap=max_gap
    )
    
    L, R = split_and_fit(lines, h, w, fit_degree)
    return L, R, roi_vertices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="yol1.mp4")
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=1, help="Fit degree: 1 or 2")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Video açılamadı: {args.video}")
        return

    ema = EMALane(alpha=0.3)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)
        
        L_poly, R_poly, roi_vertices = process(frame, fit_degree=args.degree)
        L_smoothed, R_smoothed = ema.update(L_poly, R_poly)
        
        vis = frame.copy()
        roi_overlay = np.zeros_like(frame)
        cv2.fillPoly(roi_overlay, [roi_vertices], (0, 150, 255))
        vis = cv2.addWeighted(vis, 0.8, roi_overlay, 0.2, 0)
        
        h, w = vis.shape[:2]
        y_min = int(h * 0.6)
        y_max = h
        
        if L_smoothed is not None:
            vis = draw_poly(vis, L_smoothed, y_min, y_max, (0, 0, 255), 8)
        if R_smoothed is not None:
            vis = draw_poly(vis, R_smoothed, y_min, y_max, (0, 0, 255), 8)
        if L_smoothed is not None and R_smoothed is not None:
            center_poly = (L_smoothed + R_smoothed) / 2
            vis = draw_poly(vis, center_poly, y_min, y_max, (0, 255, 255), 4)
        
        cv2.imshow("Şerit Takibi", vis)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
