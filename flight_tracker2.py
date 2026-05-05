import cv2
import numpy as np
import time

# ─── Config ────────────────────────────────────────────────────────────────────
RTSP_URL    = "rtsp://192.168.10.208:8554/live.sdp"
PLANE_LEN_M = 0.19
MIN_AREA    = 500
MIN_ASPECT  = 1.8
LOST_TIMEOUT = 0.35

ALPHA = dict(depth=0.25, speed=0.35, pos=0.40, length=0.30)

# ─── State ─────────────────────────────────────────────────────────────────────
fx = fy = 700.0

ema       = dict(depth=0.0, speed=0.0, length=0.0, cx=None, cy=None)
prev      = dict(cx=None, cy=None, t=time.perf_counter())
fps       = dict(count=0, value=0.0, t=time.perf_counter())
last_seen = time.perf_counter()

# ─── Helpers ───────────────────────────────────────────────────────────────────

# Ritar ut en lista av textrader på bilden vid givna y-koordinater (gul, fast typsnitt).
def overlay(img, items):
    for text, y in items:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# Beräknar konturens längsta axel i pixlar med PCA — stabilare än minAreaRect som ofta flippar bredd/höjd.
def major_axis_px(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 10:
        return 0.0
    c = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(c.T))
    proj = c @ vecs[:, -1]
    return float(proj.max() - proj.min())

# Uppdaterar ett värde i ema-dicten med EMA-formeln: nytt = (1-α)·gammalt + α·mätning.
def ema_update(key, value):
    ema[key] = (1 - ALPHA[key]) * ema[key] + ALPHA[key] * value

# EMA för centroid — hanterar specialfallet att ema är None vid första frame.
def ema_pos(key, value):
    ema[key] = value if ema[key] is None else (1 - ALPHA["pos"]) * ema[key] + ALPHA["pos"] * value

# Nollställer all spårningsdata när planet försvunnit för länge.
def reset_tracking():
    ema.update(depth=0.0, speed=0.0, length=0.0, cx=None, cy=None)
    prev.update(cx=None, cy=None)

# ─── Init ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Failed to open RTSP stream")

cv2.namedWindow("Trackbars")
for name, default, max_val in [
    ("H Min", 90,  179), ("H Max", 110, 179),
    ("S Min", 80,  255), ("S Max", 255, 255),
    ("V Min", 80,  255), ("V Max", 255, 255),
]:
    cv2.createTrackbar(name, "Trackbars", default, max_val, lambda x: None)

kernel = np.ones((3, 3), np.uint8)
tb = lambda n: cv2.getTrackbarPos(n, "Trackbars")
print("Controls: q=quit")

# ─── Main loop ─────────────────────────────────────────────────────────────────
first_frame = True
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame read failed")
        break

    now  = time.perf_counter()
    h, w = frame.shape[:2]

    if first_frame:
        print(f"Stream: {w}x{h}")
        first_frame = False

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([tb("H Min"), tb("S Min"), tb("V Min")], dtype=np.uint8),
                       np.array([tb("H Max"), tb("S Max"), tb("V Max")], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    if contours:
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) > MIN_AREA:
            rect = cv2.minAreaRect(best)
            (cx_f, cy_f), (rw, rh), _ = rect
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)

            if aspect >= MIN_ASPECT:
                detected  = True
                last_seen = now

                ema_pos("cx", cx_f)
                ema_pos("cy", cy_f)
                cx, cy = int(ema["cx"]), int(ema["cy"])

                raw_len = major_axis_px(best)
                if raw_len > 1:
                    ema_update("length", raw_len)
                if ema["length"] > 1:
                    ema_update("depth", (fx * PLANE_LEN_M) / ema["length"])

                dt = now - prev["t"]
                if prev["cx"] is not None and ema["depth"] > 0 and dt > 0:
                    dx = (cx - prev["cx"]) * ema["depth"] / fx
                    dy = (cy - prev["cy"]) * ema["depth"] / fy
                    ema_update("speed", float(np.hypot(dx, dy)) / dt)

                prev.update(cx=cx, cy=cy, t=now)

                cv2.drawContours(frame, [np.int32(cv2.boxPoints(rect))], 0, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                overlay(frame, [(f"Aspect: {aspect:.2f}", 150),
                                (f"L_px: {ema['length']:.1f}", 190)])

    if not detected and now - last_seen > LOST_TIMEOUT:
        reset_tracking()

    fps["count"] += 1
    if now - fps["t"] >= 1.0:
        fps["value"] = fps["count"] / (now - fps["t"])
        fps["count"] = 0
        fps["t"]     = now

    overlay(frame, [
        (f"RES: {w}x{h}",                    30),
        (f"FPS: {fps['value']:.1f}",          70),
        (f"Speed: {ema['speed']:.2f} m/s",   110),
        (f"Depth: {ema['depth']:.2f} m",     230),
    ])
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",  mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
