import cv2                                                  # OpenCV — bibliotek för bild- och videohantering
import numpy as np                                          # NumPy — för matematik och arrays
import time                                                 # för tidmätning (fps, hastighet, etc.)

# ─── Config ────────────────────────────────────────────────────────────────────
RTSP_URL    = "rtsp://192.168.10.208:8554/live.sdp"         # adress till kamerans videoström
PLANE_LEN_M = 0.19                                          # planets verkliga längd i meter (19 cm)
MIN_AREA    = 500                                           # minsta tillåtna konturyta i pixlar² (filtrerar brus)
MIN_ASPECT  = 1.8                                           # minsta tillåtna förhållande längd/bredd (avvisar plan rakt framifrån)
LOST_TIMEOUT = 0.35                                         # antal sekunder utan plan innan spårningen nollställs

ALPHA = dict(depth=0.25, speed=0.35, pos=0.40, length=0.30) # utjämningsfaktorer för EMA-filter (lågt = stabilt, högt = snabbt)

# ─── State ─────────────────────────────────────────────────────────────────────
fx = fy = 700.0                                             # fokallängd i pixlar (kamerans "zoom")

ema       = dict(depth=0.0, speed=0.0, length=0.0, cx=None, cy=None)  # senaste utjämnade värden för djup, hastighet, längd, position
prev      = dict(cx=None, cy=None, t=time.perf_counter())   # förra framens centroid och tidsstämpel (används för hastighet)
fps       = dict(count=0, value=0.0, t=time.perf_counter()) # frames-per-second-räknare
last_seen = time.perf_counter()                             # tidsstämpel för senaste lyckade detektion

# ─── Helpers ───────────────────────────────────────────────────────────────────

# Ritar ut en lista av textrader på bilden vid givna y-koordinater (gul, fast typsnitt).
def overlay(img, items):
    """Draw [(text, y), ...] onto img."""
    for text, y in items:                                   # går igenom varje (text, y-position)-par
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # ritar texten på bilden

# Beräknar konturens längsta axel i pixlar med PCA — stabilare än minAreaRect som ofta flippar bredd/höjd.
def major_axis_px(cnt):
    """PCA-based major-axis length in pixels (stable vs minAreaRect)."""
    pts = cnt.reshape(-1, 2).astype(np.float32)             # gör om konturen till en lista av (x, y)-punkter
    if len(pts) < 10:                                       # om för få punkter
        return 0.0                                          # returnera 0 (kan inte beräkna säkert)
    c = pts - pts.mean(axis=0)                              # centrera punkterna runt origo (drar bort medelvärdet)
    _, vecs = np.linalg.eigh(np.cov(c.T))                   # eigenvektorer av kovariansmatrisen → ger huvudaxelriktningen
    proj = c @ vecs[:, -1]                                  # projicerar alla punkter på huvudaxeln
    return float(proj.max() - proj.min())                   # längden = avståndet mellan största och minsta projektion

# Uppdaterar ett värde i ema-dicten med EMA-formeln: nytt = (1-α)·gammalt + α·mätning.
def ema_update(key, value):
    ema[key] = (1 - ALPHA[key]) * ema[key] + ALPHA[key] * value  # blandar gammalt värde med ny mätning enligt α

# EMA för centroid — hanterar specialfallet att ema är None vid första frame.
def ema_pos(key, value):
    """EMA for centroid — handles first-frame None case."""
    ema[key] = value if ema[key] is None else (1 - ALPHA["pos"]) * ema[key] + ALPHA["pos"] * value  # första gången: sätt direkt, annars EMA

# Nollställer all spårningsdata när planet försvunnit för länge.
def reset_tracking():
    ema.update(depth=0.0, speed=0.0, length=0.0, cx=None, cy=None)  # nollställer alla EMA-värden
    prev.update(cx=None, cy=None)                           # nollställer förra positionen

# ─── Init ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)            # öppnar videoströmmen via FFmpeg-backend
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                         # buffrar bara 1 frame → minimerar latency
if not cap.isOpened():                                      # om kameran inte kunde öppnas
    raise SystemExit("Failed to open RTSP stream")          # avsluta programmet

cv2.namedWindow("Trackbars")                                # skapar fönstret som ska hålla trackbars
for name, default, max_val in [                             # loop: skapar 6 trackbars för HSV-intervall
    ("H Min", 90,  179), ("H Max", 110, 179),               # nyans (Hue): 0–179
    ("S Min", 80,  255), ("S Max", 255, 255),               # mättnad (Saturation): 0–255
    ("V Min", 80,  255), ("V Max", 255, 255),               # ljushet (Value): 0–255
]:
    cv2.createTrackbar(name, "Trackbars", default, max_val, lambda x: None)  # skapar varje trackbar (callback gör inget)

kernel = np.ones((3, 3), np.uint8)                          # 3×3-matris av ettor för morfologi-operationer
tb = lambda n: cv2.getTrackbarPos(n, "Trackbars")           # hjälpfunktion: läser värdet från en trackbar
print("Controls: q=quit")                                   # skriver ut tangentbordsinstruktioner

# ─── Main loop ─────────────────────────────────────────────────────────────────
first_frame = True                                          # flagga: skriv bara ut upplösningen vid första framet
while True:                                                 # huvudloop — körs en gång per frame
    ret, frame = cap.read()                                 # läser nästa frame från strömmen
    if not ret or frame is None:                            # om framen inte kunde läsas
        print("Frame read failed")                          # skriv felmeddelande
        break                                               # avbryt loopen

    now  = time.perf_counter()                              # aktuell tidsstämpel (hög precision)
    h, w = frame.shape[:2]                                  # bildens höjd och bredd i pixlar

    if first_frame:                                         # första gången
        print(f"Stream: {w}x{h}")                           # skriv ut upplösningen
        first_frame = False                                 # sätt flaggan till False så det bara händer en gång

    # HSV mask
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)           # konvertera bilden från BGR till HSV-färgrymd
    mask = cv2.inRange(hsv,                                 # skapa svartvit mask: vit där färgen matchar intervallet
                       np.array([tb("H Min"), tb("S Min"), tb("V Min")], dtype=np.uint8),  # nedre HSV-gräns
                       np.array([tb("H Max"), tb("S Max"), tb("V Max")], dtype=np.uint8))  # övre HSV-gräns
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # tar bort små vita prickar (brus)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fyller små svarta hål inuti vita områden

    # Detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # hittar alla yttre konturer i masken
    detected = False                                        # flagga: blir True om ett giltigt plan hittas

    if contours:                                            # om minst en kontur finns
        best = max(contours, key=cv2.contourArea)           # välj den största konturen (antas vara planet)
        if cv2.contourArea(best) > MIN_AREA:                # om konturen är stor nog
            rect = cv2.minAreaRect(best)                    # minsta roterade rektangeln runt konturen
            (cx_f, cy_f), (rw, rh), _ = rect                # packa upp: centrum (x, y), storlek (bredd, höjd), vinkel
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)     # förhållande längsta/kortaste sidan (+1e-6 mot division med noll)

            if aspect >= MIN_ASPECT:                        # om planet är tillräckligt avlångt (sett från sidan)
                detected  = True                            # markera att planet hittades
                last_seen = now                             # uppdatera "senast sedd"-tiden

                # Smooth centroid
                ema_pos("cx", cx_f)                         # utjämna x-koordinat
                ema_pos("cy", cy_f)                         # utjämna y-koordinat
                cx, cy = int(ema["cx"]), int(ema["cy"])     # konvertera till heltal för ritning

                # Pixel length → depth
                raw_len = major_axis_px(best)               # mät planets längd i pixlar (PCA)
                if raw_len > 1:                             # om mätningen är giltig
                    ema_update("length", raw_len)           # utjämna pixellängden
                if ema["length"] > 1:                       # om utjämnad längd är giltig
                    ema_update("depth", (fx * PLANE_LEN_M) / ema["length"])  # pinhole-formel: djup = (fx · verklig_längd) / pixel_längd

                # Speed
                dt = now - prev["t"]                        # tid sedan förra framen
                if prev["cx"] is not None and ema["depth"] > 0 and dt > 0:  # bara om vi har förra positionen och giltigt djup
                    dx = (cx - prev["cx"]) * ema["depth"] / fx              # x-förflyttning i meter
                    dy = (cy - prev["cy"]) * ema["depth"] / fy              # y-förflyttning i meter
                    ema_update("speed", float(np.hypot(dx, dy)) / dt)       # hastighet = distans / tid, sen utjämnad

                prev.update(cx=cx, cy=cy, t=now)            # spara nuvarande position och tid till nästa frame

                # Draw
                cv2.drawContours(frame, [np.int32(cv2.boxPoints(rect))], 0, (0, 255, 0), 2)  # rita grön rektangel runt planet
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)             # rita röd punkt på centroiden
                overlay(frame, [(f"Aspect: {aspect:.2f}", 150),             # visa aspect ratio som debug-info
                                (f"L_px: {ema['length']:.1f}", 190)])       # visa pixellängd

    if not detected and now - last_seen > LOST_TIMEOUT:     # om planet ej setts på LOST_TIMEOUT sek
        reset_tracking()                                    # nollställ all spårningsdata

    # FPS
    fps["count"] += 1                                       # öka frame-räknaren
    if now - fps["t"] >= 1.0:                               # om en hel sekund har gått
        fps["value"] = fps["count"] / (now - fps["t"])      # beräkna fps = frames / sekunder
        fps["count"] = 0                                    # nollställ räknaren
        fps["t"]     = now                                  # uppdatera referens-tiden

    # HUD
    overlay(frame, [                                        # rita ut alla statusvärden överst på bilden
        (f"RES: {w}x{h}",                    30),           # upplösning
        (f"FPS: {fps['value']:.1f}",          70),          # frames per sekund
        (f"Speed: {ema['speed']:.2f} m/s",   110),          # uppmätt hastighet
        (f"Depth: {ema['depth']:.2f} m",     230),          # uppmätt djup
    ])
    cv2.imshow("Frame", frame)                              # visa huvudbilden i fönstret "Frame"
    cv2.imshow("Mask",  mask)                               # visa masken i fönstret "Mask"

    key = cv2.waitKey(1) & 0xFF                             # vänta 1 ms och läs ev. tangenttryck
    if key == ord('q'):                                     # om 'q' trycks
        break                                               # avbryt loopen

cap.release()                                               # stäng videoströmmen
cv2.destroyAllWindows()                                     # stäng alla OpenCV-fönster
