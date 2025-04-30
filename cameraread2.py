import sys
import serial
import numpy as np
import cv2
import argparse
from collections import deque
import mediapipe as mp

PORT         = 'COM12'
BAUDRATE     = 12500000
WIDTH        = 320
HEIGHT       = 240
FRAME_SZ     = WIDTH * HEIGHT * 2
END_MARKER   = b'\xDE\xAD\xBE\xEF'
REQUEST_BYTE = b'\x01'
INFER_HDR    = b'\xCA\xFE\xBA\xBE'

INPUTS = 64  # must match your Arduino (FFT Bins)

parser = argparse.ArgumentParser(description="Host for On-Device Features SNN")
parser.add_argument("--inc",        type=float,   default=None, help="New increment")
parser.add_argument("--leak",       type=float,   default=None, help="New leak_ratio")
parser.add_argument("--amp-mid",    type=float,   default=1.0,  help="Amplification factor for middle 50% of bins")
parser.add_argument("--alpha",      type=float,   default=1.2,  help="Contrast control (1.0=no change)")
parser.add_argument("--beta",       type=int,     default=30,   help="Brightness control (0=no change)")
parser.add_argument("--video-only", action="store_true",
                    help="Display video only; skip hand detection, serial I/O, and inference")
args = parser.parse_args()

def send_hyperparams(inc, leak):
    cmd = f"HP {inc:.3f} {leak:.3f}\n".encode('ascii')
    ser.write(cmd)
    ack = ser.readline().decode('ascii', errors='ignore').strip()
    print("Arduino ACK:", ack)

def rgb565_to_bgr(raw):
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
    hi  = arr[:,:,0].astype(np.uint16)
    lo  = arr[:,:,1].astype(np.uint16)
    pix = (hi << 8) | lo
    r = ((pix & 0xF800) >> 11) * 255 // 31
    g = ((pix & 0x07E0) >>  5) * 255 // 63
    b = ((pix & 0x001F) >>   0) * 255 // 31
    return np.dstack((b, g, r)).astype(np.uint8)

def detect_hand(bgr, hands_net):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = hands_net.process(rgb)
    if not results.multi_hand_landmarks:
        return None, None, None

    h_img, w_img, _ = bgr.shape
    lm = results.multi_hand_landmarks[0].landmark
    xs = [int(pt.x * w_img) for pt in lm]
    ys = [int(pt.y * h_img) for pt in lm]
    x1, x2 = max(0, min(xs)), min(w_img, max(xs))
    y1, y2 = max(0, min(ys)), min(h_img, max(ys))
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)

    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hand_gray = gray_full[y1:y2, x1:x2].astype(np.float32)
    mask_crop = np.ones_like(hand_gray, dtype=np.uint8) * 255
    return hand_gray, mask_crop, (x1, y1, x2 - x1, y2 - y1)

def fourier_descriptor(mask_crop, n_coeffs):
    cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No contour in mask_crop")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1,2)
    z = cnt[:,0].astype(np.float32) + 1j*cnt[:,1].astype(np.float32)
    z -= np.mean(z)
    Z = np.fft.fft(z)
    mags = np.abs(Z[1:1+n_coeffs]).astype(np.float32)
    if mags.shape[0] < n_coeffs:
        mags = np.pad(mags, (0, n_coeffs - mags.shape[0]))
    else:
        mags = mags[:n_coeffs]
    mn, mx = mags.min(), mags.max()
    if mx - mn > 1e-12:
        mags = (mags - mn) / (mx - mn)
    else:
        mags[:] = 0.0
    return mags

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

def main():
    mp_hands  = mp.solutions.hands
    hands_net = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if args.inc is not None and args.leak is not None and not args.video_only:
        send_hyperparams(args.inc, args.leak)

    if args.video_only:
        print(">>> VIDEO-ONLY: skipping hand detection, serial I/O, and inference <<<")

    cv2.namedWindow('CAMERA', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CAMERA', 1920, 1080)

    # Buffers & indices
    buf_g0, buf_g1 = deque(maxlen=5), deque(maxlen=5)
    bins_buffer    = deque(maxlen=1)
    start, end     = INPUTS//4, INPUTS*3//4

    print("Press 'q' to exit")
    while True:
        ser.reset_input_buffer()
        ser.write(REQUEST_BYTE)
        raw = ser.read(FRAME_SZ)
        if len(raw) < FRAME_SZ:
            raise RuntimeError(f"Frame timeout: got {len(raw)}/{FRAME_SZ} bytes")
        if ser.read(len(END_MARKER)) != END_MARKER:
            raise RuntimeError("Bad end marker")

        bgr = rgb565_to_bgr(raw)
        bgr = cv2.convertScaleAbs(bgr, alpha=args.alpha, beta=args.beta)

        up = cv2.flip(cv2.resize(bgr, (1920,1080), cv2.INTER_CUBIC), 0)

        if args.video_only:
            cv2.imshow('CAMERA', up)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        hand, mask, rect = detect_hand(bgr, hands_net)

        if hand is not None:
            bins = fourier_descriptor(mask, INPUTS)
            bins_buffer.append(bins)
            bins = np.mean(bins_buffer, axis=0).astype(np.float32)
            bins[start:end] *= args.amp_mid
            np.clip(bins, 0.0, 1.0, out=bins)
            ser.write(("BINS:" + ",".join(f"{b:.3f}" for b in bins) + "\n").encode('ascii'))

            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line.startswith("GROUPS:"):
                raise RuntimeError(f"Expected GROUPS:, got {line!r}")
            g0, g1 = map(int, line.replace("GROUPS:", "").split(','))
            buf_g0.append(g0); buf_g1.append(g1)
            avg_g0 = sum(buf_g0)/len(buf_g0)
            avg_g1 = sum(buf_g1)/len(buf_g1)
            pred   = 0 if avg_g0>avg_g1 else 1
            ser.read(len(INFER_HDR)); ser.read(1)

            x,y,w,h = rect
            fx, fy  = 1920/WIDTH, 1080/HEIGHT
            sx, sy  = int(x*fx), int(y*fy)
            sw, sh  = int(w*fx), int(h*fy)
            sy       = 1080 - sy - sh
            cv2.rectangle(up, (sx,sy), (sx+sw, sy+sh), (0,255,0), 2)

            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            cv2.putText(up, f"Predicted: {pred}", (up.shape[1]-300,30), font, fs, (0,255,0), th)
            cv2.putText(up, f"Groups: {g0},{g1}", (10, up.shape[0]-30), font, 0.8, (0,255,255), 2)
            cv2.putText(up, f"Avg g0: {avg_g0:.1f}", (10, up.shape[0]-60), font, 0.8, (0,200,255), 2)
            cv2.putText(up, f"Avg g1: {avg_g1:.1f}", (10, up.shape[0]-90), font, 0.8, (0,200,255), 2)
        else:
            cv2.putText(up, "No hand detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow('CAMERA', up)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands_net.close()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
