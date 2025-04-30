import os, sys
import mediapipe as mpp
import cv2
import numpy as np
import glob
import signal
import argparse
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Array, Lock
from numba import njit
import numba
import pickle

parser = argparse.ArgumentParser(
    description="SNN Gestures: W1 pretrain & W2 STDP-only with per-step STDP"
)
parser.add_argument("--w1-file", type=str, default=None,
                    help="Path to save/load pretrained W1 (.csv)")
parser.add_argument("--w2-file", type=str, default=None,
                    help="Path to save/load best model (.npz)")
parser.add_argument("--steps", type=int, default=1,
                    help="Number of simulation time-steps per input sample")
args = parser.parse_args()
num_steps = args.steps

pools = []
def signal_handler(sig, frame):
    print("\nKeyboardInterrupt detected. Terminating all pools...", flush=True)
    for p in pools:
        try:
            p.terminate(); p.join()
        except:
            pass
    os._exit(1)
signal.signal(signal.SIGINT, signal_handler)

def worker_ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

numba.set_num_threads(mp.cpu_count())

n_hidden = 64
n_output = 16
num_labels = 2
neurons_per_group = n_output // num_labels

increment    = 0.1
threshold    = 1.0
v_rest       = 0.0
leak_ratio   = 0.01
leak_amt     = increment * leak_ratio

eta_w2        = 0.0001
target_true   = 200
target_false  = 5
punish_factor = 35

w_min, w_max  = 0, 30
epochs        = 500
exploit_every = 4

sensitivity_init = 0
data_root        = "./"

eta_w1       = 0.05
W1_min_init  = 0.0
W1_max_init  = 30

inputs = n_hidden
num_bins = n_hidden
n_elites    = 6

def detect_hand(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = hands_net.process(rgb)
    if not results.multi_hand_landmarks:
        return None, None
    h, w, _ = bgr.shape
    lm = results.multi_hand_landmarks[0].landmark
    xs = [int(pt.x * w) for pt in lm]
    ys = [int(pt.y * h) for pt in lm]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    pad = int(0.1 * max(x2-x1, y2-y1))
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hand_gray = gray_full[y1:y2, x1:x2].astype(np.float32)
    mask_crop = np.ones_like(hand_gray, dtype=np.uint8)*255
    return hand_gray, mask_crop

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
        mags = np.concatenate([mags, np.zeros(n_coeffs-mags.shape[0],dtype=np.float32)])
    else:
        mags = mags[:n_coeffs]
    mn, mx = mags.min(), mags.max()
    if mx-mn > 1e-12:
        mags = (mags-mn)/(mx-mn)
    else:
        mags[:] = 0.0
    return mags

def bin_descriptor(mags: np.ndarray, num_bins: int) -> np.ndarray:

    L = len(mags)
    edges = np.linspace(0, L, num_bins + 1, dtype=int)
    binned = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        start, end = edges[i], edges[i+1]
        block = mags[start:end]
        binned[i] = block.mean() if block.size else 0.0
    return binned

CACHE_PATH = os.path.join(data_root, "hand_cache.pkl")

def load_hand_data(root, test=False):
    # check cache
    if not test and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        Xc = cache["X"]
        if Xc.shape[1] == inputs:
            print(f"Loaded hand_cache from {CACHE_PATH} (dim={inputs})")
            return Xc, cache["y"], cache["paths"], cache["crops"]
        else:
            print(f"Cache dimension mismatch: found {Xc.shape[1]} but need {inputs}, recomputing.")
            os.remove(CACHE_PATH)

    X, y, paths, crops = [], [], [], []
    for cls in sorted(os.listdir(root)):
        if (cls.endswith("_test")) != test: continue
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir): continue
        for p in glob.glob(os.path.join(cls_dir, "*.jpg")):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None: continue
            hand_gray, mask_crop = detect_hand(img)
            if hand_gray is None: continue
            crops.append(hand_gray)
            raw  = fourier_descriptor(mask_crop, num_bins)
            desc = bin_descriptor(raw, num_bins)
            X.append(desc)
            y.append(parse_label(cls))
            paths.append(p)

    if X:
        X_arr = np.vstack(X)
        y_arr = np.array(y, int)
        if not test:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({
                    "X": X_arr, "y": y_arr,
                    "paths": paths, "crops": crops
                }, f)
            print(f"Saved hand_cache to {CACHE_PATH} (dim={inputs})")

        return X_arr, y_arr, paths, crops
    else:
        return np.empty((0,0)), np.array([],int), [], []


@njit(fastmath=True)
def simulate_metrics_jit(x_vec, W1_loc, W2, sensitivity):
    n_input = x_vec.shape[0]
    v_h = np.full(n_hidden, v_rest, np.float32)
    v_o = np.full(n_output, v_rest, np.float32)
    hid_spikes = np.zeros(n_hidden, np.int32)
    out_spikes = np.zeros(n_output, np.int32)
    sum_h = sum_o = max_h = max_o = 0.0

    gs = n_hidden // n_input
    rem = n_hidden - gs * n_input

    for p in range(n_input):
        if x_vec[p] == 0.0:
            for i in range(n_hidden):
                v_h[i] = max(v_h[i] - leak_amt, v_rest)
        else:
            extra = 1 if p < rem else 0
            start = p*gs + min(p, rem)
            end   = start + gs + extra
            for h in range(start, end):
                v_h[h] += increment * W1_loc[h,p]
        clipped_h = np.minimum(v_h, threshold)
        sum_h += clipped_h.sum()
        if clipped_h.max() > max_h: max_h = clipped_h.max()
        fired_h = clipped_h >= threshold
        for i in range(n_hidden):
            if fired_h[i]:
                hid_spikes[i] += 1
                v_h[i] = v_rest
        if fired_h.any():
            for i in range(n_output):
                acc = 0.0
                for h in range(n_hidden):
                    if fired_h[h]: acc += W2[i,h]
                v_o[i] += increment * acc
        else:
            for i in range(n_output):
                v_o[i] = max(v_o[i] - leak_amt, v_rest)
        clipped_o = np.minimum(v_o, threshold)
        sum_o += clipped_o.sum()
        if clipped_o.max() > max_o: max_o = clipped_o.max()
        for i in range(n_output):
            if clipped_o[i] >= threshold:
                out_spikes[i] += 1
                v_o[i] = v_rest

    grouped = np.zeros(num_labels, np.int32)
    for j in range(num_labels):
        base = j * (n_output//num_labels)
        for k in range(n_output//num_labels):
            grouped[j] += out_spikes[base + k]
    return grouped, hid_spikes, sum_h/(n_hidden*n_input), max_h, sum_o/(n_output*n_input), max_o, out_spikes

@njit(fastmath=True)
def simulate_spike_trace_jit(x_vec, W1_loc, W2, sensitivity, steps):
    n_input = x_vec.shape[0]
    gs = n_hidden // n_input
    rem = n_hidden - gs*n_input
    v_h = np.full(n_hidden, v_rest, np.float32)
    v_o = np.full(n_output, v_rest, np.float32)
    trace = np.zeros((steps, n_output), np.int32)
    for t in range(steps):
        for p in range(n_input):
            if np.random.random() < x_vec[p]:
                extra = 1 if p<rem else 0
                start = p*gs + min(p,rem)
                end   = start+gs+extra
                for h in range(start,end):
                    v_h[h] += increment*W1_loc[h,p]
            else:
                for i in range(n_hidden):
                    v_h[i] = max(v_h[i]-leak_amt, v_rest)
        clipped_h = np.minimum(v_h,threshold)
        fired_h = clipped_h>=threshold
        for h in range(n_hidden):
            if fired_h[h]: v_h[h]=v_rest
        if fired_h.any():
            for i in range(n_output):
                acc=0.0
                for h in range(n_hidden):
                    if fired_h[h]: acc+=W2[i,h]
                v_o[i]+=increment*acc
        else:
            for i in range(n_output):
                v_o[i]=max(v_o[i]-leak_amt,v_rest)
        for i in range(n_output):
            if v_o[i]>=threshold:
                trace[t,i]=1
                v_o[i]=v_rest
            else:
                trace[t,i]=0
    return trace

@njit(fastmath=True)
def simulate_traces_jit(x_vec, W1_loc, W2, sensitivity, steps):
    # identical to above but returns in_trace, hid_trace, out_trace
    n_input = x_vec.shape[0]
    gs = n_hidden // n_input
    rem = n_hidden - gs*n_input
    v_h = np.full(n_hidden, v_rest, np.float32)
    v_o = np.full(n_output, v_rest, np.float32)
    in_trace  = np.zeros((steps,n_input), np.int32)
    hid_trace = np.zeros((steps,n_hidden),np.int32)
    out_trace = np.zeros((steps,n_output),np.int32)
    for t in range(steps):
        for p in range(n_input):
            if np.random.random() < x_vec[p]:
                in_trace[t,p]=1
                extra = 1 if p<rem else 0
                start = p*gs + min(p,rem)
                end   = start+gs+extra
                for h in range(start,end):
                    v_h[h]+=increment*W1_loc[h,p]
            else:
                in_trace[t,p]=0
                for i in range(n_hidden):
                    v_h[i]=max(v_h[i]-leak_amt,v_rest)
        clipped_h = np.minimum(v_h,threshold)
        fired_h = clipped_h>=threshold
        for h in range(n_hidden):
            if fired_h[h]:
                hid_trace[t,h]=1
                v_h[h]=v_rest
            else:
                hid_trace[t,h]=0
        if fired_h.any():
            for i in range(n_output):
                acc=0.0
                for h in range(n_hidden):
                    if fired_h[h]: acc+=W2[i,h]
                v_o[i]+=increment*acc
        else:
            for i in range(n_output):
                v_o[i]=max(v_o[i]-leak_amt,v_rest)
        for i in range(n_output):
            if v_o[i]>=threshold:
                out_trace[t,i]=1
                v_o[i]=v_rest
            else:
                out_trace[t,i]=0
    return in_trace, hid_trace, out_trace

@njit(fastmath=True)
def infer_jit(x_vec, W1_loc, W2, sensitivity):
    return simulate_metrics_jit(x_vec, W1_loc, W2, sensitivity)[0]

def parse_label(folder_name):
    return int(folder_name.rstrip('_test')[-1])

_global_X_train  = None
_global_dummy_W2 = None
_global_W1_shm   = None
_global_W1_shape = None
_global_W1_lock  = None

def init_pretrain_worker(data):
    worker_ignore_sigint()
    global _global_X_train, _global_dummy_W2, _global_W1_shm, _global_W1_shape, _global_W1_lock
    _global_X_train, _global_dummy_W2, _global_W1_shm, _global_W1_shape, _global_W1_lock = data

def pretrain_chunk(idx_list):
    W1_loc = np.frombuffer(_global_W1_shm, dtype=np.float32).reshape(_global_W1_shape)
    for i in idx_list:
        x_vec = _global_X_train[i]
        in_tr, hid_tr, _ = simulate_traces_jit(x_vec, W1_loc, _global_dummy_W2, sensitivity_init, num_steps)
        coincidence = hid_tr.T.dot(in_tr).astype(np.float32)
        delta = eta_w1 * (coincidence/num_steps)

        print(
            f"W1 sample {i}: Δmin {delta.min():.6f}, "
            f"Δmax {delta.max():.6f}, Δavg {delta.mean():.6f}",
            flush=True
        )

        with _global_W1_lock:
            W1_loc += delta
            np.clip(W1_loc, W1_min_init, W1_max_init, out=W1_loc)

def init_worker(X_tr, y_tr, paths_tr, X_te, y_te, W1_arr):
    worker_ignore_sigint()
    global X_train, y_train, paths, X_test, y_test, W1
    X_train, y_train, paths, X_test, y_test, W1 = X_tr, y_tr, paths_tr, X_te, y_te, W1_arr

def worker_pbt(args):
    model, ep = args
    W2 = model['W2']
    counter = 0

    for i in np.random.permutation(len(X_train)):
        counter += 1
        x_vec      = X_train[i]
        true_label = y_train[i]

        grouped, hid_spikes, avg_h, max_h, avg_o, max_o, out_spikes = \
            simulate_metrics_jit(x_vec, W1, W2, sensitivity_init)

        spike_trace = simulate_spike_trace_jit(
            x_vec, W1, W2, sensitivity_init, num_steps
        )

        rates = spike_trace.sum(axis=0) / x_vec.size

        labels = np.arange(n_output) // neurons_per_group
        true_mask  = (labels == true_label)
        false_mask = ~true_mask

        delta = np.empty_like(W2, dtype=np.float32)

        delta_true = eta_w2 * (target_true/num_steps - rates[true_mask])
        delta[true_mask, :] = delta_true[:, None]

        excess     = np.maximum(0.0, rates[false_mask] - (target_false/num_steps))
        delta_false = -eta_w2 * punish_factor * excess
        delta[false_mask, :] = delta_false[:, None]

        noise = np.random.uniform(-0.15, 0.15, size=W2.shape).astype(np.float32) * eta_w2

        stdp_updates = delta + noise
        W2 += stdp_updates
        np.clip(W2, w_min, w_max, out=W2)

        dmin, dmax, davg = float(stdp_updates.min()), float(stdp_updates.max()), float(stdp_updates.mean())
        wmin, wmax_, wavg = float(W2.min()), float(W2.max()), float(W2.mean())
        true_idxs  = range(true_label * neurons_per_group, (true_label+1) * neurons_per_group)
        max_true   = max(abs(stdp_updates[i, :]).max() for i in true_idxs)
        false_idxs = [i for i in range(n_output) if i // neurons_per_group != true_label]
        max_false  = max(abs(stdp_updates[i, :]).max() for i in false_idxs) if false_idxs else 0.0
        ratio_stdp = max_true / max_false if max_false > 0 else float('inf')

        print(
            f"Sample {counter}: True {true_label} | Pred {int(np.argmax(grouped))} | "
            f"Group: {grouped.tolist()} | "
            f"MaxH {max_h:.3f}, AvgH {avg_h:.3f} | "
            f"MaxO {max_o:.3f}, AvgO {avg_o:.3f} | "
            f"InSpikes:{len(x_vec)} | TotalHidSpikes:{int(hid_spikes.sum())} | "
            f"ΔWmin {dmin:.6f}, ΔWmax {dmax:.6f}, Δavg {davg:.6f} | "
            f"Wmin {wmin:.5f}, Wmax {wmax_:.5f}, Wavg {wavg:.5f} | "
            f"TrueMaxSTDP {max_true:.6f}, FalseMax {max_false:.6f}, RatioT/F {ratio_stdp:.3f}",
            flush=True
        )

    correct = 0
    for idx, (xv, tl) in enumerate(zip(X_test, y_test)):
        g = infer_jit(xv, W1, W2, sensitivity_init)
        p = int(np.argmax(g))
        correct += int(p == tl)
        print(f"Test {idx}: True {tl} | Pred {p} | Grouped {g.tolist()}", flush=True)
    acc = correct / len(y_test) * 100
    print(f"Epoch {ep} Test Accuracy: {acc:.2f}%\n", flush=True)

    model['acc_list'].append(acc)
    if acc >= model['best_acc']:
        model['best_acc']  = acc
        model['best_W2']   = W2.copy()
    model['W2'] = W2
    return model


if __name__ == "__main__":
    mp_hands   = mpp.solutions.hands
    mp_drawing = mpp.solutions.drawing_utils
    hands_net  = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    X_train, y_train, paths, crops_train = load_hand_data(data_root, False)
    X_test,  y_test,  _,     crops_test  = load_hand_data(data_root, True)
    if X_train.size == 0 or X_test.size == 0:
        sys.exit("Need valid hand detections under ./")


    labels, counts = np.unique(y_train, return_counts=True)
    min_count      = counts.min()
    selected_idxs  = []
    for lbl in labels:
        idxs = np.where(y_train == lbl)[0]
        if len(idxs) > min_count:
            idxs = np.random.choice(idxs, min_count, replace=False)
        selected_idxs.extend(idxs.tolist())
    np.random.shuffle(selected_idxs)

    X_train = X_train[selected_idxs]
    y_train = y_train[selected_idxs]
    paths   = [paths[i] for i in selected_idxs]
    crops_train = [crops_train[i] for i in selected_idxs]

    print(f"Balanced training samples per label: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print("Spike Train Length =", X_train.shape[1])

    sample_idxs = random.sample(range(len(crops_train)), min(8, len(crops_train)))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, idx in zip(axes.flatten(), sample_idxs):
        ax.imshow(crops_train[idx].astype(np.uint8), cmap='gray')
        ax.axis('off')
    fig.suptitle("8 Random Example Hand Detections (Cropped)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))
    for ax, idx in zip(axes2.flatten(), sample_idxs):
        desc = X_train[idx]   # length = inputs
        ax.stem(np.arange(inputs), desc, basefmt=" ")
        ax.set_title(f"Sample #{idx} → {inputs} bins")
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Mean magnitude")
    fig2.suptitle("Binned Fourier Descriptors (stem)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    if args.w1_file and os.path.isfile(args.w1_file):
        print(f"Loading W1 from {args.w1_file}")
        W1 = np.loadtxt(args.w1_file, delimiter=',').astype(np.float32)
    else:
        print("Pretraining W1…")
        W1 = np.random.uniform(W1_min_init, W1_max_init,
                               (n_hidden, inputs)).astype(np.float32)
        W1_shm   = Array('f', W1.flatten(), lock=False)
        W1_shape = W1.shape
        W1_lock  = Lock()
        dummy_W2 = np.zeros((n_output, n_hidden), np.float32)
        for ep in range(1, 11):
            print(f" W1 epoch {ep}/10", flush=True)
            pre_pool = mp.Pool(min(mp.cpu_count(), mp.cpu_count()*0.75),
                               initializer=init_pretrain_worker,
                               initargs=((X_train, dummy_W2, W1_shm, W1_shape, W1_lock),))
            pools.append(pre_pool)
            chunks = np.array_split(list(range(len(X_train))), mp.cpu_count())
            pre_pool.map(pretrain_chunk, chunks)
            pre_pool.close(); pre_pool.join(); pools.remove(pre_pool)
            W1 = np.frombuffer(W1_shm, dtype=np.float32).reshape(W1_shape)
            print(f"  After epoch {ep}: W1 Δavg {W1.mean():.6f}", flush=True)
        if args.w1_file:
            np.savetxt(args.w1_file, W1, delimiter=',')
            print(f"Saved pretrained W1 to {args.w1_file}")

    pop_size = min(mp.cpu_count(), mp.cpu_count() * 0.75)
    population = []
    for _ in range(pop_size):
        W2_init = np.random.uniform(w_min, w_max, (n_output, n_hidden)).astype(np.float32)
        population.append({
            'W2':        W2_init,
            'best_acc':  0.0,
            'best_W2':   W2_init.copy(),
            'acc_list':  []
        })

    plt.ion()
    fig_lp, ax_lp = plt.subplots(figsize=(8,4))
    ax_lp.set_title('Gen & All-Time Best Accuracy')
    ax_lp.set_xlabel('Epoch'); ax_lp.set_ylabel('%'); ax_lp.grid(True)
    best_gen_history = []
    best_all_history = []

    pool = mp.Pool(pop_size, initializer=init_worker,
                  initargs=(X_train, y_train, paths, X_test, y_test, W1))
    pools.append(pool)

    for ep in range(1, epochs+1):
        print(f"\n=== Epoch {ep}/{epochs} ===", flush=True)

        population.sort(key=lambda m: m['best_acc'], reverse=True)
        for i in range(n_elites):
            population[i]['W2'] = population[i]['best_W2'].copy()

        population = pool.map(worker_pbt, [(m, ep) for m in population])

        if ep % exploit_every == 0:
            population.sort(key=lambda m: m['best_acc'], reverse=True)
            half = len(population) * 3 // 4
            top = population[:half]
            bot = population[half:]

            for loser in bot:
                p1, p2 = random.sample(top, 2)
                Wp1 = p1['best_W2']
                Wp2 = p2['best_W2']

                mask = np.random.randint(0, 2, size=Wp1.shape, dtype=np.uint8)

                child = mask * Wp1 + (1 - mask) * Wp2

                noise = np.random.uniform(0.98, 1.02, size=child.shape).astype(np.float32)
                child = child * noise

                child = np.clip(child, w_min, w_max)

                loser['W2'] = child

        gen_best = max(m['acc_list'][-1] for m in population)
        all_best = max(m['best_acc'] for m in population)
        best_gen_history.append(gen_best)
        best_all_history.append(all_best)
        ax_lp.clear()
        ax_lp.plot(range(1, len(best_gen_history)+1),
                   best_gen_history, marker='o', label='Gen Best Acc')
        ax_lp.plot(range(1, len(best_all_history)+1),
                   best_all_history, marker='s', label='All-Time Best Acc')
        ax_lp.legend(); ax_lp.grid(True)
        plt.pause(0.1)

    pool.close(); pool.join()
    plt.ioff(); plt.show()

    flat_W1 = np.zeros(n_hidden, dtype=np.float32)

    gs  = n_hidden // inputs
    rem = n_hidden - gs * inputs

    for p in range(inputs):
        extra   = 1 if p < rem else 0
        start_h = p * gs + min(p, rem)
        end_h   = start_h + gs + extra
        flat_W1[start_h:end_h] = W1[start_h:end_h, p]

    if args.w1_file:
        np.savetxt(args.w1_file, flat_W1, delimiter=',', fmt='%.2f')
        print(f"Saved compact W1 ({n_hidden} weights) to {args.w1_file}")
    if args.w2_file:
        champion = max(population, key=lambda m: m['best_acc'])
        np.savetxt(args.w2_file, champion['best_W2'], delimiter=',', fmt='%.2f')
        print(f"Saved best W2 to {args.w2_file}")

    hands_net.close()

    print("Done.")
