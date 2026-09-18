[![Video Presentation and Guide](https://img.youtube.com/vi/GTWTZHz4MXo/hqdefault.jpg)](https://youtu.be/GTWTZHz4MXo)


# 🧠 SNNGesturesPortentaH7

Spiking Neural Network (SNN) for real-time gesture recognition using the **Arduino Portenta H7** and an **OV7670 camera**. This project implements biologically-inspired neural computing on edge hardware using radially invariant Fourier features extracted from video frames.

---

## 📦 Project Structure

```
SNNGesturesPortentaH7/
├── portenta/              # Arduino sketch & SNN logic
│   ├── portentacam.ino
│   ├── weights.h
│   └── ...
├── scripts/               # Python utilities for training and inference
│   ├── cameraread2.py     # Live camera inference from host side
│   ├── pbtalgo.py         # Population-Based Training algorithm
│   ├── prepweights.py     # Converts weights to C-style header
│   ├── videotoimage.py    # Splits videos into training images
│   └── ...
├── models/                # Pretrained or trained weights
│   ├── w1_64.csv
│   └── w2_64.csv
├── data/                  # Gesture image datasets
│   ├── gesture0/
│   ├── gesture0_test/
│   └── ...
└── README.md
```

---

## 📷 Hardware Requirements

- **Arduino Portenta H7**
  - STM32H747XI dual-core (M7 + M4)
  - 8 MB SDRAM / 16 MB Flash
  - MIPI DSI, DVP camera interface
- **OV7670 Camera Module**
  - VGA (640×480), used in QVGA (320×240)
  - RGB565 format at 30 FPS
- **Portenta Breakout Board**
  - Enables GPIO clock generation via HAL (for XCLK)
  - Uses 80-pin high-density connectors

---

## ⚙️ Setup Instructions

### Arduino Side

1. **Install Arduino IDE**:  
   https://www.arduino.cc/en/software

2. **Install Required Libraries**:
   - `Mbed OS`
   - `Arducam_dvp.h`
   - `Arduino_PortentaBreakout.h`

3. **Flash Arduino Code**:
   - Open and upload `portentacam.ino` to your Portenta.
   - Ensure `weights.h` matches your network size.

---

### Python Side (Host System)

#### 🔧 Environment

Install [Miniconda](https://www.anaconda.com/download) and then:

```bash
pip install opencv-python numpy matplotlib numba pyserial mediapipe==0.10.13
```

#### 🎥 Live Inference

```bash
python cameraread2.py --inc 0.078 --leak 0.01 --amp-mid 1.3
```

Optional flags:
- `--video-only` : Record raw camera input for new data
- `--alpha`, `--beta` : Adjust contrast/brightness

---

## 📊 Training New Gestures

1. **Record gesture videos**:
   ```bash
   python cameraread2.py --video-only
   ```

2. **Extract and normalize frames**:
   ```bash
   python videotoimage.py --start-index 0 myvideo.mp4 ./data/gesture1
   ```

3. **Train your model**:
   ```bash
   python pbtalgo.py --w1-file w1_64.csv --w2-file w2_64.csv --steps 100
   ```

4. **Convert weights to C header**:
   ```bash
   python prepweights.py
   ```

5. **Move new `weights.h` to Arduino project** and re-flash.

---

## 🧠 SNN Details

- Radially Invariant Fourier Transform (RIFT) for input encoding
- LIF neuron dynamics:
  - `increment`, `leak_ratio`, `threshold`, etc.
- STDP with target firing rates (`target_true`, `target_false`)
- Uses **Population-Based Training** (PBT) to select optimal weights
- Temporal processing over `steps` (e.g., 100 time steps per gesture)

---

## 🛠 Example Parameters

| Parameter        | Value   | Description                                 |
|------------------|---------|---------------------------------------------|
| `n_hidden`       | 64      | Number of hidden neurons (and input bins)   |
| `n_output`       | 16      | Number of output neurons                    |
| `neurons_per_group` | 8   | Output neurons per label (2 labels here)    |
| `eta_w2`         | 0.0001  | STDP learning rate                          |
| `target_true`    | 200     | Desired spikes for correct label            |
| `punish_factor`  | 35      | Penalty scale for incorrect spikes          |
| `epochs`         | 500     | Number of training epochs                   |

---

## 📎 Useful Links

- 🔗 [Arduino IDE Download](https://www.arduino.cc/en/software)
- 🔗 [Miniconda (Recommended)](https://www.anaconda.com/download)
- 🔗 [Project Repository](https://github.com/INQUIRELAB/SNNGesturesPortentaH7)

---

## 📌 License

This project is distributed under the PolyForm Noncommercial License 1.0.0.

## License scope

Original INQUIRE Lab code is licensed under PolyForm Noncommercial License 1.0.0. Original INQUIRE Lab datasets, figures, and documentation are licensed under CC BY-NC 4.0. See [LICENSE](LICENSE) for scope and the full license texts. Materials from other rights holders retain their original terms.
