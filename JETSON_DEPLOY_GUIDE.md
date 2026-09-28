# 🦺 PPE-Watch — Jetson Nano Deployment Guide

Complete step-by-step guide to deploy PPE-Watch on your **NVIDIA Jetson Nano**.

---

## 📋 Prerequisites

- Jetson Nano powered on and connected to **same WiFi/LAN** as your Mac
- Jetson Nano has a **display** and **camera** connected
- You know the Jetson's **IP address** and **login credentials**

---

## Step 1: Find Jetson Nano's IP Address

**On the Jetson Nano** (with display connected), open a terminal and run:

```bash
hostname -I
```

This will print something like: `192.168.1.42`

> **💡 Tip:** You can also check your router's admin page to see connected devices.

---

## Step 2: SSH into Jetson Nano from Mac

Open **Terminal** on your Mac and run:

```bash
ssh <username>@<JETSON_IP>
```

**Example** (default Jetson credentials are usually `jetson` / `jetson`):

```bash
ssh jetson@192.168.1.42
```

It will ask for the password. Type it and press Enter.

> **First time connecting?** It will ask to accept the fingerprint — type `yes` and press Enter.

---

## Step 3: Copy PPE-Watch Folder to Jetson

Open a **new Terminal tab** on your Mac (keep the SSH session open in the other tab).

Run this command to copy the entire PPE-watch folder:

```bash
scp -r /Users/neeru/Documents/iot/PPE-watch/ jetson@192.168.1.42:~/PPE-watch
```

> ⏳ This may take a few minutes because `best.pt` is ~52MB.

To verify the copy worked, go back to your **SSH tab** and run:

```bash
ls -lh ~/PPE-watch/
```

You should see: `app.py`, `best.pt`, `requirements.txt`, `run.sh`, etc.

---

## Step 4: Install PyTorch on Jetson (One-Time Setup)

⚠️ **IMPORTANT:** Do **NOT** run `pip install torch` on Jetson Nano. It won't work. You must install the **NVIDIA-provided PyTorch wheel**.

### 4a. Check your JetPack version

```bash
cat /etc/nv_tegra_release
```

or:

```bash
sudo apt-cache show nvidia-jetpack | grep Version
```

### 4b. Download the correct PyTorch wheel

Go to the NVIDIA forum page (on the Jetson's browser or download on Mac and `scp` it over):

🔗 **https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048**

Pick the PyTorch version matching your JetPack. For example:

| JetPack | PyTorch | Download |
|---|---|---|
| JetPack 4.6 | PyTorch 1.11 | `torch-1.11.0-cp36-cp36m-linux_aarch64.whl` |
| JetPack 5.x | PyTorch 2.0 | `torch-2.0.0-cp38-cp38-linux_aarch64.whl` |

### 4c. Install PyTorch from the .whl file

```bash
# Install dependencies first
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev

# Install the downloaded wheel (adjust filename to match yours)
pip3 install torch-1.11.0-cp36-cp36m-linux_aarch64.whl
```

### 4d. Verify PyTorch is working

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see `CUDA available: True` ✅

---

## Step 5: Install PPE-Watch Dependencies

SSH into the Jetson and run:

```bash
cd ~/PPE-watch

# Install all Python dependencies
pip3 install -r requirements.txt
```

This installs: `ultralytics` (YOLO), `gradio`, `opencv-python`, `numpy`

> If OpenCV gives issues, try the system package instead:
> ```bash
> sudo apt-get install -y python3-opencv
> ```

---

## Step 6: Fix Server Binding (One-Time)

To make the app accessible on the Jetson's display AND from other devices, edit `app.py`:

```bash
nano ~/PPE-watch/app.py
```

Find the **last line** with `demo.launch(...)` and change:

```python
# Change FROM:
demo.launch(server_name="127.0.0.1", share=False)

# Change TO:
demo.launch(server_name="0.0.0.0", share=False)
```

Save with `Ctrl+O`, Enter, then `Ctrl+X` to exit nano.

---

## Step 7: Run PPE-Watch! 🚀

```bash
cd ~/PPE-watch
python3 app.py
```

Or use the run script:

```bash
cd ~/PPE-watch
chmod +x run.sh
./run.sh
```

You'll see output like:

```
Running on local URL:  http://0.0.0.0:7860
```

### Open in Browser

- **On Jetson's display:** Open browser → `http://localhost:7860`
- **From your Mac:** Open browser → `http://192.168.1.42:7860`

---

## 🎯 Quick Reference — All Commands Summary

```bash
# === FROM YOUR MAC ===

# 1. Copy files to Jetson
scp -r /Users/neeru/Documents/iot/PPE-watch/ jetson@<IP>:~/PPE-watch

# 2. SSH into Jetson
ssh jetson@<IP>


# === ON THE JETSON (via SSH or local terminal) ===

# 3. Install system deps (one-time)
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev

# 4. Install PyTorch wheel (one-time) — download from NVIDIA first
pip3 install torch-<version>-linux_aarch64.whl

# 5. Install app dependencies
cd ~/PPE-watch
pip3 install -r requirements.txt

# 6. Run the app
python3 app.py
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Install PyTorch from NVIDIA wheel (Step 4) |
| `Connection refused` on Mac browser | Change `server_name` to `"0.0.0.0"` in app.py (Step 6) |
| Camera not detected in Gradio | Check camera: `ls /dev/video*` — if empty, camera isn't connected properly |
| Port 7860 already in use | Kill the old process: `kill $(lsof -t -i:7860)` then re-run |
| Slow inference | Normal on Jetson Nano (~2-5 FPS). Consider reducing input resolution |
| CUDA out of memory | Close other apps on Jetson, or reduce `max_det` in app.py |
