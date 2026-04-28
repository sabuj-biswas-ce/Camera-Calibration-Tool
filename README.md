# 📷 Camera Calibration Tool

A Python-based GUI tool for calibrating cameras using the **checkerboard method**.
It computes the **camera matrix** and **distortion coefficients** for accurate image correction and computer vision applications.

---

## 📝 Description

This tool performs camera calibration using multiple images of a checkerboard pattern.
It detects corner points, estimates intrinsic camera parameters, and allows visualization of distorted and undistorted images.

The results can be used for:

* Camera calibration
* Lens distortion correction
* Perspective transformation
* Computer vision tasks

---

## 🚀 Features

* Load calibration images from a folder
* Automatic checkerboard detection
* Sub-pixel corner refinement
* Compute:

  * Camera matrix
  * Distortion coefficients
* Visual preview:

  * Distorted image
  * Undistorted image
  * Side-by-side comparison
* Export calibration results to JSON
* Reprojection error reporting

---

## ▶️ How to Run

```bash
python Camera_Calibration.py
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
opencv-python>=4.8.0
numpy>=1.24.0
PySide6>=6.5.0
```
## ⚙️ Input Parameters

Before running calibration, you need to configure the checkerboard settings correctly:

### 🔢 Checkerboard Pattern Size

* **Columns** → Number of inner corners along width
* **Rows** → Number of inner corners along height

👉 Example:
If your checkerboard has **10 × 7 squares**, then:

* Inner corners = **9 × 6**

So you should set:

```
Columns = 9
Rows = 6
```

---

### 📏 Square Size

* Physical size of one checkerboard square
* Unit is  **millimeters (mm)** 

👉 Example:

```
Square size = 25.0 mm
```

---

### 🎯 Sub-pixel Refinement (Optional)

* Improves corner detection accuracy
* Recommended: ✅ Enabled

---

### 🖼️ Calibration Images Requirements

For best results:

* Use at least **3–10 images**
* Capture checkerboard from **different angles and positions**
* Ensure:

  * Good lighting
  * Sharp images (no blur)
  * Full checkerboard visible
---

## 📂 Output

### Calibration JSON

* Camera matrix
* Distortion coefficients
* Reprojection error
* Image size
* Pattern details

### Simplified JSON

* fx, fy, cx, cy
* k1, k2, k3, p1, p2

---

## ⚠️ Notes

* Use a checkerboard pattern (e.g., 9×6 inner corners)
* Provide at least 3–6 good-quality images
* Ensure the checkerboard is visible from different angles
* Lower reprojection error = better calibration

---

## 🛠️ Built With

* Python
* PySide6 (GUI)
* OpenCV
* NumPy

---

## 👤 Author

**Sabuj Biswas**
GUI Developed with assistance from Al Tools (ChatGPT)

---

