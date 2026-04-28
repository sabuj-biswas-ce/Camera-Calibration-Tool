import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    reprojection_error: float
    image_size: Tuple[int, int]
    valid_images: List[str]
    square_size: float
    pattern_size: Tuple[int, int]

    def to_json_dict(self) -> dict:
        # Export structure with exact key names requested.
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "reprojection_error": float(self.reprojection_error),
            "image_size": list(self.image_size),
            "valid_images": self.valid_images,
            "square_size": float(self.square_size),
            "pattern_size": list(self.pattern_size),
        }


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setFrameShape(QFrame.StyledPanel)
        self.setText("Preview will appear here")
        self._base_pixmap: Optional[QPixmap] = None

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._base_pixmap = pixmap
        self._update_scaled()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.setPixmap(QPixmap())
            return
        scaled = self._base_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class CalibrationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        pattern_size: Tuple[int, int],
        square_size: float,
        use_subpixel: bool,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.use_subpixel = use_subpixel

    @Slot()
    def run(self) -> None:
        try:
            objpoints = []
            imgpoints = []
            valid_images = []
            image_size = None

            cols, rows = self.pattern_size
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = (
                np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size
            )

            self.log.emit(f"Starting calibration using {len(self.image_paths)} images...")

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )

            for path in self.image_paths:
                img = cv2.imread(path)
                if img is None:
                    self.log.emit(f"Skipped unreadable image: {os.path.basename(path)}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if image_size is None:
                    image_size = (gray.shape[1], gray.shape[0])

                found, corners = cv2.findChessboardCorners(
                    gray,
                    self.pattern_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK
                    + cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if not found:
                    self.log.emit(f"No checkerboard found: {os.path.basename(path)}")
                    continue

                if self.use_subpixel:
                    corners = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        criteria,
                    )

                objpoints.append(objp.copy())
                imgpoints.append(corners)
                valid_images.append(path)
                self.log.emit(f"Accepted: {os.path.basename(path)}")

            if len(objpoints) < 3:
                raise RuntimeError(
                    "Not enough valid checkerboard images. At least 3 are recommended."
                )

            self.log.emit("Running cv2.calibrateCamera...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                image_size,
                None,
                None,
            )

            # Reprojection error diagnostic.
            total_error = 0.0
            for i, objp_i in enumerate(objpoints):
                projected, _ = cv2.projectPoints(
                    objp_i, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
                total_error += error
            reproj_error = total_error / len(objpoints)

            result = CalibrationResult(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                reprojection_error=float(reproj_error if ret else np.nan),
                image_size=image_size if image_size is not None else (0, 0),
                valid_images=valid_images,
                square_size=self.square_size,
                pattern_size=self.pattern_size,
            )

            self.log.emit("Calibration finished.")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class CameraCalibrationTool(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Camera Calibration Tool")
        self.resize(1440, 860)

        self.image_paths: List[str] = []
        self.current_image_path: Optional[str] = None
        self.calibration_result: Optional[CalibrationResult] = None
        self.corner_cache: dict = {}
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[CalibrationWorker] = None

        self._build_ui()
        self._apply_style()
        self._set_busy(False)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # Left preview panel.
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        title = QLabel("Processing Preview")
        title.setObjectName("panelTitle")
        subtitle = QLabel(
            "Still-image review with distorted / undistorted / side-by-side modes"
        )
        subtitle.setObjectName("panelSubtitle")
        self.preview_label = PreviewLabel()

        mode_row = QHBoxLayout()
        self.btn_show_distorted = QPushButton("Show Distorted")
        self.btn_show_undistorted = QPushButton("Show Undistorted")
        self.btn_show_side = QPushButton("Show Side by Side")
        for btn in (
            self.btn_show_distorted,
            self.btn_show_undistorted,
            self.btn_show_side,
        ):
            btn.setCheckable(True)
            mode_row.addWidget(btn)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.btn_show_distorted)
        mode_group.addButton(self.btn_show_undistorted)
        mode_group.addButton(self.btn_show_side)
        self.btn_show_distorted.setChecked(True)
        self.btn_show_distorted.clicked.connect(self.update_preview)
        self.btn_show_undistorted.clicked.connect(self.update_preview)
        self.btn_show_side.clicked.connect(self.update_preview)

        self.preview_info = QLabel("No image selected.")
        self.preview_info.setObjectName("infoLabel")

        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)
        left_layout.addWidget(self.preview_label, stretch=1)
        left_layout.addLayout(mode_row)
        left_layout.addWidget(self.preview_info)

        # Right control panel in a scroll area.
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(430)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setSpacing(12)
        right_scroll.setWidget(right_content)

        right_layout.addWidget(self._build_images_group())
        right_layout.addWidget(self._build_checkerboard_group())
        right_layout.addWidget(self._build_activity_group())
        right_layout.addWidget(self._build_numbers_group())
        right_layout.addWidget(self._build_summary_group())
        right_layout.addStretch()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _build_images_group(self) -> QGroupBox:
        box = QGroupBox("Images and Actions")
        layout = QVBoxLayout(box)

        path_row = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select image folder...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_folder)
        path_row.addWidget(self.folder_input)
        path_row.addWidget(self.btn_browse)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)

        actions_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Images")
        self.btn_calibrate = QPushButton("Calculate Distortion")
        self.btn_open_undistort = QPushButton("Open Image for Undistort")
        self.btn_refresh.clicked.connect(self.refresh_images)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_open_undistort.clicked.connect(self.open_image_for_undistort)
        actions_row.addWidget(self.btn_refresh)
        actions_row.addWidget(self.btn_calibrate)
        actions_row.addWidget(self.btn_open_undistort)

        layout.addLayout(path_row)
        layout.addWidget(self.image_list)
        layout.addLayout(actions_row)
        return box

    def _build_checkerboard_group(self) -> QGroupBox:
        box = QGroupBox("Checkerboard Details")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 50)
        self.spin_cols.setValue(9)
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 50)
        self.spin_rows.setValue(6)
        row1.addWidget(QLabel("Inner corners columns"))
        row1.addWidget(self.spin_cols)
        row1.addWidget(QLabel("Inner corners rows"))
        row1.addWidget(self.spin_rows)

        row2 = QHBoxLayout()
        self.spin_square = QDoubleSpinBox()
        self.spin_square.setRange(0.001, 1000.0)
        self.spin_square.setDecimals(3)
        self.spin_square.setValue(25.0)
        self.spin_square.setSuffix(" mm")
        row2.addWidget(QLabel("Square size"))
        row2.addWidget(self.spin_square)

        self.chk_subpixel = QCheckBox("sub-pixel corner refinement")
        self.chk_subpixel.setChecked(True)
        self.chk_draw_corners = QCheckBox("draw detected checkerboard corners on preview")
        self.chk_draw_corners.setChecked(True)
        self.chk_crop_roi = QCheckBox("crop to valid undistorted ROI")
        self.chk_crop_roi.setChecked(True)

        self.chk_draw_corners.stateChanged.connect(self.update_preview)
        self.chk_crop_roi.stateChanged.connect(self.update_preview)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.chk_subpixel)
        layout.addWidget(self.chk_draw_corners)
        layout.addWidget(self.chk_crop_roi)
        return box

    def _build_activity_group(self) -> QGroupBox:
        box = QGroupBox("Activity Monitor")
        layout = QVBoxLayout(box)

        self.running_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate.
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(130)

        layout.addWidget(self.running_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log_box)
        return box

    def _build_numbers_group(self) -> QGroupBox:
        box = QGroupBox("Calibration Numbers (10 digits)")
        layout = QVBoxLayout(box)

        self.dist_text = QPlainTextEdit()
        self.dist_text.setReadOnly(True)
        self.dist_text.setPlaceholderText("Distortion coefficients will appear here...")
        self.matrix_text = QPlainTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setPlaceholderText("Camera matrix will appear here...")
        self.btn_save_json = QPushButton("Save Calibration JSON")
        self.btn_save_json.clicked.connect(self.save_calibration_json)

        layout.addWidget(QLabel("Distortion coefficients"))
        layout.addWidget(self.dist_text)
        layout.addWidget(QLabel("Camera matrix"))
        layout.addWidget(self.matrix_text)
        layout.addWidget(self.btn_save_json)
        return box

    def _build_summary_group(self) -> QGroupBox:
        box = QGroupBox("Calibration Summary")
        layout = QVBoxLayout(box)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(140)
        self.summary_text.setPlaceholderText(
            "Calibration results, warnings, and diagnostics will appear here."
        )
        layout.addWidget(self.summary_text)
        return box

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI";
                font-size: 10.5pt;
                color: #1f2937;
                background: #f6f8fb;
            }
            QGroupBox {
                border: 1px solid #d9e1ec;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px;
                background: #ffffff;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: -8px;
                padding: 0 6px;
                background: #f6f8fb;
            }
            QPushButton {
                background: #e7eef8;
                border: 1px solid #c7d5e7;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #dfeaf9;
            }
            QPushButton:checked {
                background: #2962ff;
                color: #ffffff;
                border: 1px solid #2458e2;
            }
            QLineEdit, QListWidget, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
                border: 1px solid #d0dae8;
                border-radius: 8px;
                padding: 6px;
                background: #ffffff;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            #panelTitle {
                font-size: 16pt;
                font-weight: 700;
            }
            #panelSubtitle {
                color: #5b6778;
            }
            #infoLabel {
                color: #3e4c60;
                font-style: italic;
                padding-top: 2px;
            }
            """
        )

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.running_label.setText("Running..." if busy else "Idle")
        self.btn_calibrate.setEnabled(not busy)
        self.btn_refresh.setEnabled(not busy)
        self.btn_browse.setEnabled(not busy)
        self.btn_open_undistort.setEnabled(not busy)

    def _append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {text}")

    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.folder_input.setText(folder)
        self.refresh_images()

    def refresh_images(self) -> None:
        folder = self.folder_input.text().strip()
        self.image_list.clear()
        self.image_paths = []
        self.current_image_path = None
        self.corner_cache.clear()

        if not folder or not os.path.isdir(folder):
            self.preview_info.setText("Select a valid folder to load images.")
            self.preview_label.set_pixmap(None)
            return

        files = []
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(folder, name))

        for path in files:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.image_list.addItem(item)

        self.image_paths = files
        self._append_log(f"Detected {len(files)} image(s) in folder.")
        if files:
            self.image_list.setCurrentRow(0)
        else:
            self.preview_info.setText("No supported images found.")
            self.preview_label.set_pixmap(None)

    def on_image_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        _ = previous
        if current is None:
            self.current_image_path = None
            self.preview_label.set_pixmap(None)
            return
        self.current_image_path = current.data(Qt.UserRole)
        self.update_preview()

    def _get_detection_for_image(self, image_path: str):
        cache_key = (
            image_path,
            self.spin_cols.value(),
            self.spin_rows.value(),
            self.chk_subpixel.isChecked(),
        )
        if cache_key in self.corner_cache:
            return self.corner_cache[cache_key]

        img = cv2.imread(image_path)
        if img is None:
            result = (False, None)
            self.corner_cache[cache_key] = result
            return result

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            (self.spin_cols.value(), self.spin_rows.value()),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found and self.chk_subpixel.isChecked():
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        result = (found, corners)
        self.corner_cache[cache_key] = result
        return result

    def _undistort_image(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.calibration_result is None:
            return img_bgr

        h, w = img_bgr.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.calibration_result.camera_matrix,
            self.calibration_result.distortion_coefficients,
            (w, h),
            1,
            (w, h),
        )
        undist = cv2.undistort(
            img_bgr,
            self.calibration_result.camera_matrix,
            self.calibration_result.distortion_coefficients,
            None,
            new_mtx,
        )
        if self.chk_crop_roi.isChecked():
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                undist = undist[y : y + rh, x : x + rw]
        return undist

    def _to_qpixmap(self, img_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def update_preview(self) -> None:
        if not self.current_image_path:
            self.preview_info.setText("No image selected.")
            self.preview_label.set_pixmap(None)
            return

        img = cv2.imread(self.current_image_path)
        if img is None:
            self.preview_info.setText("Failed to load selected image.")
            self.preview_label.set_pixmap(None)
            return

        found, corners = self._get_detection_for_image(self.current_image_path)
        distorted = img.copy()

        if self.chk_draw_corners.isChecked() and found and corners is not None:
            cv2.drawChessboardCorners(
                distorted,
                (self.spin_cols.value(), self.spin_rows.value()),
                corners,
                found,
            )

        if self.btn_show_distorted.isChecked():
            render = distorted
            mode_name = "Distorted"
        elif self.btn_show_undistorted.isChecked():
            render = self._undistort_image(img)
            mode_name = "Undistorted"
        else:
            und = self._undistort_image(img)
            # Keep side-by-side robust when ROI crop changes dimensions.
            target_h = max(distorted.shape[0], und.shape[0])
            def pad_to_height(image, hgt):
                if image.shape[0] == hgt:
                    return image
                pad = hgt - image.shape[0]
                return cv2.copyMakeBorder(
                    image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
                )
            left = pad_to_height(distorted, target_h)
            right = pad_to_height(und, target_h)
            render = np.hstack([left, right])
            mode_name = "Side by Side"

        self.preview_label.set_pixmap(self._to_qpixmap(render))
        status = "checkerboard found" if found else "checkerboard not found"
        self.preview_info.setText(
            f"{os.path.basename(self.current_image_path)} | {mode_name} | {status}"
        )

    def start_calibration(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return

        pattern_size = (self.spin_cols.value(), self.spin_rows.value())
        square_size = self.spin_square.value()
        use_subpixel = self.chk_subpixel.isChecked()

        self._append_log(
            f"Calibration requested. Pattern={pattern_size}, square={square_size} mm."
        )
        self._set_busy(True)

        self.worker_thread = QThread(self)
        self.worker = CalibrationWorker(
            image_paths=list(self.image_paths),
            pattern_size=pattern_size,
            square_size=square_size,
            use_subpixel=use_subpixel,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self.on_calibration_finished)
        self.worker.failed.connect(self.on_calibration_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @Slot(object)
    def on_calibration_finished(self, result: CalibrationResult) -> None:
        self.calibration_result = result
        self._set_busy(False)
        self._append_log("Calibration data updated in UI.")

        self.dist_text.setPlainText(
            json.dumps(
                result.distortion_coefficients.round(10).tolist(),
                indent=2,
            )
        )
        self.matrix_text.setPlainText(
            json.dumps(result.camera_matrix.round(10).tolist(), indent=2)
        )

        warnings = []
        if len(result.valid_images) < 6:
            warnings.append("Low number of valid images may reduce calibration quality.")
        if result.reprojection_error > 1.0:
            warnings.append("Reprojection error is relatively high (> 1.0 px).")
        if not warnings:
            warnings.append("No major warnings.")

        summary_lines = [
            "Calibration complete.",
            f"Valid images: {len(result.valid_images)} / {len(self.image_paths)}",
            f"Pattern size (cols, rows): {result.pattern_size}",
            f"Square size (mm): {result.square_size}",
            f"Image size (w, h): {result.image_size}",
            f"Mean reprojection error: {result.reprojection_error:.10f}",
            "",
            "Warnings / Diagnostics:",
            *[f"- {w}" for w in warnings],
        ]
        self.summary_text.setPlainText("\n".join(summary_lines))
        self.update_preview()

    @Slot(str)
    def on_calibration_failed(self, message: str) -> None:
        self._set_busy(False)
        self._append_log(f"Calibration failed: {message}")
        QMessageBox.critical(self, "Calibration Failed", message)

    def open_image_for_undistort(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image for Undistort",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        folder = os.path.dirname(path)
        self.folder_input.setText(folder)
        self.refresh_images()

        # Select if image exists in list, otherwise still display it.
        found_item = None
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == path:
                found_item = item
                break

        if found_item is not None:
            self.image_list.setCurrentItem(found_item)
        else:
            self.current_image_path = path
            self.update_preview()

    def save_calibration_json(self) -> None:
        if self.calibration_result is None:
            QMessageBox.information(
                self, "No Calibration", "Please run calibration before saving JSON."
            )
            return

        default_name = "calibration_result.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration JSON",
            default_name,
            "JSON Files (*.json)",
        )
        if not path:
            return

        payload = self.calibration_result.to_json_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Also save a simplified intrinsic/distortion JSON with fx/fy/cx/cy/k1/k2/k3/p1/p2.
        cm = self.calibration_result.camera_matrix
        dc = self.calibration_result.distortion_coefficients.flatten()

        def _dc(i: int) -> float:
            return float(dc[i]) if i < len(dc) else 0.0

        calibration_simple = {
            "fx": float(cm[0, 0]),
            "fy": float(cm[1, 1]),
            "cx": float(cm[0, 2]),
            "cy": float(cm[1, 2]),
            "k1": _dc(0),
            "k2": _dc(1),
            "k3": _dc(4),  # OpenCV order: k1, k2, p1, p2, k3, ...
            "p1": _dc(2),
            "p2": _dc(3),
        }

        simple_path = os.path.join(os.path.dirname(path), "camera_calibration.json")
        with open(simple_path, "w", encoding="utf-8") as f:
            json.dump(calibration_simple, f, indent=2)

        self._append_log(f"Saved calibration JSON: {path}")
        self._append_log(f"Saved simplified calibration JSON: {simple_path}")


def main() -> None:
    app = QApplication(sys.argv)
    window = CameraCalibrationTool()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
