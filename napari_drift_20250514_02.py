import sys
import numpy as np
import pandas as pd
from skimage.io import imsave
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSpinBox, QLabel, QGroupBox, QFileDialog, QMessageBox
)
from napari import Viewer
from napari._qt.qt_viewer import QtViewer
from scipy.optimize import curve_fit
from contextlib import contextmanager


class DualViewerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Napari Drift Correction GUI")
        self.setGeometry(50, 50, 2400, 900)

        layout = QHBoxLayout()
        self.setLayout(layout)

        # --- Napari Viewers Setup (left/middle/right) ---
        self.viewer_raw = Viewer()
        self.viewer_raw.window.title = "Raw Data"
        self.qt_raw = QtViewer(self.viewer_raw)
        layout.addWidget(self.qt_raw)

        self.viewer_corr = Viewer()
        self.viewer_corr.window.title = "Tentative Correction"
        self.qt_corr = QtViewer(self.viewer_corr)
        layout.addWidget(self.qt_corr)

        self.viewer_refined = Viewer()
        self.viewer_refined.window.title = "Gaussian Drift Correction"
        self.qt_refined = QtViewer(self.viewer_refined)
        layout.addWidget(self.qt_refined)

        # --- Points Layers ---
        self.point_layer = self.viewer_raw.add_points(name="atom_positions", ndim=3, size=3, face_color="red")
        self.point_layer.editable = True
        self.point_layer.events.data.connect(lambda e: self.update_corrected())

        # 補完ポイント用レイヤー（可視／不可視切替）
        self.interp_layer = self.viewer_raw.add_points(name="interpolated", ndim=3, size=3, face_color="yellow")
        self.interp_layer.visible = False
        self.interp_layer.editable = False

        # --- Slider Sync ---
        self.viewer_raw.dims.events.current_step.connect(
            lambda e: self.viewer_corr.dims.set_current_step(0, self.viewer_raw.dims.current_step[0])
        )
        self.viewer_raw.dims.events.current_step.connect(
            lambda e: self.viewer_refined.dims.set_current_step(0, self.viewer_raw.dims.current_step[0])
        )

        self.patch_size = 9
        self.refined_points = []

        self.setup_control_panel(layout)

        print("🖼️ TIFF stack をドラッグ＆ドロップで左ビューに読み込んでください。")

    def update_patch_size(self, value):
        if value % 2 == 1:
            self.patch_size = value

    def setup_control_panel(self, layout):
        control_panel = QGroupBox("Point Movement Controller")
        control_layout = QVBoxLayout()

        self.move_step = 1
        self.coarse = False

        toggle_btn = QPushButton("👁️ Show Interpolated Points")
        toggle_btn.setCheckable(True)
        toggle_btn.clicked.connect(self.toggle_interpolated_visibility)
        control_layout.addWidget(toggle_btn)

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        start_end_layout = QHBoxLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setPrefix("Start Frame: ")
        self.spin_start.setMaximum(10000)
        start_end_layout.addWidget(self.spin_start)

        self.spin_end = QSpinBox()
        self.spin_end.setPrefix("End Frame: ")
        self.spin_end.setMaximum(10000)
        start_end_layout.addWidget(self.spin_end)

        select_all_btn = QPushButton("Select All Frames")
        select_all_btn.clicked.connect(self.select_all_frames)
        start_end_layout.addWidget(select_all_btn)
        control_layout.addLayout(start_end_layout)

        # 2段目（🗑️ Delete Positions）
        delete_btn = QPushButton("🗑️ Delete Positions in Range")
        delete_btn.clicked.connect(self.delete_positions_in_range)
        control_layout.addWidget(delete_btn)

        # 3段目（移動コントローラーとスピード）
        move_speed_layout = QVBoxLayout()

        arrow_layout = QVBoxLayout()
        up_button = QPushButton("↑")
        up_button.clicked.connect(lambda: self.move_points(0, -1))
        arrow_layout.addWidget(up_button)

        center_layout = QHBoxLayout()
        left_button = QPushButton("←")
        left_button.clicked.connect(lambda: self.move_points(-1, 0))
        center_layout.addWidget(left_button)

        down_button = QPushButton("↓")
        down_button.clicked.connect(lambda: self.move_points(0, 1))
        center_layout.addWidget(down_button)

        right_button = QPushButton("→")
        right_button.clicked.connect(lambda: self.move_points(1, 0))
        center_layout.addWidget(right_button)

        arrow_layout.addLayout(center_layout)

        speed_button_layout = QHBoxLayout()
        self.coarse_btn = QPushButton("Coarse")
        self.coarse_btn.setCheckable(True)
        self.coarse_btn.clicked.connect(self.toggle_coarse)
        speed_button_layout.addWidget(self.coarse_btn)

        self.maxspeed_btn = QPushButton("MAX SPEED!!")
        self.maxspeed_btn.setCheckable(True)
        self.maxspeed_btn.clicked.connect(self.toggle_maxspeed)
        speed_button_layout.addWidget(self.maxspeed_btn)

        arrow_layout.addLayout(speed_button_layout)
        move_speed_layout.addLayout(arrow_layout)
        control_layout.addLayout(move_speed_layout)

        # 4段目（Patch Size + Run）
        patch_run_layout = QVBoxLayout()
        self.patch_spin = QSpinBox()
        self.patch_spin.setPrefix("Patch Size: ")
        self.patch_spin.setRange(5, 21)
        self.patch_spin.setSingleStep(2)
        self.patch_spin.setValue(self.patch_size)
        self.patch_spin.valueChanged.connect(self.update_patch_size)
        patch_run_layout.addWidget(self.patch_spin)

        refine_btn = QPushButton("📌 Run Gaussian Drift Correction")
        refine_btn.clicked.connect(self.update_refined_stack)
        patch_run_layout.addWidget(refine_btn)
        control_layout.addLayout(patch_run_layout)

        # 5段目（Export）
        export_layout = QVBoxLayout()
        export_btn1 = QPushButton("📤 Export Tentative Points")
        export_btn1.clicked.connect(self.export_csv_tentative)
        export_layout.addWidget(export_btn1)

        export_btn2 = QPushButton("📤 Export Gaussian Fit Centers")
        export_btn2.clicked.connect(self.export_csv_refined)
        export_layout.addWidget(export_btn2)
        control_layout.addLayout(export_layout)

        # 6段目（Save）
        save_layout = QVBoxLayout()
        save_btn1 = QPushButton("💾 Save Tentative Corrected Stack")
        save_btn1.clicked.connect(self.save_corrected_stack)
        save_layout.addWidget(save_btn1)

        save_btn2 = QPushButton("💾 Save Gaussian Corrected Stack")
        save_btn2.clicked.connect(self.save_refined_stack)
        save_layout.addWidget(save_btn2)
        control_layout.addLayout(save_layout)
        
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

    def toggle_interpolated_visibility(self, checked):
        self.interp_layer.visible = checked

    def delete_positions_in_range(self):
        start = self.spin_start.value()
        end = self.spin_end.value()

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete positions from frame {start} to {end}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        points = self.point_layer.data.copy()
        filtered_points = np.array([p for p in points if not (start <= int(p[0]) <= end)])
        self.point_layer.data = filtered_points

    def export_csv_tentative(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Tentative Points CSV", "", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.point_layer.data, columns=["frame", "y", "x"])
            df.to_csv(path, index=False)

    def export_csv_refined(self):
        if not self.refined_points:
            print("❌ Refined points are not available. Run Gaussian Correction first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Gaussian Fit Centers CSV", "", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.refined_points, columns=["frame", "y", "x"])
            df.to_csv(path, index=False)

    def save_corrected_stack(self):
        if "corrected_stack" not in self.viewer_corr.layers:
            print("❌ Tentative corrected stack not found.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Tentative Corrected Stack", "", "TIFF files (*.tif)")
        if path:
            stack = self.viewer_corr.layers["corrected_stack"].data
            imsave(path, stack.astype("float32"))
            print(f"✅ Saved tentative corrected stack to: {path}")

    def save_refined_stack(self):
        if "refined_stack" not in self.viewer_refined.layers:
            print("❌ Gaussian corrected stack not found.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Gaussian Corrected Stack", "", "TIFF files (*.tif)")
        if path:
            stack = self.viewer_refined.layers["refined_stack"].data
            imsave(path, stack.astype("float32"))
            print(f"✅ Saved Gaussian corrected stack to: {path}")


        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

    def toggle_coarse(self, checked):
        self.coarse = checked
        if checked:
            self.move_step = 8
            self.maxspeed_btn.setChecked(False)
        else:
            self.move_step = 1

    def toggle_maxspeed(self, checked):
        if checked:
            self.move_step = 50
            self.coarse_btn.setChecked(False)
        else:
            self.move_step = 1

    def move_points(self, dx, dy):
        start = self.spin_start.value()
        end = self.spin_end.value()
        points = self.point_layer.data.copy()

        for i in range(len(points)):
            frame = int(points[i][0])
            if start <= frame <= end:
                points[i][1] += dy * self.move_step
                points[i][2] += dx * self.move_step

        self.point_layer.data = points

    def get_input_images(self):
        for layer in self.viewer_raw.layers:
            if hasattr(layer, "data") and layer.name != "atom_positions":
                return layer.data
        return None


    def update_corrected(self):
        raw_points = self.point_layer.data
        if len(raw_points) < 2:
            return
    
        # 実処理用補間 + 補間点をinterpolatedレイヤーに表示
        interp_points = self.interpolate_points_only_for_processing(raw_points)
        self.interp_layer.data = np.array([p for p in interp_points if p.tolist() not in raw_points.tolist()])
    
        images = self.get_input_images()
        if images is None or images.ndim != 3:
            return
    
        points = interp_points
        df = pd.DataFrame(points, columns=["frame", "y", "x"])
        df["frame"] = df["frame"].astype(int)
        df = df.sort_values("frame")

        try:
            ref_x, ref_y = df.iloc[0]["x"], df.iloc[0]["y"]
        except IndexError:
            return

        H, W = images.shape[1:]
        shifts, starts = [], []

        for i in range(max(df["frame"]) + 1):
            row = df[df["frame"] == i]
            if row.empty:
                dx, dy = 0, 0
            else:
                dx = int(round(row.iloc[0]["x"] - ref_x))
                dy = int(round(row.iloc[0]["y"] - ref_y))
            x_start, y_start = -dx, -dy
            shifts.append((dx, dy))
            starts.append((y_start, x_start))

        max_y = max(y + H for y, _ in starts)
        max_x = max(x + W for _, x in starts)
        min_y = min(y for y, _ in starts)
        min_x = min(x for _, x in starts)
        canvas_shape = (max_y - min_y, max_x - min_x)

        aligned_images = []
        for img, (dy, dx), (y_start, x_start) in zip(images, shifts, starts):
            canvas = np.zeros(canvas_shape, dtype=img.dtype)
            ys = y_start - min_y
            xs = x_start - min_x
            canvas[ys:ys + H, xs:xs + W] = img
            aligned_images.append(canvas)

        aligned_stack = np.stack(aligned_images, axis=0)

        if "corrected_stack" in self.viewer_corr.layers:
            self.viewer_corr.layers["corrected_stack"].data = aligned_stack
        else:
            self.viewer_corr.add_image(aligned_stack, name="corrected_stack")

    def update_refined_stack(self):
        images = self.get_input_images()
        if images is None or images.ndim != 3:
            return
    
        H, W = images.shape[1:]
        points = self.interpolate_points_only_for_processing(self.point_layer.data)
        df = pd.DataFrame(points, columns=["frame", "y", "x"])  # ← 修正済み
        df["frame"] = df["frame"].astype(int)
        df = df.sort_values("frame")

        try:
            ref_frame = df["frame"].min()
            ref_row = df[df["frame"] == ref_frame].iloc[0]
            ref_x, ref_y = ref_row["x"], ref_row["y"]
        except:
            return

        shifts, starts = [], []
        half = self.patch_size // 2
        for i in range(len(images)):
            row = df[df["frame"] == i]
            if row.empty:
                dx, dy = 0, 0
            else:
                y0, x0 = int(round(row.iloc[0]["y"])), int(round(row.iloc[0]["x"]))
                if half <= y0 < H - half and half <= x0 < W - half:
                    patch = images[i][y0-half:y0+half+1, x0-half:x0+half+1]
                    xg, yg = np.meshgrid(np.arange(self.patch_size), np.arange(self.patch_size))
                    try:
                        p0 = (patch.max(), half, half, 1.0, 1.0, patch.min())
                        popt, _ = curve_fit(self.gaussian_2d, (xg.ravel(), yg.ravel()), patch.ravel(), p0=p0)
                        fit_x, fit_y = popt[1], popt[2]
                        dx = int(round(fit_x - half - (ref_x - x0)))
                        dy = int(round(fit_y - half - (ref_y - y0)))
                    except:
                        dx, dy = 0, 0
                else:
                    dx, dy = 0, 0
            x_start, y_start = -dx, -dy
            shifts.append((dx, dy))
            starts.append((y_start, x_start))

        max_y = max(y + H for y, _ in starts)
        max_x = max(x + W for _, x in starts)
        min_y = min(y for y, _ in starts)
        min_x = min(x for _, x in starts)
        canvas_H = max_y - min_y
        canvas_W = max_x - min_x

        aligned_images = []
        for i, (img, (dy, dx), (y_start, x_start)) in enumerate(zip(images, shifts, starts)):
            canvas = np.zeros((canvas_H, canvas_W), dtype=img.dtype)
            ys = y_start - min_y
            xs = x_start - min_x
            canvas[ys:ys + H, xs:xs + W] = img
            aligned_images.append(canvas)

        self.refined_points = []
        for frame, row in df.groupby("frame").first().iterrows():
            y0, x0 = int(round(row.y)), int(round(row.x))
            if half <= y0 < H - half and half <= x0 < W - half:
                patch = images[frame][y0-half:y0+half+1, x0-half:x0+half+1]
                xg, yg = np.meshgrid(np.arange(self.patch_size), np.arange(self.patch_size))
                try:
                    p0 = (patch.max(), half, half, 1.0, 1.0, patch.min())
                    popt, _ = curve_fit(self.gaussian_2d, (xg.ravel(), yg.ravel()), patch.ravel(), p0=p0)
                    fit_x, fit_y = x0 + (popt[1] - half), y0 + (popt[2] - half)
                    self.refined_points.append([frame, fit_y, fit_x])
                except:
                    self.refined_points.append([frame, y0, x0])
            else:
                self.refined_points.append([frame, y0, x0])

        refined_stack = np.stack(aligned_images, axis=0)

        if "refined_stack" in self.viewer_refined.layers:
            self.viewer_refined.layers["refined_stack"].data = refined_stack
        else:
            self.viewer_refined.add_image(refined_stack, name="refined_stack")

    def gaussian_2d(self, coords, A, x0, y0, sigma_x, sigma_y, offset):
        x, y = coords
        exp_x = ((x - x0) ** 2) / (2 * sigma_x ** 2)
        exp_y = ((y - y0) ** 2) / (2 * sigma_y ** 2)
        return A * np.exp(-(exp_x + exp_y)) + offset

    def select_all_frames(self):
        images = self.get_input_images()
        if images is not None and images.ndim == 3:
            max_frame = images.shape[0] - 1
            self.spin_start.setValue(0)
            self.spin_end.setValue(max_frame)
            
    def interpolate_missing_points(self):
        from contextlib import suppress
        with suppress(Exception):
            points = self.point_layer.data
            if len(points) < 2:
                return
    
            df = pd.DataFrame(points, columns=["frame", "y", "x"])
            df["frame"] = df["frame"].astype(int)
            df = df.sort_values("frame")
            all_frames = np.arange(df["frame"].min(), df["frame"].max() + 1)
    
            existing_frames = df["frame"].values
            missing_frames = sorted(set(all_frames) - set(existing_frames))
    
            interp_y = np.interp(missing_frames, df["frame"], df["y"])
            interp_x = np.interp(missing_frames, df["frame"], df["x"])
    
            interpolated = np.array([[f, y, x] for f, y, x in zip(missing_frames, interp_y, interp_x)])
            if len(interpolated) > 0:
                new_data = np.vstack([points, interpolated])
                new_data = new_data[np.argsort(new_data[:, 0])]
    
                # update_corrected のループ防止
                self.point_layer.events.data.block()
                self.point_layer.data = new_data
                self.point_layer.events.data.unblock()
                
    def interpolate_points_only_for_processing(self, raw_points):
        if len(raw_points) < 2:
            return raw_points

        df = pd.DataFrame(raw_points, columns=["frame", "y", "x"])
        df["frame"] = df["frame"].astype(int)
        df = df.sort_values("frame")
        all_frames = np.arange(df["frame"].min(), df["frame"].max() + 1)

        existing_frames = df["frame"].values
        missing_frames = sorted(set(all_frames) - set(existing_frames))

        interp_y = np.interp(missing_frames, df["frame"], df["y"])
        interp_x = np.interp(missing_frames, df["frame"], df["x"])

        interpolated = np.array([[f, y, x] for f, y, x in zip(missing_frames, interp_y, interp_x)])
        if len(interpolated) > 0:
            combined = np.vstack([raw_points, interpolated])
            return combined[np.argsort(combined[:, 0])]
        else:
            return raw_points




app = QApplication(sys.argv)
window = DualViewerWindow()
window.show()
app.exec_()
