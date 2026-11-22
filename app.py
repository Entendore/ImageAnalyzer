#!/usr/bin/env python3
"""
Image Analyzer Pro — A comprehensive PySide6 application for image analysis.
Features: LUTs/Color Maps, Histograms, Eigenvalues/Eigenvectors, 
Height Maps, Heat Maps, Statistics, Filters, Intensity Profiles, and Adjustments.
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTabWidget, QFileDialog,
    QGroupBox, QGridLayout, QSplitter, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,  # Removed QAction from here
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QAction  # Added QAction here
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from scipy import ndimage
from scipy.linalg import eigh

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORMAP_LUTS = {
    "Viridis": "viridis", "Plasma": "plasma", "Inferno": "inferno",
    "Magma": "magma", "Cividis": "cividis", "Jet": "jet", "Hot": "hot",
    "Cool": "cool", "Spring": "spring", "Summer": "summer",
    "Autumn": "autumn", "Winter": "winter", "Bone": "bone",
    "Copper": "copper", "Pink": "pink", "Twilight": "twilight",
    "Turbo": "turbo", "HSV Map": "hsv", "Rainbow": "rainbow",
    "Ocean": "ocean", "Terrain": "terrain",
}

STYLESHEET = """
QMainWindow{background:#1e1e1e}
QWidget{background:#2b2b2b;color:#ddd;font-size:12px}
QGroupBox{border:1px solid #555;border-radius:4px;margin-top:8px;
          padding-top:14px;font-weight:bold;color:#8af}
QGroupBox::title{subcontrol-origin:margin;left:10px}
QPushButton{background:#3a3a3a;border:1px solid #555;border-radius:4px;
            padding:6px 14px;color:#ddd}
QPushButton:hover{background:#4a4a4a;border-color:#777}
QPushButton:pressed{background:#555}
QComboBox{background:#3a3a3a;border:1px solid #555;border-radius:4px;
          padding:4px 8px;color:#ddd}
QComboBox::drop-down{border:none}
QComboBox QAbstractItemView{background:#3a3a3a;color:#ddd;selection-background-color:#555}
QSpinBox,QDoubleSpinBox{background:#3a3a3a;border:1px solid #555;
                         border-radius:4px;padding:4px;color:#ddd}
QSlider::groove:horizontal{background:#555;height:6px;border-radius:3px}
QSlider::handle:horizontal{background:#8af;width:16px;margin:-5px 0;border-radius:8px}
QTabWidget::pane{border:1px solid #555;background:#2b2b2b}
QTabBar::tab{background:#333;border:1px solid #555;padding:8px 14px;
             border-top-left-radius:4px;border-top-right-radius:4px;color:#999}
QTabBar::tab:selected{background:#444;color:#fff;border-bottom-color:#444}
QTableWidget{background:#333;gridline-color:#555;color:#ddd;border:1px solid #555}
QTableWidget::item{padding:3px}
QHeaderView::section{background:#3a3a3a;border:1px solid #555;padding:4px;color:#aaa}
QScrollArea{border:none}
QStatusBar{background:#1e1e1e;color:#888}
QMenuBar{background:#252525;color:#ddd}
QMenuBar::item:selected{background:#3a3a3a}
QMenu{background:#2b2b2b;color:#ddd;border:1px solid #444}
QMenu::item:selected{background:#3a3a3a}
QCheckBox{color:#ddd;spacing:6px}
QLabel{color:#ccc}
QSplitter::handle{background:#444;width:3px}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(img):
    """Compute luminance from an RGB or gray image."""
    if img.ndim == 2:
        return img.astype(np.float64)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def rgb_to_hsv(rgb):
    """Convert uint8 RGB array → (H, S, V) float arrays."""
    f = rgb.astype(np.float64) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cmax = np.max(f, axis=-1)
    cmin = np.min(f, axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta != 0
    mr = mask & (cmax == r)
    mg = mask & (cmax == g) & ~mr
    mb = mask & (cmax == b) & ~mr & ~mg
    h[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60.0 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    h[mb] = 60.0 * (((r[mb] - g[mb]) / delta[mb]) + 4)
    s = np.zeros_like(cmax)
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    v = cmax
    return h, s, v


def numpy_to_qpixmap(arr):
    """Convert an H×W×3 uint8 numpy array to a QPixmap."""
    if arr is None:
        return QPixmap()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def simple_kmeans(data, k, max_iter=20, sample_size=5000):
    """Minimal K-means for dominant-color extraction."""
    if len(data) > sample_size:
        idx = np.random.choice(len(data), sample_size, replace=False)
        sample = data[idx].astype(np.float64)
    else:
        sample = data.astype(np.float64)
    ci = np.random.choice(len(sample), k, replace=False)
    centers = sample[ci].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=-1)
        labels = np.argmin(dists, axis=1)
        new_c = np.zeros_like(centers)
        for i in range(k):
            m = labels == i
            new_c[i] = sample[m].mean(axis=0) if m.any() else centers[i]
        if np.allclose(centers, new_c, atol=0.5):
            break
        centers = new_c
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    return centers[order].astype(np.uint8), counts[order]


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM LABEL — pixel hover tracking
# ═══════════════════════════════════════════════════════════════════════════════

class ImageDisplayLabel(QLabel):
    pixel_hover = Signal(int, int, int, int, int)   # x, y, r, g, b

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._img = None
        self._rect = None          # (x, y, w, h) of pixmap inside label

    def set_image_data(self, data):
        self._img = data

    def set_pixmap_with_rect(self, pixmap):
        self.setPixmap(pixmap)
        if pixmap and not pixmap.isNull():
            lw, lh = self.width(), self.height()
            pw, ph = pixmap.width(), pixmap.height()
            self._rect = ((lw - pw) // 2, (lh - ph) // 2, pw, ph)
        else:
            self._rect = None

    def mouseMoveEvent(self, event):
        if self._img is not None and self._rect is not None:
            pos = event.position().toPoint()
            px, py, pw, ph = self._rect
            rx = (pos.x() - px) / pw
            ry = (pos.y() - py) / ph
            h, w = self._img.shape[:2]
            ix, iy = int(rx * w), int(ry * h)
            if 0 <= ix < w and 0 <= iy < h:
                px_val = self._img[iy, ix]
                if self._img.ndim == 3:
                    r, g, b = int(px_val[0]), int(px_val[1]), int(px_val[2])
                else:
                    r = g = b = int(px_val)
                self.pixel_hover.emit(ix, iy, r, g, b)
        super().mouseMoveEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analyzer Pro")
        self.setMinimumSize(1400, 900)
        self.original_image = None      # H×W×3  uint8
        self.processed_image = None
        self.zoom_factor = 1.0
        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hbox(*widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for ww in widgets:
            h.addWidget(ww)
        return w

    def _make_canvas(self, figsize=(8, 5)):
        fig = Figure(figsize=figsize, facecolor="#2b2b2b")
        canvas = FigureCanvas(fig)
        return fig, canvas

    @staticmethod
    def _cen_item(text):
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignCenter)
        return it

    # ── menu ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        for label, sc, slot in [
            ("Open Image",       "Ctrl+O", self.open_image),
            ("Save Processed",   "Ctrl+S", self.save_image),
            ("Exit",             "Ctrl+Q", self.close),
        ]:
            a = QAction(label, self); a.setShortcut(sc); a.triggered.connect(slot)
            fm.addAction(a)
            if label == "Save Processed":
                fm.addSeparator()
        am = mb.addMenu("Analysis")
        a = QAction("Analyze All", self); a.setShortcut("F5")
        a.triggered.connect(self.analyze_all); am.addAction(a)

    # ── main layout ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_menu()
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal); root.addWidget(splitter)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(4,4,4,4)

        self.img_label = ImageDisplayLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(480, 380)
        self.img_label.setText("Load an image to begin\nFile → Open Image  (Ctrl+O)")
        self.img_label.setStyleSheet(
            "QLabel{background:#1a1a1a;color:#555;border:2px dashed #444;"
            "border-radius:8px;font-size:15px}")
        self.img_label.pixel_hover.connect(self._on_pixel_hover)
        ll.addWidget(self.img_label, stretch=1)

        self.pixel_info = QLabel("Pixel: —")
        self.pixel_info.setStyleSheet("color:#888;font-size:11px")
        ll.addWidget(self.pixel_info)

        # LUT group
        lg = QGroupBox("LUT / Color Map"); gl = QGridLayout(lg)
        self.lut_combo = QComboBox()
        self.lut_combo.addItems([
            "Original", "Grayscale", "Invert", "Sepia",
            *COLORMAP_LUTS.keys(),
            "Threshold", "Posterize", "Solarize",
        ])
        self.lut_combo.currentTextChanged.connect(self._on_lut_changed)

        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 255); self.thresh_slider.setValue(128)
        self.thresh_slider.valueChanged.connect(self._apply_lut)
        self.thresh_lbl = QLabel("128")
        self.thresh_w = self._hbox(QLabel("Threshold:"), self.thresh_slider, self.thresh_lbl)
        self.thresh_w.setVisible(False)

        self.poster_spin = QSpinBox(); self.poster_spin.setRange(2, 16); self.poster_spin.setValue(4)
        self.poster_spin.valueChanged.connect(self._apply_lut)
        self.poster_w = self._hbox(QLabel("Levels:"), self.poster_spin)
        self.poster_w.setVisible(False)

        self.solar_slider = QSlider(Qt.Horizontal)
        self.solar_slider.setRange(0, 255); self.solar_slider.setValue(128)
        self.solar_slider.valueChanged.connect(self._apply_lut)
        self.solar_lbl = QLabel("128")
        self.solar_w = self._hbox(QLabel("Threshold:"), self.solar_slider, self.solar_lbl)
        self.solar_w.setVisible(False)

        gl.addWidget(QLabel("LUT:"), 0, 0); gl.addWidget(self.lut_combo, 0, 1, 1, 3)
        gl.addWidget(self.thresh_w, 1, 0, 1, 4)
        gl.addWidget(self.poster_w, 1, 0, 1, 4)
        gl.addWidget(self.solar_w, 1, 0, 1, 4)
        ll.addWidget(lg)

        # Brightness / Contrast / Gamma
        adj = QGroupBox("Adjustments"); al = QGridLayout(adj)
        self.bright_slider = QSlider(Qt.Horizontal); self.bright_slider.setRange(-100,100); self.bright_slider.setValue(0)
        self.contrast_slider = QSlider(Qt.Horizontal); self.contrast_slider.setRange(-100,100); self.contrast_slider.setValue(0)
        self.gamma_slider = QSlider(Qt.Horizontal); self.gamma_slider.setRange(10,300); self.gamma_slider.setValue(100)
        self.bright_lbl = QLabel("0"); self.contrast_lbl = QLabel("0"); self.gamma_lbl = QLabel("1.00")
        for s in (self.bright_slider, self.contrast_slider, self.gamma_slider):
            s.valueChanged.connect(self._apply_lut)
        al.addWidget(QLabel("Brightness:"), 0, 0); al.addWidget(self.bright_slider, 0, 1); al.addWidget(self.bright_lbl, 0, 2)
        al.addWidget(QLabel("Contrast:"), 1, 0); al.addWidget(self.contrast_slider, 1, 1); al.addWidget(self.contrast_lbl, 1, 2)
        al.addWidget(QLabel("Gamma:"), 2, 0); al.addWidget(self.gamma_slider, 2, 1); al.addWidget(self.gamma_lbl, 2, 2)
        ll.addWidget(adj)

        # Nav buttons
        nav = QHBoxLayout()
        for txt, slot in [
            ("🔍+ Zoom In", self.zoom_in), ("🔍- Zoom Out", self.zoom_out),
            ("⊞ Fit", self.fit_image), ("↺ Reset", self.reset_lut),
            ("⚡ Analyze All", self.analyze_all),
        ]:
            b = QPushButton(txt); b.clicked.connect(slot); nav.addWidget(b)
        ll.addLayout(nav)
        splitter.addWidget(left)

        # ── RIGHT PANEL ───────────────────────────────────────────────────
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(4,4,4,4)
        self.tabs = QTabWidget(); rl.addWidget(self.tabs)

        self._tab_histogram()
        self._tab_eigen()
        self._tab_heightmap()
        self._tab_heatmap()
        self._tab_statistics()
        self._tab_filters()
        self._tab_profile()

        splitter.addWidget(right)
        splitter.setSizes([520, 780])
        self.statusBar().showMessage("Ready — open an image to begin analysis")

    # ── tab builders ──────────────────────────────────────────────────────

    def _tab_histogram(self):
        t = QWidget(); ly = QVBoxLayout(t)
        c = QHBoxLayout()
        self.hist_rgb = QCheckBox("RGB"); self.hist_rgb.setChecked(True)
        self.hist_lum = QCheckBox("Luminance"); self.hist_lum.setChecked(True)
        self.hist_bins = QSpinBox(); self.hist_bins.setRange(16,512); self.hist_bins.setValue(256)
        b = QPushButton("Refresh"); b.clicked.connect(self.update_histogram)
        c.addWidget(self.hist_rgb); c.addWidget(self.hist_lum)
        c.addWidget(QLabel("Bins:")); c.addWidget(self.hist_bins); c.addWidget(b)
        ly.addLayout(c)
        self.hist_fig, self.hist_cv = self._make_canvas((8,6))
        ly.addWidget(self.hist_cv)
        self.tabs.addTab(t, "📊 Histogram")

    def _tab_eigen(self):
        t = QWidget(); ly = QVBoxLayout(t)
        ly.addWidget(QLabel(
            "Eigen analysis of the RGB covariance matrix — each pixel is a 3-D point (R, G, B)."
        ))
        b = QPushButton("Compute Eigen Analysis"); b.clicked.connect(self.update_eigen)
        ly.addWidget(b)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sw = QWidget(); sl = QVBoxLayout(sw)

        for attr, title, rows, cols, hhdr, vhdr in [
            ("cov_tbl",  "Covariance Matrix", 3, 3, ["R","G","B"], ["R","G","B"]),
            ("eigval_tbl","Eigenvalues",       3, 3, ["#","Eigenvalue","Var %"], None),
            ("eigvec_tbl","Eigenvectors",      3, 4, ["#","R","G","B"], None),
        ]:
            g = QGroupBox(title); gl = QVBoxLayout(g)
            tbl = QTableWidget(rows, cols); tbl.setHorizontalHeaderLabels(hhdr)
            if vhdr: tbl.setVerticalHeaderLabels(vhdr)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            gl.addWidget(tbl); sl.addWidget(g)
            setattr(self, attr, tbl)

        self.eigen_fig, self.eigen_cv = self._make_canvas((8,4))
        sl.addWidget(self.eigen_cv)
        sc.setWidget(sw); ly.addWidget(sc)
        self.tabs.addTab(t, "🔢 Eigen Analysis")

    def _tab_heightmap(self):
        t = QWidget(); ly = QVBoxLayout(t)
        c = QHBoxLayout()
        self.hm_cmap = QComboBox()
        self.hm_cmap.addItems(["terrain","viridis","plasma","gray","coolwarm",
                                "RdYlGn","gist_earth","copper","jet"])
        self.hm_elev = QSpinBox(); self.hm_elev.setRange(10,90); self.hm_elev.setValue(30)
        self.hm_azim = QSpinBox(); self.hm_azim.setRange(0,360); self.hm_azim.setValue(315)
        self.hm_ds   = QSpinBox(); self.hm_ds.setRange(1,32);   self.hm_ds.setValue(4)
        self.hm_cmap.currentTextChanged.connect(self.update_heightmap)
        for w in (self.hm_elev, self.hm_azim, self.hm_ds):
            w.valueChanged.connect(self.update_heightmap)
        for lbl, w in [("Cmap:",self.hm_cmap),("Elev:",self.hm_elev),
                        ("Azim:",self.hm_azim),("Downsample:",self.hm_ds)]:
            c.addWidget(QLabel(lbl)); c.addWidget(w)
        ly.addLayout(c)
        self.hm_fig, self.hm_cv = self._make_canvas((8,6))
        ly.addWidget(self.hm_cv)
        self.tabs.addTab(t, "⛰ Height Map")

    def _tab_heatmap(self):
        t = QWidget(); ly = QVBoxLayout(t)
        c = QHBoxLayout()
        self.ht_cmap = QComboBox()
        self.ht_cmap.addItems(["hot","inferno","plasma","magma","viridis","jet",
                                "coolwarm","RdYlBu","seismic","twilight","turbo",
                                "rainbow","gnuplot2","CMRmap"])
        self.ht_ch = QComboBox()
        self.ht_ch.addItems(["Luminance","Red","Green","Blue","Hue","Saturation","Value"])
        self.ht_interp = QComboBox()
        self.ht_interp.addItems(["nearest","bilinear","bicubic","gaussian"])
        for w in (self.ht_cmap, self.ht_ch, self.ht_interp):
            w.currentTextChanged.connect(self.update_heatmap)
        for lbl, w in [("Cmap:",self.ht_cmap),("Channel:",self.ht_ch),
                        ("Interp:",self.ht_interp)]:
            c.addWidget(QLabel(lbl)); c.addWidget(w)
        ly.addLayout(c)
        self.ht_fig, self.ht_cv = self._make_canvas((8,6))
        ly.addWidget(self.ht_cv)
        self.tabs.addTab(t, "🌡 Heat Map")

    def _tab_statistics(self):
        t = QWidget(); ly = QVBoxLayout(t)
        b = QPushButton("Compute Statistics"); b.clicked.connect(self.update_statistics)
        ly.addWidget(b)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        sw = QWidget(); sl = QVBoxLayout(sw)

        for attr, title, rows, cols, hhdr, vhdr in [
            ("ch_tbl","Channel Statistics",8,4,
             ["Stat","R","G","B"],["Mean","Std","Min","Max","Median","Var","Skew","Kurt"]),
            ("lum_tbl","Luminance Statistics",8,2,
             ["Stat","Value"],["Mean","Std","Min","Max","Median","Var","DynRange","Contrast"]),
            ("prop_tbl","Image Properties",7,2,
             ["Property","Value"],["W","H","Ch","Dtype","Pixels","Aspect","Size KB"]),
        ]:
            g = QGroupBox(title); gl = QVBoxLayout(g)
            tbl = QTableWidget(rows,cols); tbl.setHorizontalHeaderLabels(hhdr)
            if vhdr: tbl.setVerticalHeaderLabels(vhdr)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            gl.addWidget(tbl); sl.addWidget(g)
            setattr(self, attr, tbl)

        g = QGroupBox("Dominant Colors (K-Means k=8)"); gl = QVBoxLayout(g)
        self.dom_fig, self.dom_cv = self._make_canvas((8,2.5))
        gl.addWidget(self.dom_cv); sl.addWidget(g)
        sc.setWidget(sw); ly.addWidget(sc)
        self.tabs.addTab(t, "📈 Statistics")

    def _tab_filters(self):
        t = QWidget(); ly = QVBoxLayout(t)
        c = QHBoxLayout()
        self.filt_combo = QComboBox()
        self.filt_combo.addItems([
            "Sobel X","Sobel Y","Sobel Combined","Laplacian","Gaussian Blur",
            "Sharpen","Emboss","Edge Enhance","Prewitt","Dilation","Erosion",
            "Median Filter","Unsharp Mask",
        ])
        self.filt_param = QDoubleSpinBox()
        self.filt_param.setRange(0.1,20); self.filt_param.setValue(1); self.filt_param.setSingleStep(0.5)
        b = QPushButton("Apply"); b.clicked.connect(self.apply_filter)
        c.addWidget(QLabel("Filter:")); c.addWidget(self.filt_combo,stretch=1)
        c.addWidget(QLabel("σ/Size:")); c.addWidget(self.filt_param); c.addWidget(b)
        ly.addLayout(c)
        self.filt_fig, self.filt_cv = self._make_canvas((8,6))
        ly.addWidget(self.filt_cv)
        self.tabs.addTab(t, "🎛 Filters")

    def _tab_profile(self):
        t = QWidget(); ly = QVBoxLayout(t)
        c = QHBoxLayout()
        self.prof_row = QSpinBox(); self.prof_row.setRange(0,99999); self.prof_row.setValue(0)
        self.prof_col = QSpinBox(); self.prof_col.setRange(0,99999); self.prof_col.setValue(0)
        b = QPushButton("Plot Profiles"); b.clicked.connect(self.update_profile)
        c.addWidget(QLabel("Row:")); c.addWidget(self.prof_row)
        c.addWidget(QLabel("Col:")); c.addWidget(self.prof_col)
        c.addWidget(b)
        ly.addLayout(c)
        self.prof_fig, self.prof_cv = self._make_canvas((8,5))
        ly.addWidget(self.prof_cv)
        self.tabs.addTab(t, "📏 Profile")

    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE I/O
    # ══════════════════════════════════════════════════════════════════════════

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp);;All (*)")
        if not path:
            return
        qimg = QImage(path)
        if qimg.isNull():
            QMessageBox.warning(self, "Error", f"Cannot load:\n{path}"); return
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()  # Bytes per line (may include padding)
        
        # Get raw bytes - compatible with both old and new PySide6
        ptr = qimg.constBits()
        if isinstance(ptr, memoryview):
            raw = bytes(ptr)
        else:
            ptr.setsize(h * bpl)
            raw = bytes(ptr)
        
        # Handle potential row alignment/padding
        if bpl == w * 3:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        else:
            # Strip padding bytes from each row
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpl))[:, :w*3].reshape((h, w, 3)).copy()
        
        self.original_image = arr
        self.processed_image = arr.copy()
        self.lut_combo.setCurrentText("Original")
        self.bright_slider.setValue(0); self.contrast_slider.setValue(0); self.gamma_slider.setValue(100)
        self.zoom_factor = 1.0
        self.img_label.set_image_data(arr)
        self.prof_row.setMaximum(h - 1); self.prof_col.setMaximum(w - 1)
        self._display_image()
        self.statusBar().showMessage(f"Loaded {path}  ({w}×{h})")

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.information(self, "Info", "No image to save."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if not path:
            return
        arr = np.clip(self.processed_image, 0, 255).astype(np.uint8)
        h, w, _ = arr.shape
        qimg = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        qimg.save(path)
        self.statusBar().showMessage(f"Saved → {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  DISPLAY
    # ══════════════════════════════════════════════════════════════════════════

    def _display_image(self):
        if self.processed_image is None:
            return
        pm = numpy_to_qpixmap(self.processed_image)
        if self.zoom_factor == 1.0:
            pm = pm.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            nw = int(pm.width() * self.zoom_factor)
            nh = int(pm.height() * self.zoom_factor)
            pm = pm.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.set_pixmap_with_rect(pm)

    def _on_pixel_hover(self, x, y, r, g, b):
        lum = int(0.299*r + 0.587*g + 0.114*b)
        self.pixel_info.setText(
            f"({x}, {y})  R:{r} G:{g} B:{b}  L:{lum}  #{r:02X}{g:02X}{b:02X}")

    # ══════════════════════════════════════════════════════════════════════════
    #  ZOOM / NAV
    # ══════════════════════════════════════════════════════════════════════════

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.25, 8.0); self._display_image()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.1); self._display_image()

    def fit_image(self):
        self.zoom_factor = 1.0; self._display_image()

    def reset_lut(self):
        self.lut_combo.setCurrentText("Original")
        self.bright_slider.setValue(0); self.contrast_slider.setValue(0); self.gamma_slider.setValue(100)

    # ══════════════════════════════════════════════════════════════════════════
    #  LUT / ADJUSTMENTS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_lut_changed(self, text):
        self.thresh_w.setVisible(text == "Threshold")
        self.poster_w.setVisible(text == "Posterize")
        self.solar_w.setVisible(text == "Solarize")
        self._apply_lut()

    def _apply_lut(self):
        if self.original_image is None:
            return
        img = self.original_image.astype(np.float64)
        name = self.lut_combo.currentText()
        self.thresh_lbl.setText(str(self.thresh_slider.value()))
        self.solar_lbl.setText(str(self.solar_slider.value()))
        self.bright_lbl.setText(str(self.bright_slider.value()))
        self.contrast_lbl.setText(str(self.contrast_slider.value()))
        self.gamma_lbl.setText(f"{self.gamma_slider.value()/100:.2f}")

        # ── LUT ────────────────────
        if name == "Original":
            out = img
        elif name == "Grayscale":
            g = luminance(img).astype(np.uint8)
            out = np.stack([g, g, g], axis=-1).astype(np.float64)
        elif name == "Invert":
            out = 255 - img
        elif name == "Sepia":
            m = np.array([[.393,.769,.189],[.349,.686,.168],[.272,.534,.131]])
            out = img @ m.T
        elif name in COLORMAP_LUTS:
            g = luminance(img) / 255.0
            cmap = cm.get_cmap(COLORMAP_LUTS[name])
            out = (cmap(g)[..., :3] * 255)
        elif name == "Threshold":
            t = self.thresh_slider.value()
            g = (luminance(img) > t).astype(np.float64) * 255
            out = np.stack([g, g, g], axis=-1)
        elif name == "Posterize":
            lev = self.poster_spin.value()
            step = 256 // lev
            out = (img // step) * step + step // 2
        elif name == "Solarize":
            t = self.solar_slider.value()
            out = img.copy()
            out[img > t] = 255 - out[img > t]
        else:
            out = img

        # ── Adjustments ────────────
        bright = self.bright_slider.value()
        cont   = self.contrast_slider.value()
        gamma  = self.gamma_slider.value() / 100.0

        if bright != 0:
            out = out + bright
        if cont != 0:
            factor = (100 + cont) / 100.0
            out = (out - 128) * factor + 128
        if gamma != 1.0:
            out = 255 * np.power(np.clip(out / 255.0, 0, 1), 1.0 / gamma)

        self.processed_image = np.clip(out, 0, 255).astype(np.uint8)
        self._display_image()

    # ══════════════════════════════════════════════════════════════════════════
    #  HISTOGRAM
    # ══════════════════════════════════════════════════════════════════════════

    def update_histogram(self):
        if self.original_image is None:
            return
        img = self.original_image
        bins = self.hist_bins.value()
        show_rgb = self.hist_rgb.isChecked()
        show_lum = self.hist_lum.isChecked()
        n = int(show_rgb) + int(show_lum)
        if n == 0:
            self.hist_fig.clear(); self.hist_cv.draw(); return

        self.hist_fig.clear()
        idx = 1
        if show_rgb:
            ax = self.hist_fig.add_subplot(1, n, idx); idx += 1
            ax.set_facecolor("#1e1e1e")
            colors = ['#ff4444', '#44ff44', '#4488ff']
            labels = ['Red', 'Green', 'Blue']
            for c in range(3):
                hist, edges = np.histogram(img[..., c].ravel(), bins=bins, range=(0, 256))
                ax.fill_between(edges[:-1], hist, alpha=0.35, color=colors[c])
                ax.plot(edges[:-1], hist, color=colors[c], linewidth=1.2, label=labels[c])
            ax.set_title("RGB Histogram", color='#ddd', fontsize=11)
            ax.legend(fontsize=9, facecolor='#333', edgecolor='#555', labelcolor='#ddd')
            ax.tick_params(colors='#999')
            ax.set_xlabel("Intensity", color='#aaa', fontsize=9)
            ax.set_ylabel("Count", color='#aaa', fontsize=9)

        if show_lum:
            ax = self.hist_fig.add_subplot(1, n, idx)
            ax.set_facecolor("#1e1e1e")
            lum = luminance(img).ravel()
            hist, edges = np.histogram(lum, bins=bins, range=(0, 256))
            ax.fill_between(edges[:-1], hist, alpha=0.4, color='#ffaa33')
            ax.plot(edges[:-1], hist, color='#ffaa33', linewidth=1.2, label='Luminance')
            ax.set_title("Luminance Histogram", color='#ddd', fontsize=11)
            ax.legend(fontsize=9, facecolor='#333', edgecolor='#555', labelcolor='#ddd')
            ax.tick_params(colors='#999')
            ax.set_xlabel("Luminance", color='#aaa', fontsize=9)
            ax.set_ylabel("Count", color='#aaa', fontsize=9)

        self.hist_fig.tight_layout()
        self.hist_cv.draw()
        self.statusBar().showMessage("Histogram updated")

    # ══════════════════════════════════════════════════════════════════════════
    #  EIGEN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def update_eigen(self):
        if self.original_image is None:
            return
        img = self.original_image
        h, w = img.shape[:2]

        # Sample pixels for speed
        n_samples = min(50000, h * w)
        idx = np.random.choice(h * w, n_samples, replace=False)
        pixels = img.reshape(-1, 3).astype(np.float64)[idx]

        mean = pixels.mean(axis=0)
        centered = pixels - mean
        cov = np.cov(centered.T)

        # Fill covariance table
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem(f"{cov[i, j]:.2f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.cov_tbl.setItem(i, j, item)

        # Eigen decomposition
        eigvals, eigvecs = eigh(cov)
        order = np.argsort(-eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        total_var = eigvals.sum()

        for i in range(3):
            pct = (eigvals[i] / total_var * 100) if total_var > 0 else 0
            self.eigval_tbl.setItem(i, 0, self._cen_item(f"λ{i+1}"))
            self.eigval_tbl.setItem(i, 1, self._cen_item(f"{eigvals[i]:.2f}"))
            self.eigval_tbl.setItem(i, 2, self._cen_item(f"{pct:.1f}%"))
            self.eigvec_tbl.setItem(i, 0, self._cen_item(f"v{i+1}"))
            for j in range(3):
                self.eigvec_tbl.setItem(i, j + 1, self._cen_item(f"{eigvecs[j, i]:.4f}"))

        # Plots
        self.eigen_fig.clear()
        projected = centered @ eigvecs

        # PCA scatter
        ax1 = self.eigen_fig.add_subplot(121)
        ax1.set_facecolor("#1e1e1e")
        sub = np.random.choice(len(projected), min(3000, len(projected)), replace=False)
        ax1.scatter(projected[sub, 0], projected[sub, 1],
                    c=pixels[sub] / 255.0, s=4, alpha=0.6)
        ax1.set_title("PC1 vs PC2", color='#ddd', fontsize=10)
        ax1.set_xlabel(f"PC1 ({eigvals[0]/total_var*100:.1f}%)", color='#aaa', fontsize=8)
        ax1.set_ylabel(f"PC2 ({eigvals[1]/total_var*100:.1f}%)", color='#aaa', fontsize=8)
        ax1.tick_params(colors='#999')

        # Eigenvector bar chart
        ax2 = self.eigen_fig.add_subplot(122)
        ax2.set_facecolor("#1e1e1e")
        x = np.arange(3)
        bw = 0.25
        ch_colors = ['#ff4444', '#44ff44', '#4488ff']
        for i in range(3):
            ax2.bar(x + i * bw, eigvecs[:, i], bw,
                    label=f"v{i+1}", color=ch_colors[i], alpha=0.7)
        ax2.set_xticks(x + bw)
        ax2.set_xticklabels(["R", "G", "B"])
        ax2.set_title("Eigenvector Components", color='#ddd', fontsize=10)
        ax2.legend(fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='#ddd')
        ax2.tick_params(colors='#999')
        ax2.axhline(0, color='#555', linewidth=0.5)

        self.eigen_fig.tight_layout()
        self.eigen_cv.draw()
        self.statusBar().showMessage("Eigen analysis complete")

    # ══════════════════════════════════════════════════════════════════════════
    #  HEIGHT MAP
    # ══════════════════════════════════════════════════════════════════════════

    def update_heightmap(self):
        if self.original_image is None:
            return
        ds = max(1, self.hm_ds.value())
        gray = luminance(self.original_image)
        gray = gray[::ds, ::ds]

        self.hm_fig.clear()
        ax = self.hm_fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#1e1e1e")

        h, w = gray.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        cmap_name = self.hm_cmap.currentText()
        norm = gray / max(gray.max(), 1)
        cmap_obj = cm.get_cmap(cmap_name)
        colors = cmap_obj(norm)

        ax.plot_surface(X, Y, gray, facecolors=colors,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=True)
        ax.view_init(elev=self.hm_elev.value(), azim=self.hm_azim.value())
        ax.set_title("Height Map (intensity → elevation)", color='#ddd', fontsize=10)
        ax.tick_params(colors='#999', labelsize=7)
        ax.set_xlabel("X", color='#aaa', fontsize=8)
        ax.set_ylabel("Y", color='#aaa', fontsize=8)
        ax.set_zlabel("Intensity", color='#aaa', fontsize=8)
        try:
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor('#444')
        except Exception:
            pass

        self.hm_fig.tight_layout()
        self.hm_cv.draw()
        self.statusBar().showMessage("Height map updated")

    # ══════════════════════════════════════════════════════════════════════════
    #  HEAT MAP
    # ══════════════════════════════════════════════════════════════════════════

    def update_heatmap(self):
        if self.original_image is None:
            return
        img = self.original_image
        ch = self.ht_ch.currentText()

        if ch == "Luminance":
            data = luminance(img)
        elif ch == "Red":
            data = img[..., 0].astype(np.float64)
        elif ch == "Green":
            data = img[..., 1].astype(np.float64)
        elif ch == "Blue":
            data = img[..., 2].astype(np.float64)
        elif ch in ("Hue", "Saturation", "Value"):
            hh, ss, vv = rgb_to_hsv(img)
            data = {"Hue": hh, "Saturation": ss * 255, "Value": vv * 255}[ch]
        else:
            data = luminance(img)

        self.ht_fig.clear()
        ax = self.ht_fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        im = ax.imshow(data, cmap=self.ht_cmap.currentText(),
                       interpolation=self.ht_interp.currentText(), aspect='auto')
        cbar = self.ht_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='#999', labelsize=8)
        cbar.outline.set_edgecolor('#555')
        ax.set_title(f"Heat Map — {ch}", color='#ddd', fontsize=10)
        ax.tick_params(colors='#999', labelsize=7)

        self.ht_fig.tight_layout()
        self.ht_cv.draw()
        self.statusBar().showMessage(f"Heat map updated ({ch})")

    # ══════════════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ══════════════════════════════════════════════════════════════════════════

    def update_statistics(self):
        if self.original_image is None:
            return
        img = self.original_image.astype(np.float64)

        def ch_stats(arr):
            m, sd = arr.mean(), arr.std()
            mn, mx = arr.min(), arr.max()
            med = np.median(arr)
            var = arr.var()
            skew = float(np.mean(((arr - m) / (sd + 1e-10)) ** 3)) if sd > 0 else 0
            kurt = float(np.mean(((arr - m) / (sd + 1e-10)) ** 4) - 3) if sd > 0 else 0
            return [f"{m:.2f}", f"{sd:.2f}", f"{mn:.0f}", f"{mx:.0f}",
                    f"{med:.1f}", f"{var:.1f}", f"{skew:.3f}", f"{kurt:.3f}"]

        row_labels = ["Mean","Std Dev","Min","Max","Median","Variance","Skewness","Kurtosis"]
        ch_data = [ch_stats(img[..., c]) for c in range(3)]
        for r in range(8):
            self.ch_tbl.setItem(r, 0, self._cen_item(row_labels[r]))
            for c in range(3):
                self.ch_tbl.setItem(r, c + 1, self._cen_item(ch_data[c][r]))

        # Luminance stats
        lum = luminance(img).ravel()
        m, sd = lum.mean(), lum.std()
        mn, mx = lum.min(), lum.max()
        med = np.median(lum)
        lum_vals = [f"{m:.2f}", f"{sd:.2f}", f"{mn:.0f}", f"{mx:.0f}",
                    f"{med:.1f}", f"{lum.var():.1f}", f"{mx-mn:.0f}",
                    f"{mx/max(mn,1):.1f}:1"]
        lum_labels = ["Mean","Std Dev","Min","Max","Median","Variance",
                      "Dynamic Range","Contrast Ratio"]
        for r in range(8):
            self.lum_tbl.setItem(r, 0, self._cen_item(lum_labels[r]))
            self.lum_tbl.setItem(r, 1, self._cen_item(lum_vals[r]))

        # Properties
        h, w = img.shape[:2]
        props = [str(w), str(h), str(self.original_image.shape[2]),
                 str(self.original_image.dtype), f"{h*w:,}",
                 f"{w/h:.3f}", f"{self.original_image.nbytes/1024:.1f}"]
        prop_labels = ["Width","Height","Channels","Dtype","Total Pixels",
                       "Aspect Ratio","Data Size (KB)"]
        for r in range(7):
            self.prop_tbl.setItem(r, 0, self._cen_item(prop_labels[r]))
            self.prop_tbl.setItem(r, 1, self._cen_item(props[r]))

        # Dominant colors
        pixels = self.original_image.reshape(-1, 3)
        centers, counts = simple_kmeans(pixels.astype(np.float64), 8)

        self.dom_fig.clear()
        ax = self.dom_fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        total = counts.sum()
        for i, (color, cnt) in enumerate(zip(centers, counts)):
            rgb_hex = "#{:02X}{:02X}{:02X}".format(*color)
            pct = cnt / total * 100
            ax.barh(i, pct, color=rgb_hex, edgecolor='#555', linewidth=0.5)
            ax.text(pct + 0.5, i, f"  {rgb_hex}  ({pct:.1f}%)",
                    va='center', color='#ddd', fontsize=9)
        ax.set_yticks(range(len(centers)))
        ax.set_yticklabels([f"C{i+1}" for i in range(len(centers))])
        ax.set_xlabel("Percentage (%)", color='#aaa', fontsize=9)
        ax.set_title("Dominant Colors", color='#ddd', fontsize=10)
        ax.tick_params(colors='#999')
        ax.invert_yaxis()
        self.dom_fig.tight_layout()
        self.dom_cv.draw()
        self.statusBar().showMessage("Statistics updated")

    # ══════════════════════════════════════════════════════════════════════════
    #  FILTERS
    # ══════════════════════════════════════════════════════════════════════════

    def apply_filter(self):
        if self.original_image is None:
            return
        img = self.original_image.astype(np.float64)
        gray = luminance(img)
        name = self.filt_combo.currentText()
        p = self.filt_param.value()

        if name == "Sobel X":
            result = np.abs(ndimage.sobel(gray, axis=1))
        elif name == "Sobel Y":
            result = np.abs(ndimage.sobel(gray, axis=0))
        elif name == "Sobel Combined":
            result = np.hypot(ndimage.sobel(gray, 1), ndimage.sobel(gray, 0))
        elif name == "Laplacian":
            result = np.abs(ndimage.laplace(gray))
        elif name == "Gaussian Blur":
            result = ndimage.gaussian_filter(gray, sigma=p)
        elif name == "Sharpen":
            blurred = ndimage.gaussian_filter(gray, sigma=p)
            result = gray + (gray - blurred) * 1.5
        elif name == "Unsharp Mask":
            blurred = ndimage.gaussian_filter(gray, sigma=p)
            result = gray + (gray - blurred) * 2.0
        elif name == "Emboss":
            kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float64)
            result = ndimage.convolve(gray, kernel)
        elif name == "Edge Enhance":
            result = np.abs(ndimage.gaussian_laplace(gray, sigma=p))
        elif name == "Prewitt":
            result = np.hypot(ndimage.prewitt(gray, 0), ndimage.prewitt(gray, 1))
        elif name == "Dilation":
            sz = max(3, int(p)*2+1)
            result = ndimage.grey_dilation(gray, size=sz)
        elif name == "Erosion":
            sz = max(3, int(p)*2+1)
            result = ndimage.grey_erosion(gray, size=sz)
        elif name == "Median Filter":
            sz = max(3, int(p)*2+1)
            result = ndimage.median_filter(gray, size=sz)
        else:
            result = gray

        # Normalize for display
        rmin, rmax = result.min(), result.max()
        display = ((result - rmin) / (rmax - rmin + 1e-10) * 255).astype(np.uint8)

        self.filt_fig.clear()

        ax1 = self.filt_fig.add_subplot(121)
        ax1.set_facecolor("#1e1e1e")
        ax1.imshow(gray.astype(np.uint8), cmap='gray', aspect='auto')
        ax1.set_title("Original (Grayscale)", color='#ddd', fontsize=10)
        ax1.tick_params(colors='#999', labelsize=7)

        ax2 = self.filt_fig.add_subplot(122)
        ax2.set_facecolor("#1e1e1e")
        ax2.imshow(display, cmap='inferno', aspect='auto')
        ax2.set_title(f"{name} (σ/size={p})", color='#ddd', fontsize=10)
        ax2.tick_params(colors='#999', labelsize=7)

        self.filt_fig.tight_layout()
        self.filt_cv.draw()
        self.statusBar().showMessage(f"Filter applied: {name}")

    # ══════════════════════════════════════════════════════════════════════════
    #  INTENSITY PROFILE
    # ══════════════════════════════════════════════════════════════════════════

    def update_profile(self):
        if self.original_image is None:
            return
        img = self.original_image
        row = min(self.prof_row.value(), img.shape[0] - 1)
        col = min(self.prof_col.value(), img.shape[1] - 1)

        self.prof_fig.clear()

        # Horizontal profile (row)
        ax1 = self.prof_fig.add_subplot(211)
        ax1.set_facecolor("#1e1e1e")
        ch_colors = ['#ff4444','#44ff44','#4488ff','#ffaa33']
        ch_labels = ['Red','Green','Blue','Luminance']
        h_line = luminance(img[row, :])
        for c in range(3):
            ax1.plot(img[row, :, c], color=ch_colors[c], linewidth=1, alpha=0.8, label=ch_labels[c])
        ax1.plot(h_line, color=ch_colors[3], linewidth=1.2, linestyle='--', label=ch_labels[3])
        ax1.axvline(col, color='#fff', linewidth=0.5, alpha=0.4)
        ax1.set_title(f"Horizontal Profile — Row {row}", color='#ddd', fontsize=10)
        ax1.legend(fontsize=7, facecolor='#333', edgecolor='#555', labelcolor='#ddd', ncol=4)
        ax1.tick_params(colors='#999', labelsize=7)
        ax1.set_xlabel("Column", color='#aaa', fontsize=8)
        ax1.set_ylabel("Intensity", color='#aaa', fontsize=8)
        ax1.set_ylim(-5, 260)

        # Vertical profile (col)
        ax2 = self.prof_fig.add_subplot(212)
        ax2.set_facecolor("#1e1e1e")
        v_line = luminance(img[:, col])
        for c in range(3):
            ax2.plot(img[:, col, c], color=ch_colors[c], linewidth=1, alpha=0.8, label=ch_labels[c])
        ax2.plot(v_line, color=ch_colors[3], linewidth=1.2, linestyle='--', label=ch_labels[3])
        ax2.axvline(row, color='#fff', linewidth=0.5, alpha=0.4)
        ax2.set_title(f"Vertical Profile — Col {col}", color='#ddd', fontsize=10)
        ax2.legend(fontsize=7, facecolor='#333', edgecolor='#555', labelcolor='#ddd', ncol=4)
        ax2.tick_params(colors='#999', labelsize=7)
        ax2.set_xlabel("Row", color='#aaa', fontsize=8)
        ax2.set_ylabel("Intensity", color='#aaa', fontsize=8)
        ax2.set_ylim(-5, 260)

        self.prof_fig.tight_layout()
        self.prof_cv.draw()
        self.statusBar().showMessage(f"Profile updated — Row {row}, Col {col}")

    # ══════════════════════════════════════════════════════════════════════════
    #  ANALYZE ALL
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_all(self):
        if self.original_image is None:
            QMessageBox.information(self, "Info", "Load an image first (Ctrl+O).")
            return
        self.statusBar().showMessage("⏳ Analyzing…")
        QApplication.processEvents()

        self.update_histogram();   QApplication.processEvents()
        self.update_eigen();       QApplication.processEvents()
        self.update_heightmap();   QApplication.processEvents()
        self.update_heatmap();     QApplication.processEvents()
        self.update_statistics();  QApplication.processEvents()
        self.apply_filter();       QApplication.processEvents()
        self.update_profile()

        self.statusBar().showMessage("✅ All analyses complete")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()