"""
Multiband Raster to Animated GIF / MP4
Standalone GUI application - no QGIS required.

Dependencies:
    pip install numpy Pillow gdal opencv-python-headless
    (ffmpeg on PATH is optional but recommended for H.264 MP4 output)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from PIL import Image
from osgeo import gdal


# ---------------------------------------------------------------------------
# Processing engine (no QGIS, no tkinter — pure logic)
# ---------------------------------------------------------------------------

class Feedback:
    """Thin shim so engine methods can log and report progress without QGIS."""
    def __init__(self, log_fn, progress_fn):
        self._log = log_fn
        self._progress = progress_fn

    def pushInfo(self, msg):
        self._log(msg)

    def reportError(self, msg):
        self._log(f'ERROR: {msg}')

    def setProgress(self, pct):
        self._progress(pct)


class RasterAnimationEngine:
    """All processing logic, ported from the QGIS Processing script."""

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self, params, feedback):
        input_path   = params['input_path']
        output_gif   = params['output_gif']
        duration     = params['duration']
        loop_count   = params['loop_count']
        scale        = params['scale']
        nodata_value = params['nodata_value']        # float or None
        nodata_colour = params['nodata_colour']      # (R, G, B)
        auto_scale   = params['auto_scale']
        value_min    = params['value_min']
        value_max    = params['value_max']
        save_png     = params['save_png']
        save_mp4     = params['save_mp4']
        mp4_fps      = params['mp4_fps']
        mp4_crf      = params['mp4_crf']
        watermark    = params['watermark_text']
        wm_font_size = params['watermark_font_size']
        wm_opacity   = params['watermark_opacity']
        rgb_mode     = params['rgb_mode']
        rgb_band_gap = params['rgb_band_gap']

        feedback.pushInfo(f'Input:  {input_path}')
        feedback.pushInfo(f'Output: {output_gif}')

        # Load frames
        if rgb_mode:
            feedback.pushInfo(f'RGB composite mode | band gap = {rgb_band_gap}')
            frames = self.load_bands_as_rgb_frames(
                input_path, rgb_band_gap, nodata_value, nodata_colour,
                auto_scale, value_min, value_max, feedback)
        else:
            frames = self.load_bands_as_images(
                input_path, None, nodata_value, nodata_colour,
                auto_scale, value_min, value_max, feedback)

        if not frames:
            feedback.reportError('No frames loaded — aborting.')
            return False

        # Scale
        if scale != 1.0:
            feedback.pushInfo(f'Scaling frames by {scale}x...')
            scaled = []
            for i, f in enumerate(frames):
                new_size = (max(1, int(f.width * scale)), max(1, int(f.height * scale)))
                scaled.append(f.resize(new_size, Image.LANCZOS))
                if i % 5 == 0:
                    feedback.setProgress(int((i / len(frames)) * 30))
            frames = scaled

        # GIF
        feedback.pushInfo(f'Saving GIF...')
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop_count,
            optimize=False,
        )
        if os.path.exists(output_gif):
            mb = os.path.getsize(output_gif) / 1024 / 1024
            feedback.pushInfo(f'GIF saved: {len(frames)} frames | {mb:.2f} MB')
        feedback.setProgress(50)

        # MP4
        if save_mp4:
            output_mp4 = output_gif.replace('.gif', '.mp4')
            self.save_frames_as_mp4(frames, output_mp4, mp4_fps, mp4_crf,
                                    watermark, wm_font_size, wm_opacity, feedback)

        # PNG frames + HTML viewer
        if save_png:
            png_dir = output_gif.replace('.gif', '_frames')
            png_files = self.save_frames_as_pngs(frames, png_dir, feedback)
            self.create_interactive_viewer(png_files, png_dir, len(frames), feedback)

        feedback.setProgress(100)
        feedback.pushInfo('Done.')
        return True

    # ------------------------------------------------------------------ #
    # Band loading                                                         #
    # ------------------------------------------------------------------ #

    def _open_raster(self, path, feedback):
        ds = gdal.Open(path)
        if ds is None:
            feedback.reportError(f'Could not open raster: {path}')
        return ds

    def _read_all_bands(self, ds, nodata_value, feedback):
        arrays, masks = [], []
        for b in range(1, ds.RasterCount + 1):
            band   = ds.GetRasterBand(b)
            nodata = band.GetNoDataValue()
            arr    = band.ReadAsArray().astype(np.float32)

            mask = np.zeros(arr.shape, dtype=bool)
            if nodata is not None:
                mask |= (arr == nodata)
            if nodata_value is not None:
                mask |= (arr == nodata_value)
            mask |= ~np.isfinite(arr)

            arrays.append(arr)
            masks.append(mask)

            valid = arr[~mask]
            if valid.size > 0:
                feedback.pushInfo(
                    f'  Band {b}: min={valid.min():.4f}  max={valid.max():.4f}')
        return arrays, masks

    def _value_range(self, arrays, masks, auto_scale, value_min, value_max, feedback):
        if auto_scale:
            all_valid = np.concatenate(
                [a[~m] for a, m in zip(arrays, masks) if (~m).any()])
            if all_valid.size > 0:
                vmin, vmax = float(all_valid.min()), float(all_valid.max())
                feedback.pushInfo(f'Auto value range: {vmin:.4f} – {vmax:.4f}')
            else:
                vmin, vmax = 0.0, 1.0
                feedback.pushInfo('WARNING: no valid data; using range 0–1')
        else:
            vmin, vmax = value_min, value_max
            feedback.pushInfo(f'Fixed value range: {vmin} – {vmax}')
        return vmin, vmax

    @staticmethod
    def _normalise(arr, mask, vmin, vmax):
        out = np.zeros_like(arr, dtype=np.float32)
        if vmax > vmin:
            out[~mask] = np.clip((arr[~mask] - vmin) / (vmax - vmin), 0, 1)
        return (out * 255).astype(np.uint8)

    def load_bands_as_images(self, raster_path, band_list, nodata_value,
                             nodata_colour, auto_scale, value_min, value_max, feedback):
        ds = self._open_raster(raster_path, feedback)
        if ds is None:
            return []

        num_bands    = ds.RasterCount
        bands_to_use = band_list if band_list else list(range(1, num_bands + 1))
        feedback.pushInfo(f'Raster: {num_bands} bands | loading {bands_to_use}')

        arrays, masks = [], []
        for b in bands_to_use:
            if b < 1 or b > num_bands:
                feedback.pushInfo(f'WARNING: band {b} out of range, skipping')
                continue
            band   = ds.GetRasterBand(b)
            nodata = band.GetNoDataValue()
            arr    = band.ReadAsArray().astype(np.float32)
            mask   = np.zeros(arr.shape, dtype=bool)
            if nodata is not None:
                mask |= (arr == nodata)
            if nodata_value is not None:
                mask |= (arr == nodata_value)
            mask |= ~np.isfinite(arr)
            arrays.append(arr)
            masks.append(mask)
            valid = arr[~mask]
            if valid.size > 0:
                feedback.pushInfo(
                    f'  Band {b}: min={valid.min():.4f}  max={valid.max():.4f}')
        ds = None

        vmin, vmax = self._value_range(arrays, masks, auto_scale,
                                       value_min, value_max, feedback)
        frames = []
        for arr, mask in zip(arrays, masks):
            ch  = self._normalise(arr, mask, vmin, vmax)
            rgb = np.dstack([ch, ch, ch])
            rgb[mask] = nodata_colour
            frames.append(Image.fromarray(rgb, mode='RGB'))
        return frames

    def load_bands_as_rgb_frames(self, raster_path, band_gap, nodata_value,
                                  nodata_colour, auto_scale, value_min, value_max,
                                  feedback):
        """
        RGB composite mode.
        Frame i:  R = band[(i)       % N]
                  G = band[(i+gap)   % N]
                  B = band[(i+2*gap) % N]
        Total frames = N (one per unique start position).
        """
        ds = self._open_raster(raster_path, feedback)
        if ds is None:
            return []

        N = ds.RasterCount
        feedback.pushInfo(f'Raster: {N} bands | RGB mode | gap={band_gap} | {N} frames')

        arrays, masks = self._read_all_bands(ds, nodata_value, feedback)
        ds = None

        vmin, vmax = self._value_range(arrays, masks, auto_scale,
                                       value_min, value_max, feedback)
        norm = [self._normalise(a, m, vmin, vmax) for a, m in zip(arrays, masks)]

        frames = []
        for i in range(N):
            ri, gi, bi = i % N, (i + band_gap) % N, (i + 2 * band_gap) % N
            feedback.pushInfo(
                f'  Frame {i+1}: R=band{ri+1}  G=band{gi+1}  B=band{bi+1}')
            combined_mask = masks[ri] | masks[gi] | masks[bi]
            rgb = np.dstack([norm[ri], norm[gi], norm[bi]])
            rgb[combined_mask] = nodata_colour
            frames.append(Image.fromarray(rgb, mode='RGB'))
        return frames

    # ------------------------------------------------------------------ #
    # MP4                                                                  #
    # ------------------------------------------------------------------ #

    def save_frames_as_mp4(self, frames, output_path, fps, crf,
                           watermark_text, font_size, opacity, feedback):
        try:
            import cv2
        except ImportError:
            feedback.reportError(
                'OpenCV (cv2) not installed.\n'
                'Run: pip install opencv-python-headless')
            return

        width, height = frames[0].size
        feedback.pushInfo(
            f'Saving MP4: {width}x{height} | {fps} fps | CRF {crf}')
        if watermark_text:
            feedback.pushInfo(
                f'Watermark: "{watermark_text}" | size {font_size} | opacity {opacity}')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            feedback.reportError(f'Cannot open VideoWriter: {output_path}')
            return

        overlay = None
        if watermark_text:
            overlay = self._build_watermark_overlay(
                watermark_text, width, height, font_size, opacity, cv2)

        for i, frame in enumerate(frames):
            bgr = np.array(frame)[:, :, ::-1].copy()
            if overlay is not None:
                bgr = self._apply_watermark(bgr, overlay)
            writer.write(bgr)
            if i % 5 == 0:
                feedback.setProgress(50 + int((i / len(frames)) * 40))

        writer.release()

        if os.path.exists(output_path):
            mb = os.path.getsize(output_path) / 1024 / 1024
            feedback.pushInfo(f'MP4 saved: {len(frames)} frames | {mb:.2f} MB')
            self._try_reencode_h264(output_path, fps, crf, feedback)
        else:
            feedback.reportError('MP4 file was not created.')

    def _build_watermark_overlay(self, text, width, height, font_size, opacity, cv2):
        overlay    = np.zeros((height, width, 4), dtype=np.uint8)
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = font_size / 30.0
        thickness  = max(1, font_size // 20)

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = max(8, font_size // 3)
        x = (width - tw) // 2
        y = height - margin

        px, py   = max(10, font_size // 2), max(6, font_size // 4)
        bg_alpha = min(200, opacity)
        overlay[max(0, y-th-py): min(height, y+baseline+py),
                max(0, x-px):    min(width,  x+tw+px)] = (0, 0, 0, bg_alpha)

        cv2.putText(overlay, text, (x, y), font, font_scale,
                    (255, 255, 255, opacity), thickness, cv2.LINE_AA)
        return overlay

    def _apply_watermark(self, bgr_frame, overlay):
        alpha      = overlay[:, :, 3:4].astype(np.float32) / 255.0
        blended    = (bgr_frame.astype(np.float32) * (1 - alpha)
                      + overlay[:, :, :3].astype(np.float32) * alpha)
        return blended.astype(np.uint8)

    def _try_reencode_h264(self, mp4_path, fps, crf, feedback):
        import subprocess
        tmp = mp4_path.replace('.mp4', '_h264.mp4')
        cmd = ['ffmpeg', '-y', '-i', mp4_path,
               '-vcodec', 'libx264', '-crf', str(crf),
               '-pix_fmt', 'yuv420p', '-r', str(fps), tmp]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.exists(tmp):
                os.replace(tmp, mp4_path)
                feedback.pushInfo('Re-encoded to H.264 via ffmpeg.')
            else:
                feedback.pushInfo('ffmpeg unavailable; keeping mp4v encoding.')
                if os.path.exists(tmp):
                    os.remove(tmp)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            feedback.pushInfo('ffmpeg not found; keeping mp4v encoding.')

    # ------------------------------------------------------------------ #
    # PNG frames                                                           #
    # ------------------------------------------------------------------ #

    def save_frames_as_pngs(self, frames, output_dir, feedback):
        os.makedirs(output_dir, exist_ok=True)
        png_files = []
        feedback.pushInfo(f'Saving PNG frames to: {output_dir}')
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f'frame_{i:04d}.png')
            frame.save(path)
            png_files.append(f'frame_{i:04d}.png')
            if i % 5 == 0:
                feedback.setProgress(50 + int((i / len(frames)) * 40))
        feedback.pushInfo(f'Saved {len(frames)} PNG frames')
        return png_files

    # ------------------------------------------------------------------ #
    # HTML viewer                                                          #
    # ------------------------------------------------------------------ #

    def create_interactive_viewer(self, png_files, output_dir, num_frames, feedback):
        html_path = os.path.join(output_dir, 'viewer.html')
        png_list  = '["' + '", "'.join(png_files) + '"]'
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Hillshade Frame Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px;
                margin: 20px auto; padding: 20px; background: #f0f0f0; }}
        .container {{ background: white; padding: 20px; border-radius: 8px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        #mainImage {{ max-width: 100%; height: auto; display: block;
                      margin: 0 auto; border: 1px solid #ddd; }}
        .controls {{ margin: 20px 0; text-align: center; }}
        .slider {{ width: 80%; margin: 10px auto; display: block; }}
        .buttons {{ margin: 15px 0; }}
        button {{ padding: 10px 20px; margin: 0 5px; font-size: 16px;
                  cursor: pointer; border: none; border-radius: 4px;
                  background: #4CAF50; color: white; }}
        button:hover {{ background: #45a049; }}
        .info {{ text-align: center; margin: 10px 0; font-size: 16px;
                 color: #333; font-weight: bold; }}
        .instructions {{ background: #e3f2fd; padding: 15px; border-radius: 4px;
                          margin: 10px 0; font-size: 14px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Hillshade Animation Viewer</h1>
    <div class="instructions">
        <strong>Controls:</strong> ← → arrow keys | Spacebar play/pause | drag slider to scrub
    </div>
    <img id="mainImage" src="{png_files[0]}" alt="Hillshade">
    <div class="controls">
        <div class="info">
            Frame: <span id="currentFrame">1</span> / {num_frames}
            &nbsp;&nbsp;&nbsp; Speed: <span id="speedDisplay">500</span> ms
        </div>
        <input type="range" class="slider" id="frameSlider" min="0" max="{num_frames-1}" value="0">
        <div class="buttons">
            <button onclick="firstFrame()">⏮ First</button>
            <button onclick="previousFrame()">◀ Prev</button>
            <button id="playButton" onclick="togglePlay()">▶ Play</button>
            <button onclick="nextFrame()">Next ▶</button>
            <button onclick="lastFrame()">Last ⏭</button>
        </div>
        <div>
            Speed: <input type="range" class="slider" id="speedSlider"
                          min="50" max="2000" value="500" step="50">
        </div>
    </div>
</div>
<script>
    const frames = {png_list};
    const total = {num_frames};
    let cur = 0, playing = false, timer = null, speed = 500;
    const img = document.getElementById('mainImage');
    const slider = document.getElementById('frameSlider');
    const btn = document.getElementById('playButton');
    function show(n) {{ cur = n; img.src = frames[cur];
        slider.value = cur;
        document.getElementById('currentFrame').textContent = cur + 1; }}
    function nextFrame() {{ show((cur + 1) % total); }}
    function previousFrame() {{ show((cur - 1 + total) % total); }}
    function firstFrame() {{ show(0); }}
    function lastFrame() {{ show(total - 1); }}
    function togglePlay() {{
        if (playing) {{ playing=false; btn.textContent='▶ Play'; clearInterval(timer); }}
        else {{ playing=true; btn.textContent='⏸ Pause'; timer=setInterval(nextFrame, speed); }}
    }}
    slider.addEventListener('input', function() {{ if(playing)togglePlay(); show(parseInt(this.value)); }});
    document.getElementById('speedSlider').addEventListener('input', function() {{
        speed = parseInt(this.value);
        document.getElementById('speedDisplay').textContent = speed;
        if (playing) {{ clearInterval(timer); timer = setInterval(nextFrame, speed); }}
    }});
    document.addEventListener('keydown', function(e) {{
        if (e.key==='ArrowRight') {{ e.preventDefault(); nextFrame(); }}
        if (e.key==='ArrowLeft')  {{ e.preventDefault(); previousFrame(); }}
        if (e.key===' ')          {{ e.preventDefault(); togglePlay(); }}
    }});
</script>
</body>
</html>'''
        with open(html_path, 'w') as fh:
            fh.write(html)
        feedback.pushInfo(f'HTML viewer: {html_path}')


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Multiband Raster → GIF / MP4')
        self.resizable(True, True)
        self.minsize(620, 700)

        self._engine   = RasterAnimationEngine()
        self._running  = False

        self._build_ui()
        self._on_rgb_toggle()       # set initial widget states
        self._on_autoscale_toggle()
        self._on_mp4_toggle()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        self._tab_input(nb)
        self._tab_output(nb)
        self._tab_animation(nb)
        self._tab_nodata(nb)
        self._tab_mp4(nb)
        self._tab_rgb(nb)

        # Log + Run
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self._log = scrolledtext.ScrolledText(
            frm, height=8, state='disabled',
            font=('Consolas', 9), wrap='word')
        self._log.pack(fill='both', expand=True)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill='x', padx=8, pady=(0, 8))

        self._progress = ttk.Progressbar(btn_row, length=400, mode='determinate')
        self._progress.pack(side='left', fill='x', expand=True, padx=(0, 8))

        self._run_btn = ttk.Button(btn_row, text='▶  Run', command=self._run)
        self._run_btn.pack(side='right')

    # --- Tab: Input -------------------------------------------------------
    def _tab_input(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Input')

        ttk.Label(f, text='Input raster (.tif / .vrt / etc.)').grid(
            row=0, column=0, sticky='w')
        self._input_path = tk.StringVar()
        ttk.Entry(f, textvariable=self._input_path, width=55).grid(
            row=1, column=0, sticky='ew', pady=(2, 0))
        ttk.Button(f, text='Browse…', command=self._browse_input).grid(
            row=1, column=1, padx=(4, 0))
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text='Scale factor (0.1 – 1.0)').grid(
            row=2, column=0, sticky='w', pady=(10, 0))
        self._scale = tk.DoubleVar(value=0.5)
        ttk.Spinbox(f, from_=0.1, to=1.0, increment=0.05,
                    textvariable=self._scale, width=8).grid(
            row=3, column=0, sticky='w')

    # --- Tab: Output ------------------------------------------------------
    def _tab_output(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Output')

        ttk.Label(f, text='Output GIF path').grid(row=0, column=0, sticky='w')
        self._output_gif = tk.StringVar()
        ttk.Entry(f, textvariable=self._output_gif, width=55).grid(
            row=1, column=0, sticky='ew', pady=(2, 0))
        ttk.Button(f, text='Browse…', command=self._browse_output).grid(
            row=1, column=1, padx=(4, 0))
        f.columnconfigure(0, weight=1)

        self._save_png = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Save individual PNG frames + HTML viewer',
                        variable=self._save_png).grid(
            row=2, column=0, sticky='w', pady=(12, 0))

    # --- Tab: Animation ---------------------------------------------------
    def _tab_animation(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Animation')

        def row(label, var, r, from_, to, inc, width=8):
            ttk.Label(f, text=label).grid(row=r*2,   column=0, sticky='w')
            ttk.Spinbox(f, from_=from_, to=to, increment=inc,
                        textvariable=var, width=width).grid(
                row=r*2+1, column=0, sticky='w', pady=(2, 8))

        self._duration   = tk.IntVar(value=500)
        self._loop_count = tk.IntVar(value=0)

        row('Frame duration (ms)',     self._duration,   0, 50,   5000, 50)
        row('Loop count (0=infinite)', self._loop_count, 1,  0, 100,    1)

    # --- Tab: NoData / Scaling --------------------------------------------
    def _tab_nodata(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='NoData / Scaling')

        # NoData value
        ttk.Label(f, text='NoData value (blank = auto-detect)').grid(
            row=0, column=0, columnspan=2, sticky='w')
        self._nodata_value = tk.StringVar(value='')
        ttk.Entry(f, textvariable=self._nodata_value, width=14).grid(
            row=1, column=0, sticky='w', pady=(2, 10))

        # NoData colour
        ttk.Label(f, text='NoData fill colour (R / G / B)').grid(
            row=2, column=0, columnspan=4, sticky='w')
        self._nd_r = tk.IntVar(value=255)
        self._nd_g = tk.IntVar(value=255)
        self._nd_b = tk.IntVar(value=255)
        for col, var, lbl in ((0, self._nd_r, 'R'), (1, self._nd_g, 'G'),
                               (2, self._nd_b, 'B')):
            ttk.Label(f, text=lbl).grid(row=3, column=col*2,   sticky='e', padx=(8, 2))
            ttk.Spinbox(f, from_=0, to=255, textvariable=var,
                        width=5).grid(row=3, column=col*2+1, sticky='w')

        # Auto-scale
        sep = ttk.Separator(f, orient='horizontal')
        sep.grid(row=4, column=0, columnspan=6, sticky='ew', pady=10)

        self._auto_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Auto-scale pixel values to 0–255',
                        variable=self._auto_scale,
                        command=self._on_autoscale_toggle).grid(
            row=5, column=0, columnspan=6, sticky='w')

        ttk.Label(f, text='Value min').grid(row=6, column=0, sticky='w', pady=(6, 0))
        self._value_min = tk.DoubleVar(value=0.0)
        self._vmin_spin = ttk.Spinbox(f, from_=-1e9, to=1e9, increment=0.1,
                                       textvariable=self._value_min, width=10)
        self._vmin_spin.grid(row=7, column=0, sticky='w')

        ttk.Label(f, text='Value max').grid(row=6, column=1, sticky='w', padx=(16, 0), pady=(6, 0))
        self._value_max = tk.DoubleVar(value=1.0)
        self._vmax_spin = ttk.Spinbox(f, from_=-1e9, to=1e9, increment=0.1,
                                       textvariable=self._value_max, width=10)
        self._vmax_spin.grid(row=7, column=1, sticky='w', padx=(16, 0))

    # --- Tab: MP4 ---------------------------------------------------------
    def _tab_mp4(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='MP4')

        self._save_mp4 = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text='Export MP4 video (requires OpenCV)',
                        variable=self._save_mp4,
                        command=self._on_mp4_toggle).grid(
            row=0, column=0, columnspan=2, sticky='w')

        ttk.Label(f, text='Frames per second').grid(
            row=1, column=0, sticky='w', pady=(10, 0))
        self._mp4_fps = tk.DoubleVar(value=2.0)
        self._mp4_fps_spin = ttk.Spinbox(
            f, from_=0.1, to=60.0, increment=0.5,
            textvariable=self._mp4_fps, width=8)
        self._mp4_fps_spin.grid(row=2, column=0, sticky='w', pady=(2, 8))

        ttk.Label(f, text='CRF quality (0=lossless, 51=worst)').grid(
            row=3, column=0, sticky='w')
        self._mp4_crf = tk.IntVar(value=18)
        self._mp4_crf_spin = ttk.Spinbox(
            f, from_=0, to=51, textvariable=self._mp4_crf, width=8)
        self._mp4_crf_spin.grid(row=4, column=0, sticky='w', pady=(2, 8))

        sep = ttk.Separator(f, orient='horizontal')
        sep.grid(row=5, column=0, columnspan=2, sticky='ew', pady=8)
        ttk.Label(f, text='Watermark (bottom centre, MP4 only)').grid(
            row=6, column=0, columnspan=2, sticky='w')

        ttk.Label(f, text='Text (blank = none)').grid(
            row=7, column=0, sticky='w', pady=(4, 0))
        self._wm_text = tk.StringVar(value='')
        self._wm_text_entry = ttk.Entry(
            f, textvariable=self._wm_text, width=36)
        self._wm_text_entry.grid(row=8, column=0, sticky='w', pady=(2, 6))

        ttk.Label(f, text='Font size (px)').grid(row=9, column=0, sticky='w')
        self._wm_size = tk.IntVar(value=24)
        self._wm_size_spin = ttk.Spinbox(
            f, from_=8, to=120, textvariable=self._wm_size, width=8)
        self._wm_size_spin.grid(row=10, column=0, sticky='w', pady=(2, 6))

        ttk.Label(f, text='Opacity (0–255)').grid(row=11, column=0, sticky='w')
        self._wm_opacity = tk.IntVar(value=180)
        self._wm_opacity_spin = ttk.Spinbox(
            f, from_=0, to=255, textvariable=self._wm_opacity, width=8)
        self._wm_opacity_spin.grid(row=12, column=0, sticky='w', pady=(2, 0))

        self._mp4_widgets = [
            self._mp4_fps_spin, self._mp4_crf_spin,
            self._wm_text_entry, self._wm_size_spin, self._wm_opacity_spin,
        ]

    # --- Tab: RGB ---------------------------------------------------------
    def _tab_rgb(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='RGB Mode')

        self._rgb_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f,
            text='RGB composite mode\n'
                 '(each frame is 3 bands composited as R, G, B)',
            variable=self._rgb_mode,
            command=self._on_rgb_toggle,
        ).grid(row=0, column=0, columnspan=2, sticky='w')

        ttk.Label(
            f,
            text=(
                '\nBand gap — spacing between R, G, B bands within each frame.\n'
                '\n'
                '  gap=3:  frame 1 = [1, 4, 7]   frame 2 = [2, 5, 8]  …\n'
                '  gap=2:  frame 1 = [1, 3, 5]   frame 2 = [2, 4, 6]  …\n'
                '\n'
                'Total frames = total number of bands.\n'
                'Bands wrap around at the end.\n'
            ),
            justify='left',
            foreground='#444',
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=(4, 8))

        ttk.Label(f, text='Band gap').grid(row=2, column=0, sticky='w')
        self._rgb_gap = tk.IntVar(value=3)
        self._rgb_gap_spin = ttk.Spinbox(
            f, from_=1, to=9999, textvariable=self._rgb_gap, width=8)
        self._rgb_gap_spin.grid(row=3, column=0, sticky='w', pady=(2, 0))

    # ------------------------------------------------------------------ #
    # Widget state helpers                                                  #
    # ------------------------------------------------------------------ #

    def _on_rgb_toggle(self):
        state = 'normal' if self._rgb_mode.get() else 'disabled'
        self._rgb_gap_spin.config(state=state)

    def _on_autoscale_toggle(self):
        state = 'disabled' if self._auto_scale.get() else 'normal'
        self._vmin_spin.config(state=state)
        self._vmax_spin.config(state=state)

    def _on_mp4_toggle(self):
        state = 'normal' if self._save_mp4.get() else 'disabled'
        for w in self._mp4_widgets:
            w.config(state=state)

    # ------------------------------------------------------------------ #
    # File dialogs                                                         #
    # ------------------------------------------------------------------ #

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title='Select input raster',
            filetypes=[('Raster files', '*.tif *.tiff *.vrt *.img *.asc *.nc'),
                       ('All files', '*.*')])
        if path:
            self._input_path.set(path)
            if not self._output_gif.get():
                base = os.path.splitext(path)[0]
                self._output_gif.set(base + '_animation.gif')

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title='Save GIF as…',
            defaultextension='.gif',
            filetypes=[('GIF files', '*.gif'), ('All files', '*.*')])
        if path:
            self._output_gif.set(path)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _log_msg(self, msg):
        def _insert():
            self._log.config(state='normal')
            self._log.insert('end', msg + '\n')
            self._log.see('end')
            self._log.config(state='disabled')
        self.after(0, _insert)

    def _set_progress(self, pct):
        self.after(0, lambda: self._progress.config(value=pct))

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #

    def _run(self):
        if self._running:
            return

        # Validate
        inp = self._input_path.get().strip()
        out = self._output_gif.get().strip()
        if not inp:
            messagebox.showerror('Missing input', 'Please select an input raster.')
            return
        if not os.path.exists(inp):
            messagebox.showerror('File not found', f'Input not found:\n{inp}')
            return
        if not out:
            messagebox.showerror('Missing output', 'Please specify an output GIF path.')
            return

        # Parse nodata
        nd_str = self._nodata_value.get().strip()
        nodata_value = float(nd_str) if nd_str else None

        params = dict(
            input_path      = inp,
            output_gif      = out,
            duration        = self._duration.get(),
            loop_count      = self._loop_count.get(),
            scale           = self._scale.get(),
            nodata_value    = nodata_value,
            nodata_colour   = (self._nd_r.get(), self._nd_g.get(), self._nd_b.get()),
            auto_scale      = self._auto_scale.get(),
            value_min       = self._value_min.get(),
            value_max       = self._value_max.get(),
            save_png        = self._save_png.get(),
            save_mp4        = self._save_mp4.get(),
            mp4_fps         = self._mp4_fps.get(),
            mp4_crf         = self._mp4_crf.get(),
            watermark_text  = self._wm_text.get().strip(),
            watermark_font_size = self._wm_size.get(),
            watermark_opacity   = self._wm_opacity.get(),
            rgb_mode        = self._rgb_mode.get(),
            rgb_band_gap    = self._rgb_gap.get(),
        )

        self._running = True
        self._run_btn.config(state='disabled', text='Running…')
        self._progress.config(value=0)

        # Clear log
        self._log.config(state='normal')
        self._log.delete('1.0', 'end')
        self._log.config(state='disabled')

        def worker():
            fb = Feedback(self._log_msg, self._set_progress)
            try:
                ok = self._engine.run(params, fb)
                if ok:
                    self.after(0, lambda: messagebox.showinfo(
                        'Complete', f'Output saved to:\n{out}'))
                else:
                    self.after(0, lambda: messagebox.showerror(
                        'Failed', 'Processing failed — see log for details.'))
            except Exception as exc:
                self._log_msg(f'EXCEPTION: {exc}')
                import traceback
                self._log_msg(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror(
                    'Error', str(exc)))
            finally:
                self._running = False
                self.after(0, lambda: self._run_btn.config(
                    state='normal', text='▶  Run'))

        threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = App()
    app.mainloop()
