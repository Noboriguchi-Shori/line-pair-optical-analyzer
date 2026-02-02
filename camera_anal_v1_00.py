import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import signal, ndimage
from scipy.fft import fft, fftfreq
import sys
import os
import traceback

# ==========================================
# グローバル設定
# ==========================================
plt.style.use('default')
# 日本語フォント設定 (Windows向け MS Gothic)
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

class FullFeaturedLineAnalyzer:
    """
    ラインスキャンカメラ画像などの輝度プロファイルを解析するアプリケーションクラス。
    MTF, コントラスト, ピッチ, 直線性(Linearity)などを算出します。
    """

    def __init__(self, root):
        # --- ウィンドウの初期設定 ---
        self.root = root
        self.root.title("Line-Pair Optical Analyzer v1.1 Refactored")
        self.root.geometry("1650x1000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- データ管理用変数 ---
        self.df = None              # 読み込んだ生データフレーム
        self.original_data = None   # 解析対象の輝度配列(生)
        self.filtered_data = None   # フィルタ処理後の輝度配列
        self.p_list = []            # 解析結果（各ラインペアの辞書リスト）
        self.file_name = "未読み込み"
        
        # --- グラフ操作用変数 ---
        self._dragging = False
        self._prev_x = None
        
        # --- サブウィンドウ管理リスト ---
        self.trend_sub_windows = []

        # --- UI制御用変数 (Tkinter Variables) ---
        self.filter_type = tk.StringVar(value="None")
        self.p1_var = tk.DoubleVar(value=10.0)       # フィルタパラメータ(px単位)
        self.threshold_var = tk.DoubleVar(value=128.0) # 二値化しきい値
        self.show_envelope = tk.BooleanVar(value=True) # グラフ上の判定線表示
        self.show_trend_line = tk.BooleanVar(value=False) # トレンドグラフの近似曲線
        self.edge_low_per = tk.DoubleVar(value=10.0)   # エッジ立ち上がり判定 % (Low)
        self.edge_high_per = tk.DoubleVar(value=90.0)  # エッジ立ち上がり判定 % (High)
        
        # --- 評価項目定義 (全17項目) ---
        # 各項目のラベル、表示ON/OFF、グラフ色、Y軸範囲設定を管理
        self.metrics_info = {
            "pitch":      {"label": "ピッチ [px]", "var": tk.BooleanVar(value=True), "color": "#d35400", "ymin": "Auto", "ymax": "Auto"},
            "duty":       {"label": "Duty比 [%]", "var": tk.BooleanVar(value=False), "color": "#27ae60", "ymin": "0", "ymax": "100"},
            "contrast":   {"label": "コントラスト", "var": tk.BooleanVar(value=True), "color": "#c0392b", "ymin": "0", "ymax": "1"},
            "ratio":      {"label": "コントラスト比", "var": tk.BooleanVar(value=False), "color": "#8e44ad", "ymin": "Auto", "ymax": "Auto"},
            "rise_px":    {"label": "立上り幅 [px]", "var": tk.BooleanVar(value=False), "color": "#2980b9", "ymin": "Auto", "ymax": "Auto"},
            "fall_px":    {"label": "立下り幅 [px]", "var": tk.BooleanVar(value=False), "color": "#8e44ad", "ymin": "Auto", "ymax": "Auto"},
            "slope":      {"label": "エッジ傾斜度", "var": tk.BooleanVar(value=False), "color": "#00acc1", "ymin": "Auto", "ymax": "Auto"},
            "asymmetry":  {"label": "非対称性 [px]", "var": tk.BooleanVar(value=False), "color": "#5e35b1", "ymin": "Auto", "ymax": "Auto"},
            "distortion": {"label": "歪曲偏差 [%]", "var": tk.BooleanVar(value=False), "color": "#e67e22", "ymin": "-5", "ymax": "5"},
            "shading":    {"label": "明部相対輝度 [%]", "var": tk.BooleanVar(value=False), "color": "#fbc02d", "ymin": "0", "ymax": "110"},
            "dark_shade": {"label": "暗部均一性 [%]", "var": tk.BooleanVar(value=False), "color": "#455a64", "ymin": "0", "ymax": "200"},
            "overshoot":  {"label": "オーバーシュート", "var": tk.BooleanVar(value=False), "color": "#e91e63", "ymin": "0", "ymax": "20"},
            "snr":        {"label": "SNR [dB]", "var": tk.BooleanVar(value=False), "color": "#00897b", "ymin": "Auto", "ymax": "Auto"},
            "jitter":     {"label": "隣接誤差 [px]", "var": tk.BooleanVar(value=False), "color": "#546e7a", "ymin": "Auto", "ymax": "Auto"},
            "high":       {"label": "High輝度", "var": tk.BooleanVar(value=False), "color": "#fdd835", "ymin": "Auto", "ymax": "Auto"},
            "low":        {"label": "Low輝度", "var": tk.BooleanVar(value=False), "color": "#757575", "ymin": "Auto", "ymax": "Auto"},
            "linearity":  {"label": "黒位置ズレ [px]", "var": tk.BooleanVar(value=False), "color": "#000000", "ymin": "Auto", "ymax": "Auto"}
        }
        
        # 文字列型の "Auto" などを格納するためにStringVarへ変換
        for k, v in self.metrics_info.items():
            if not isinstance(v["ymin"], tk.StringVar): v["ymin"] = tk.StringVar(value=v["ymin"])
            if not isinstance(v["ymax"], tk.StringVar): v["ymax"] = tk.StringVar(value=v["ymax"])

        # UI構築とイベントバインドの実行
        self.setup_ui()
        self.setup_events()

    # ==========================================
    # UI構築メソッド群
    # ==========================================
    def setup_ui(self):
        """メイン画面のレイアウトを構築します"""
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- サイドバー (左側操作パネル) ---
        self.sidebar = tk.Frame(main_paned, padx=10, pady=10)
        main_paned.add(self.sidebar, width=320)

        # ファイル読み込みボタン
        tk.Button(self.sidebar, text="📂 CSVファイルを開く", command=self.load_file, 
                  bg="#e1f5fe", font=("Meiryo UI", 10, "bold")).pack(fill=tk.X, pady=(0, 10))

        # 1. 信号処理セクション
        self.add_header("1. 信号処理フィルタ")
        self.filter_combo = ttk.Combobox(self.sidebar, textvariable=self.filter_type, state="readonly", 
                                         values=["None", "Lowpass", "Highpass", "Moving Average", "Median", "Gaussian"])
        self.filter_combo.pack(fill=tk.X, pady=5)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot(recalc=True))
        tk.Button(self.sidebar, text="⚙ フィルタ詳細設定", command=self.open_filter_config).pack(fill=tk.X)

        # 2. 解析パラメータセクション
        self.add_header("2. 解析パラメータ")
        tk.Button(self.sidebar, text="📏 エッジ判定範囲 (Low-High)", command=self.open_edge_config, bg="#fff3e0").pack(fill=tk.X, pady=5)
        self.create_val_input(self.sidebar, self.threshold_var, 0, 255, 0.1, "二値化しきい値:")
        tk.Button(self.sidebar, text="↩ しきい値を平均値に戻す", command=self.reset_threshold, 
                  bg="#eeeeee", font=("Meiryo UI", 8)).pack(fill=tk.X, pady=(0, 5))

        # 3. 表示設定セクション
        self.add_header("3. グラフ表示オプション")
        tk.Button(self.sidebar, text="📈 メイングラフ設定", command=self.open_trend_config, bg="#e8f5e9").pack(fill=tk.X, pady=5)
        tk.Button(self.sidebar, text="📊 新規トレンド窓を追加", command=self.spawn_trend_window, 
                  bg="#d1c4e9", font=("Meiryo UI", 9, "bold")).pack(fill=tk.X, pady=5)
        
        chk_frame = tk.Frame(self.sidebar)
        chk_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(chk_frame, text="判定ガイドを表示 (プロファイル)", variable=self.show_envelope, command=self.update_plot).pack(anchor=tk.W)
        tk.Checkbutton(chk_frame, text="近似曲線を表示 (トレンド)", variable=self.show_trend_line, command=self.update_plot).pack(anchor=tk.W)

        # 4. 情報表示セクション
        self.add_header("4. ファイル・解析サマリー")
        info_frame = tk.Frame(self.sidebar)
        info_frame.pack(fill=tk.X, pady=5)
        self.info_text = tk.Text(info_frame, height=12, width=30, font=("MS Gothic", 9), relief="groove", padx=5, pady=5)
        sb = tk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=sb.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(state=tk.DISABLED)
        tk.Button(self.sidebar, text="🔄 表示リセット", command=lambda: self.update_plot(reset_view=True, recalc=True)).pack(fill=tk.X, pady=5)

        # 5. データ出力セクション
        self.add_header("5. データ出力")
        tk.Button(self.sidebar, text="💾 解析結果をCSV保存", command=self.save_csv, 
                  bg="#cfd8dc", font=("Meiryo UI", 9, "bold")).pack(fill=tk.X, pady=5)

        # --- メイングラフエリア (右側) ---
        self.right_frame = tk.Frame(main_paned)
        main_paned.add(self.right_frame)

        # Matplotlib Figure作成
        # 4行2列のグリッドレイアウトを使用
        self.fig = plt.figure(figsize=(10, 11))
        gs = self.fig.add_gridspec(4, 2, height_ratios=[1.2, 0.7, 0.7, 1.1])
        
        self.ax1 = self.fig.add_subplot(gs[0, :])     # プロファイル (上段全体)
        self.ax_diff = self.fig.add_subplot(gs[1, :], sharex=self.ax1) # 微分波形 (中段上全体)
        self.ax2 = self.fig.add_subplot(gs[2, 0])     # FFT (中段下左)
        self.ax_mtf = self.fig.add_subplot(gs[2, 1])  # MTF (中段下右)
        self.ax3 = self.fig.add_subplot(gs[3, :])     # トレンド (下段全体)
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Canvasへの埋め込み
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ツールバーとスクロールバー
        self.scrollbar = tk.Scrollbar(self.right_frame, orient=tk.HORIZONTAL, command=self.on_scrollbar)
        self.scrollbar.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)

        # --- データテーブル (最下部) ---
        tbl_fr = tk.Frame(self.right_frame, height=150)
        tbl_fr.pack(fill=tk.X, side=tk.BOTTOM)
        
        cols = ("id", "pos", "pitch", "contrast", "slope", "ovs", "ds", "lin")
        self.tree = ttk.Treeview(tbl_fr, columns=cols, show='headings', height=6)
        headers = ["#", "位置", "ピッチ", "コントラスト", "傾斜度", "OverS%", "DarkS%", "黒ズレ"]
        
        for c, h in zip(cols, headers): 
            self.tree.heading(c, text=h)
            self.tree.column(c, width=80, anchor=tk.CENTER)
            
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        sb_y = tk.Scrollbar(tbl_fr, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=sb_y.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)

    # ==========================================
    # ヘルパーメソッド (UI部品作成など)
    # ==========================================
    def add_header(self, text):
        """サイドバー用の見出しラベルを作成"""
        tk.Label(self.sidebar, text=text, font=("Meiryo UI", 10, "bold"), fg="#37474f").pack(anchor=tk.W, pady=(15, 5))

    def create_val_input(self, parent, var, from_, to_, resolution, label_text=None):
        """スライダー付き数値入力フレームを作成するユーティリティ"""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        if label_text: 
            tk.Label(frame, text=label_text).pack(side=tk.TOP, anchor=tk.W)
            
        sub = tk.Frame(frame)
        sub.pack(fill=tk.X)
        
        tk.Scale(sub, from_=from_, to=to_, resolution=resolution, orient=tk.HORIZONTAL, variable=var, 
                 command=lambda v: self.update_plot(recalc=True), showvalue=False).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        ent = tk.Entry(sub, width=8, textvariable=var)
        ent.pack(side=tk.RIGHT, padx=(5, 0))
        ent.bind("<Return>", lambda e: self.update_plot(recalc=True))
        return frame

    def parse_limit(self, val_str):
        """グラフ範囲入力用の文字列パース (数値 or None)"""
        try: 
            return float(val_str)
        except: 
            return None

    def reset_threshold(self):
        """しきい値をデータの平均値にリセット"""
        if self.original_data is not None:
            self.threshold_var.set(np.mean(self.original_data))
            self.update_plot(recalc=True)

    # ==========================================
    # ファイル操作 / データ入出力
    # ==========================================
    def load_file(self):
        """CSVファイルを読み込み、初期解析を実行"""
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p: 
            self.file_name = os.path.basename(p)
            # 1列目のデータを輝度配列として読み込む
            self.df = pd.read_csv(p, header=None)
            self.original_data = self.df.iloc[:, 0].dropna().values
            
            # しきい値の自動初期設定
            self.threshold_var.set(np.mean(self.original_data))
            
            # 全描画リセット
            self.update_plot(reset_view=True, recalc=True)

    def save_csv(self):
        """現在の解析結果(self.p_list)をCSV形式で保存"""
        if not self.p_list:
            messagebox.showwarning("警告", "保存するデータがありません。\n先にファイルを読み込んで解析を行ってください。")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="解析結果をCSV保存",
                filetypes=[("CSVファイル", "*.csv")],
                defaultextension=".csv",
                initialfile=f"result_{os.path.splitext(self.file_name)[0]}.csv"
            )

            if filename:
                # pandasを使って保存 (encoding='utf-8-sig' でExcel文字化け防止)
                df_out = pd.DataFrame(self.p_list)
                df_out.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("完了", f"保存しました:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"保存中にエラーが発生しました:\n{e}")

    # ==========================================
    # 解析ロジック (Signal Processing)
    # ==========================================
    def apply_advanced_filter(self, data):
        """
        選択されたアルゴリズムで輝度データをフィルタリングします。
        パラメータはピクセル単位(px)で解釈されます。
        """
        ft = self.filter_type.get()
        p1 = self.p1_var.get()
        
        # データが少なすぎる場合は処理しない
        if len(data) < 10: return data

        try:
            if ft == "Lowpass":
                # 指定したピクセル周期以下の細かい波を除去
                cutoff_period = max(2.1, p1)
                wn = 2.0 / cutoff_period
                b, a = signal.butter(4, wn, 'low')
                return signal.filtfilt(b, a, data)

            if ft == "Highpass":
                # 指定したピクセル周期以上のうねりを除去
                cutoff_period = max(2.1, p1)
                wn = 2.0 / cutoff_period
                b, a = signal.butter(4, wn, 'high')
                return signal.filtfilt(b, a, data)

            if ft == "Moving Average":
                # 指定幅での移動平均
                w = max(1, int(p1))
                return np.convolve(data, np.ones(w)/w, mode='same')

            if ft == "Median":
                # 指定幅でのメディアンフィルタ
                k = int(p1)
                if k % 2 == 0: k += 1 # 奇数にする
                return signal.medfilt(data, kernel_size=max(3, k))

            if ft == "Gaussian":
                # ガウシアンぼかし
                return ndimage.gaussian_filter1d(data, sigma=p1)

        except Exception as e:
            print(f"Filter Error: {e}")
        
        return data

    def calc_edge_info(self, sig, idx, vh, vl, mode):
        """
        エッジの詳細解析（10% - 90%幅など）を行います。
        線形補間によりサブピクセル精度で座標を特定します。
        
        Args:
            sig: 輝度データ配列
            idx: 概略のエッジ位置インデックス
            vh: High輝度レベル
            vl: Low輝度レベル
            mode: 'rise' または 'fall'
        Returns:
            low_idx, high_idx, low_val, high_val
        """
        diff = vh - vl
        lv = vl + diff * (self.edge_low_per.get() / 100) # 例: 10%レベル
        hv = vl + diff * (self.edge_high_per.get() / 100) # 例: 90%レベル
        
        def find_subpixel(target_val):
            # idxの前後15pxを探索
            search_range = range(max(0, idx - 15), min(len(sig) - 1, idx + 15))
            for j in search_range:
                # 立上り: 現在 < 目標 <= 次
                if mode == 'rise' and sig[j] < target_val <= sig[j+1]:
                    # 線形補間: j + (残りの高さ / 傾き)
                    return j + (target_val - sig[j]) / (sig[j+1] - sig[j] + 1e-9)
                
                # 立下り: 現在 > 目標 >= 次
                if mode == 'fall' and sig[j] > target_val >= sig[j+1]:
                    return j + (sig[j] - target_val) / (sig[j] - sig[j+1] + 1e-9)
            return idx

        return find_subpixel(lv), find_subpixel(hv), lv, hv

    def analyze_data(self):
        """
        フィルタ済みデータからラインペアを検出し、各種指標を計算します。
        結果は self.p_list に格納されます。
        """
        thresh = self.threshold_var.get()
        # 二値化 (True/False -> 1/0)
        binary = (self.filtered_data > thresh).astype(int)
        
        # 変化点を検出 (0->1:Rise, 1->0:Fall)
        risings = np.where((binary[:-1] == 0) & (binary[1:] == 1))[0]
        fallings = np.where((binary[:-1] == 1) & (binary[1:] == 0))[0]
        
        self.p_list = []
        temp_black_positions = [] # 線形性計算用

        # 各立ち上がりペア間でループ (Rise -> Fall -> Next Rise)
        for i in range(len(risings)-1):
            r1, r2 = risings[i], risings[i+1]
            
            # 間に立下りがあるか確認
            f_in = [f for f in fallings if r1 < f < r2]
            if not f_in: continue
            f1 = f_in[0]
            
            # --- 基本輝度の取得 ---
            hz = self.filtered_data[r1:f1] # 白領域
            # High/Lowレベル（領域平均）
            vh = np.mean(hz) if len(hz) > 0 else 0
            vl = np.mean(self.filtered_data[f1:r2]) if r2 > f1 else 0
            
            # --- 詳細エッジ解析 ---
            rl, rh, rv_l, rv_h = self.calc_edge_info(self.filtered_data, r1, vh, vl, 'rise')
            fh, fl, fv_h, fv_l = self.calc_edge_info(self.filtered_data, f1, vh, vl, 'fall')
            
            # --- 各種指標計算 ---
            rise_w = abs(rh - rl)
            fall_w = abs(fl - fh)
            local_max = np.max(hz) if len(hz) > 0 else vh
            overshoot = ((local_max - vh) / vh * 100) if vh != 0 else 0
            slope = (vh - vl) / rise_w if rise_w > 0 else 0
            ratio = vh / (vl + 1e-9)
            
            # --- サブピクセル位置計算 (しきい値との交点) ---
            # 白開始位置
            dr1 = (self.filtered_data[r1+1] - self.filtered_data[r1] + 1e-9)
            pos = r1 + (thresh - self.filtered_data[r1]) / dr1
            
            # 次の白開始位置
            dr2 = (self.filtered_data[r2+1] - self.filtered_data[r2] + 1e-9)
            npos = r2 + (thresh - self.filtered_data[r2]) / dr2
            
            # 黒開始位置 (白終了位置)
            df1 = (self.filtered_data[f1] - self.filtered_data[f1+1] + 1e-9)
            f_pos = f1 + (self.filtered_data[f1] - thresh) / df1
            
            # 黒ラインの中心 = (黒開始 + 次の白開始) / 2
            blk_center = (f_pos + npos) / 2.0
            temp_black_positions.append(blk_center)

            # 辞書に格納
            self.p_list.append({
                "id": i + 1,
                "pos": pos,
                "pitch": npos - pos,
                "freq": 1.0 / (npos - pos),
                "contrast": (vh - vl) / (vh + vl + 1e-9),
                "high": vh,
                "low": vl,
                "rise_px": rise_w,
                "fall_px": fall_w,
                "rv_l": rv_l,
                "rv_h": rv_h,
                "asymmetry": abs(rise_w - fall_w),
                "snr": 20 * np.log10(vh / (np.std(hz) + 1e-6)) if len(hz) > 0 else 0,
                "overshoot": overshoot,
                "slope": slope,
                "ratio": ratio,
                "blk_pos_raw": blk_center
            })

        # --- 全体統計と線形性(Linearity)計算 ---
        if self.p_list:
            ap = np.mean([p["pitch"] for p in self.p_list])
            mh = np.max([p["high"] for p in self.p_list])
            ml = np.mean([p["low"] for p in self.p_list])
            
            # 黒位置の理想直線からのズレを計算 (最小二乗法)
            if len(temp_black_positions) > 1:
                x_idxs = np.arange(len(temp_black_positions))
                y_centers = np.array(temp_black_positions)
                a, b = np.polyfit(x_idxs, y_centers, 1) # y = ax + b
                ideals = a * x_idxs + b
                residuals = y_centers - ideals
            else:
                residuals = [0] * len(self.p_list)

            # 統計値を各辞書に追加
            for i, p in enumerate(self.p_list):
                p["distortion"] = ((p["pitch"] - ap) / ap) * 100
                p["shading"] = (p["high"] / mh) * 100
                p["dark_shade"] = (p["low"] / ml * 100) if ml != 0 else 0
                p["jitter"] = abs(p["pitch"] - self.p_list[i-1]["pitch"]) if i > 0 else 0
                p["linearity"] = residuals[i]

    # ==========================================
    # 描画更新処理 (Main Plot Loop)
    # ==========================================
    def update_plot(self, reset_view=False, recalc=False, *args):
        """
        フィルタ処理 -> 解析 -> グラフ描画の一連の流れを制御します。
        
        Args:
            reset_view (bool): X軸のズームを初期状態に戻すか
            recalc (bool): 解析(analyze_data)をやり直すか
        """
        if self.original_data is None: return
        
        try:
            # 現在のズーム状態を保存
            xlim = self.ax1.get_xlim()
            
            # 必要であれば再解析
            if recalc or self.filtered_data is None:
                self.filtered_data = self.apply_advanced_filter(self.original_data)
                self.analyze_data()
                
                # 情報表示系の更新 (エラーが起きてもグラフは止まらないようにtry-except)
                try: self.update_info_summary() 
                except Exception as e: print(f"Info Update Error: {e}")
                
                try: self.update_table() 
                except Exception as e: print(f"Table Update Error: {e}")
                
                try: self.update_all_sub_windows() 
                except Exception as e: print(f"SubWin Update Error: {e}")

            x = np.arange(len(self.filtered_data))
            
            # --- 1. プロファイルグラフ (上段) ---
            self.ax1.clear()
            self.ax1.plot(x, self.original_data, 'silver', alpha=0.5, lw=1, label="元データ")
            self.ax1.plot(x, self.filtered_data, color='#1976d2', lw=1.2, label="フィルタ後")
            self.ax1.set_title("輝度プロファイル & 判定エッジ", fontsize=10, fontweight="bold")
            # しきい値線
            self.ax1.axhline(self.threshold_var.get(), color='#d32f2f', ls='--', alpha=0.6)
            self.ax1.grid(alpha=0.3)
            
            # 判定ガイド(緑/赤線)の描画
            if self.show_envelope.get() and self.p_list:
                for p in self.p_list:
                    self.ax1.hlines([p["rv_l"], p["rv_h"]], p["pos"]-5, p["pos"]+5, 
                                    colors=['#388e3c', '#d32f2f'], alpha=0.6, linestyles=':')

            # --- 2. 微分グラフ (中段上) ---
            self.ax_diff.clear()
            self.ax_diff.plot(x, np.gradient(self.filtered_data), color='#00796b', lw=1)
            self.ax_diff.set_title("輝度微分 (エッジ強度)", fontsize=9)
            self.ax_diff.grid(alpha=0.3)

            # --- 3. FFTグラフ (中段下左) ---
            self.ax2.clear()
            n_f = len(self.filtered_data)
            if n_f > 1:
                yf = fft(self.filtered_data - np.mean(self.filtered_data))
                xf = fftfreq(n_f, 1)[:n_f//2]
                self.ax2.plot(xf, 2.0/n_f * np.abs(yf[:n_f//2]), color='#7b1fa2', lw=1)
                self.ax2.set_yscale('log')
                self.ax2.set_title("空間周波数解析 (FFT)", fontsize=9)
                self.ax2.grid(alpha=0.3)
            
            # --- 4. MTFグラフ (中段下右) ---
            self.ax_mtf.clear()
            self.ax_mtf.set_title("MTF特性 (コントラスト vs 周波数)", fontsize=9)
            self.ax_mtf.grid(alpha=0.3)
            if self.p_list:
                self.ax_mtf.scatter([p["freq"] for p in self.p_list], 
                                    [p["contrast"] for p in self.p_list], 
                                    color='#d32f2f', s=10, alpha=0.6)
            
            # --- 5. トレンドグラフ (下段) ---
            self.ax3.clear()
            self.ax3.set_title("品質指標トレンド", fontsize=10, fontweight="bold")
            self.ax3.grid(alpha=0.3)
            
            if self.p_list:
                ids = [p["id"] for p in self.p_list]
                manual_ymin, manual_ymax = [] , []
                
                # 選択された指標をプロット
                for k, info in self.metrics_info.items():
                    if info["var"].get():
                        y_vals = [p.get(k, 0) for p in self.p_list]
                        self.ax3.plot(ids, y_vals, 'o-', ms=3, lw=1, label=info["label"], color=info["color"])
                        
                        # 近似曲線の描画
                        if self.show_trend_line.get() and len(ids) > 3:
                            try:
                                z = np.polyfit(ids, y_vals, 2)
                                p_fit = np.poly1d(z)
                                self.ax3.plot(ids, p_fit(ids), linestyle='--', linewidth=1.5, alpha=0.8, color=info["color"])
                            except: pass
                        
                        # 軸範囲の自動調整用
                        mn = self.parse_limit(info["ymin"].get())
                        mx = self.parse_limit(info["ymax"].get())
                        if mn is not None: manual_ymin.append(mn)
                        if mx is not None: manual_ymax.append(mx)

                self.ax3.legend(fontsize=8, loc='upper right')
                bottom = min(manual_ymin) if manual_ymin else None
                top = max(manual_ymax) if manual_ymax else None
                self.ax3.set_ylim(bottom=bottom, top=top)

            # 表示位置の復元
            self.ax1.set_xlim(xlim if not reset_view else (0, len(self.filtered_data)))
            self.update_scrollbar()
            
        except Exception as e:
            print(f"Critical Plot Error: {e}")
            traceback.print_exc()
        finally:
            self.canvas.draw_idle()

    def update_info_summary(self):
        """左サイドバーのテキストボックスにサマリーを表示"""
        if self.original_data is None: return
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        
        if not self.p_list:
            self.info_text.insert(tk.END, f"ファイル: {self.file_name}\n検出数: 0")
        else:
            cnt = len(self.p_list)
            p_avg = np.mean([p["pitch"] for p in self.p_list])
            c_avg = np.mean([p["contrast"] for p in self.p_list])
            lin_max = np.max([abs(p["linearity"]) for p in self.p_list])
            
            txt = f"■ ファイル: {self.file_name}\n"
            txt += f" データ点: {len(self.original_data):,}\n\n"
            txt += f"■ 解析サマリー (N={cnt})\n"
            txt += f" ピッチ平均: {p_avg:.2f} px\n"
            txt += f" コントラスト平均: {c_avg:.3f}\n"
            txt += f" 黒位置直線性(Max): {lin_max:.3f} px\n"
            self.info_text.insert(tk.END, txt)
        
        self.info_text.config(state=tk.DISABLED)

    def update_table(self):
        """下部のテーブルに詳細データを表示"""
        for item in self.tree.get_children(): 
            self.tree.delete(item)
            
        if not self.p_list: return
        
        for p in self.p_list:
            vals = (p["id"], 
                    f"{p['pos']:.1f}", 
                    f"{p['pitch']:.2f}", 
                    f"{p['contrast']:.3f}", 
                    f"{p['slope']:.2f}", 
                    f"{p['overshoot']:.1f}", 
                    f"{p['dark_shade']:.1f}", 
                    f"{p['linearity']:.2f}")
            self.tree.insert("", "end", values=vals)

    # ==========================================
    # イベント処理 (マウス操作など)
    # ==========================================
    def setup_events(self):
        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def on_zoom(self, event):
        """マウスホイールでのズーム処理"""
        if not event.inaxes: return
        scale = 0.8 if event.button == 'up' else 1.25
        cx = self.ax1.get_xlim()
        nw = (cx[1] - cx[0]) * scale
        rel = (cx[1] - event.xdata) / (cx[1] - cx[0] + 1e-9)
        self.ax1.set_xlim([event.xdata - nw * (1 - rel), event.xdata + nw * rel])
        self.update_scrollbar()
        self.canvas.draw_idle()

    def on_press(self, event):
        """ドラッグ開始"""
        if event.button == 1 and event.inaxes: 
            self._dragging = True
            self._prev_x = event.xdata

    def on_drag(self, event):
        """ドラッグ中のパン処理"""
        if self._dragging and event.inaxes: 
            dx = self._prev_x - event.xdata
            cx = self.ax1.get_xlim()
            self.ax1.set_xlim(cx[0] + dx, cx[1] + dx)
            self.update_scrollbar()
            self.canvas.draw_idle()

    def on_release(self, event): 
        """ドラッグ終了"""
        self._dragging = False

    def on_scrollbar(self, *args):
        """スクロールバー操作時の連動"""
        if self.original_data is None: return
        tw = len(self.original_data)
        cx = self.ax1.get_xlim()
        
        if args[0] == 'moveto':
            ns = float(args[1]) * tw
        else:
            ns = cx[0] + tw * 0.05 * int(args[1])
            
        ns = np.clip(ns, 0, tw - (cx[1] - cx[0]))
        self.ax1.set_xlim(ns, ns + (cx[1] - cx[0]))
        self.update_scrollbar()
        self.canvas.draw_idle()

    def update_scrollbar(self):
        """現在の表示範囲に合わせてスクロールバーの位置を更新"""
        tw = len(self.original_data) if self.original_data is not None else 1
        cx = self.ax1.get_xlim()
        self.scrollbar.set(max(0, cx[0] / tw), min(1, cx[1] / tw))
    
    # ==========================================
    # サブウィンドウ / 設定画面
    # ==========================================
    def open_filter_config(self):
        """フィルタの詳細設定ウィンドウを開く"""
        ft = self.filter_type.get()
        win = tk.Toplevel(self.root)
        win.title(f"Filter: {ft}")
        win.geometry("400x180")
        
        settings = {
            "None": ("設定なし", 0, 1, 1, 0),
            "Lowpass": ("カットオフ周期 [px] (これより細かい波を除去)", 2.5, 200.0, 0.5, 10.0),
            "Highpass": ("カットオフ周期 [px] (これより緩やかな波を除去)", 2.5, 500.0, 1.0, 100.0),
            "Moving Average": ("平均化ウィンドウ幅 [px]", 2, 100, 1, 5),
            "Median": ("除去ウィンドウ幅 [px]", 3, 51, 2, 5),
            "Gaussian": ("シグマ [px] (ぼかし強度)", 0.5, 50.0, 0.1, 2.0)
        }
        
        lbl, vmin, vmax, vres, dflt = settings.get(ft, settings["None"])
        
        # 範囲外ならデフォルト値に戻す
        if self.p1_var.get() < vmin or self.p1_var.get() > vmax: 
            self.p1_var.set(dflt)
        
        tk.Label(win, text=lbl, anchor=tk.W).pack(fill=tk.X, padx=20, pady=(15, 5))
        fr = tk.Frame(win)
        fr.pack(fill=tk.X, padx=20)
        
        tk.Scale(fr, from_=vmin, to=vmax, resolution=vres, orient=tk.HORIZONTAL, variable=self.p1_var, 
                 command=lambda v: self.update_plot(recalc=True)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Entry(fr, width=6, textvariable=self.p1_var).pack(side=tk.RIGHT)

    def open_edge_config(self):
        """エッジ判定レベル(10-90%など)の設定画面"""
        win = tk.Toplevel(self.root)
        win.title("Edge Params")
        win.geometry("380x250")
        self.create_val_input(win, self.edge_low_per, 0, 45, 1, "Low %")
        self.create_val_input(win, self.edge_high_per, 55, 100, 1, "High %")

    def open_trend_config(self):
        """トレンドグラフに表示する項目を選択する画面"""
        win = tk.Toplevel(self.root)
        win.title("Metrics Config")
        win.geometry("500x650")
        
        for k, info in self.metrics_info.items():
            f = tk.Frame(win)
            f.pack(fill=tk.X, padx=10, pady=2)
            tk.Checkbutton(f, text=info["label"], variable=info["var"], 
                           command=lambda: self.update_plot(recalc=True), width=22, anchor=tk.W).pack(side=tk.LEFT)
            tk.Entry(f, textvariable=info["ymin"], width=7).pack(side=tk.RIGHT)
            tk.Entry(f, textvariable=info["ymax"], width=7).pack(side=tk.RIGHT)

    def spawn_trend_window(self):
        """独立したトレンドグラフウィンドウを生成"""
        win = tk.Toplevel(self.root)
        win.title(f"トレンド詳細 {len(self.trend_sub_windows)+1}")
        win.geometry("600x400")
        
        def close_sub():
            plt.close(win.fig)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", close_sub)
        
        win.vars = {}
        ctrl_frame = tk.Frame(win, bg="#f5f5f5", padx=5, pady=5)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(ctrl_frame, text="表示項目:", bg="#f5f5f5", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        # チェックボックス用のスクロールエリア
        canvas_chk = tk.Canvas(ctrl_frame, bg="#f5f5f5", width=160)
        sb_chk = tk.Scrollbar(ctrl_frame, orient="vertical", command=canvas_chk.yview)
        chk_inner = tk.Frame(canvas_chk, bg="#f5f5f5")
        
        chk_inner.bind("<Configure>", lambda e: canvas_chk.configure(scrollregion=canvas_chk.bbox("all")))
        canvas_chk.create_window((0, 0), window=chk_inner, anchor="nw")
        canvas_chk.configure(yscrollcommand=sb_chk.set)
        
        canvas_chk.pack(side="left", fill="both", expand=True)
        sb_chk.pack(side="right", fill="y")
        
        for k, info in self.metrics_info.items():
            var = tk.BooleanVar(value=info["var"].get())
            win.vars[k] = var
            tk.Checkbutton(chk_inner, text=info["label"], variable=var, bg="#f5f5f5", anchor=tk.W, 
                           command=lambda w=win: self.refresh_sub_window(w)).pack(fill=tk.X, padx=2)
                           
        win.fig = plt.figure(figsize=(5, 4))
        win.ax = win.fig.add_subplot(111)
        win.canvas = FigureCanvasTkAgg(win.fig, master=win)
        win.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.refresh_sub_window(win)
        self.trend_sub_windows.append(win)

    def refresh_sub_window(self, win):
        """サブウィンドウのグラフを更新"""
        if not self.p_list: return
        try:
            win.ax.clear()
            ids = [p["id"] for p in self.p_list]
            has_plot = False
            manual_ymin, manual_ymax = [], []
            
            for k, info in self.metrics_info.items():
                if win.vars[k].get():
                    y_vals = [p.get(k, 0) for p in self.p_list]
                    win.ax.plot(ids, y_vals, 'o-', ms=3, label=info["label"], color=info["color"])
                    
                    if self.show_trend_line.get() and len(ids) > 3:
                        try:
                            z = np.polyfit(ids, y_vals, 2)
                            p_fit = np.poly1d(z)
                            win.ax.plot(ids, p_fit(ids), linestyle='--', linewidth=1.5, alpha=0.8, color=info["color"])
                        except: pass
                        
                    mn = self.parse_limit(info["ymin"].get())
                    mx = self.parse_limit(info["ymax"].get())
                    if mn is not None: manual_ymin.append(mn)
                    if mx is not None: manual_ymax.append(mx)
                    has_plot = True
            
            if has_plot:
                win.ax.legend(fontsize=8)
                win.ax.grid(alpha=0.3)
                win.ax.set_title("カスタムトレンド", fontname="MS Gothic")
                bottom = min(manual_ymin) if manual_ymin else None
                top = max(manual_ymax) if manual_ymax else None
                win.ax.set_ylim(bottom=bottom, top=top)
            else:
                win.ax.text(0.5, 0.5, "項目を選択してください", ha='center', fontname="MS Gothic")
                
            win.canvas.draw()
        except: pass

    def update_all_sub_windows(self):
        """全てのサブウィンドウを再描画"""
        self.trend_sub_windows = [w for w in self.trend_sub_windows if w.winfo_exists()]
        for win in self.trend_sub_windows: self.refresh_sub_window(win)

    def on_closing(self):
        """アプリ終了時の後処理"""
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullFeaturedLineAnalyzer(root)
    root.mainloop()