"""
Line-Pair Optical Analyzer - Gradio版 (Google Colab対応)
ラインスキャンカメラ画像などの輝度プロファイルを解析するアプリケーション
MTF, コントラスト, ピッチ, 直線性(Linearity)などを算出します。

【Google Colabでの使用方法】
1. 以下のセルでライブラリをインストール:
   !pip install gradio pandas numpy plotly scipy

2. このファイルをアップロードするか、以下のコードをセルにコピー

3. 最後のセルで実行:
   demo.launch(share=True)  # share=TrueでパブリックURLを生成
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, ndimage
from scipy.fft import fft, fftfreq
import tempfile
import os

# ==========================================
# 解析クラス
# ==========================================
class LineAnalyzer:
    """輝度プロファイル解析エンジン"""
    
    def __init__(self):
        self.original_data = None
        self.filtered_data = None
        self.p_list = []
        self.file_name = "未読み込み"
    
    def load_csv(self, file):
        """CSVファイルを読み込み"""
        if file is None:
            return None, "ファイルが選択されていません"
        
        try:
            # Gradio 6ではfileは直接ファイルパス（文字列）
            file_path = file if isinstance(file, str) else file.name
            df = pd.read_csv(file_path, header=None)
            self.original_data = df.iloc[:, 0].dropna().values
            self.file_name = os.path.basename(file_path)
            return np.mean(self.original_data), f"読み込み完了: {self.file_name} ({len(self.original_data):,} 点)"
        except Exception as e:
            return None, f"読み込みエラー: {e}"
    
    def apply_filter(self, data, filter_type, p1):
        """信号処理フィルタを適用"""
        if data is None or len(data) < 10:
            return data
        
        try:
            if filter_type == "Lowpass":
                cutoff_period = max(2.1, p1)
                wn = 2.0 / cutoff_period
                b, a = signal.butter(4, wn, 'low')
                return signal.filtfilt(b, a, data)
            
            elif filter_type == "Highpass":
                cutoff_period = max(2.1, p1)
                wn = 2.0 / cutoff_period
                b, a = signal.butter(4, wn, 'high')
                return signal.filtfilt(b, a, data)
            
            elif filter_type == "Moving Average":
                w = max(1, int(p1))
                return np.convolve(data, np.ones(w)/w, mode='same')
            
            elif filter_type == "Median":
                k = int(p1)
                if k % 2 == 0:
                    k += 1
                return signal.medfilt(data, kernel_size=max(3, k))
            
            elif filter_type == "Gaussian":
                return ndimage.gaussian_filter1d(data, sigma=p1)
        
        except Exception as e:
            print(f"Filter Error: {e}")
        
        return data
    
    def calc_edge_info(self, sig, idx, vh, vl, mode, edge_low_per, edge_high_per):
        """エッジの詳細解析（10% - 90%幅など）"""
        diff = vh - vl
        lv = vl + diff * (edge_low_per / 100)
        hv = vl + diff * (edge_high_per / 100)
        
        def find_subpixel(target_val):
            search_range = range(max(0, idx - 15), min(len(sig) - 1, idx + 15))
            for j in search_range:
                if mode == 'rise' and sig[j] < target_val <= sig[j+1]:
                    ratio = (target_val - sig[j]) / (sig[j+1] - sig[j] + 1e-9)
                    return j + ratio
                if mode == 'fall' and sig[j] > target_val >= sig[j+1]:
                    ratio = (sig[j] - target_val) / (sig[j] - sig[j+1] + 1e-9)
                    return j + ratio
            return idx
        
        return find_subpixel(lv), find_subpixel(hv), lv, hv
    
    def analyze(self, filter_type, filter_param, threshold, edge_low_per, edge_high_per):
        """メイン解析処理"""
        if self.original_data is None:
            return None, "データが読み込まれていません"
        
        # フィルタ適用
        if filter_type == "None":
            self.filtered_data = self.original_data.copy()
        else:
            self.filtered_data = self.apply_filter(self.original_data, filter_type, filter_param)
        
        # 二値化
        binary = (self.filtered_data > threshold).astype(int)
        
        # エッジ検出
        risings = np.where((binary[:-1] == 0) & (binary[1:] == 1))[0]
        fallings = np.where((binary[:-1] == 1) & (binary[1:] == 0))[0]
        
        self.p_list = []
        temp_black_positions = []
        
        for i in range(len(risings) - 1):
            r1, r2 = risings[i], risings[i + 1]
            f_in = [f for f in fallings if r1 < f < r2]
            if not f_in:
                continue
            f1 = f_in[0]
            
            # 基本輝度の取得
            hz = self.filtered_data[r1:f1]
            vh = np.mean(hz) if len(hz) > 0 else 0
            vl = np.mean(self.filtered_data[f1:r2]) if r2 > f1 else 0
            
            # 詳細エッジ解析
            rl, rh, rv_l, rv_h = self.calc_edge_info(self.filtered_data, r1, vh, vl, 'rise', edge_low_per, edge_high_per)
            fh, fl, fv_h, fv_l = self.calc_edge_info(self.filtered_data, f1, vh, vl, 'fall', edge_low_per, edge_high_per)
            
            # 各種指標計算
            rise_w = abs(rh - rl)
            fall_w = abs(fl - fh)
            local_max = np.max(hz) if len(hz) > 0 else vh
            overshoot = ((local_max - vh) / vh * 100) if vh != 0 else 0
            slope = (vh - vl) / rise_w if rise_w > 0 else 0
            ratio = vh / (vl + 1e-9)
            
            # サブピクセル位置計算
            dr1 = (self.filtered_data[r1 + 1] - self.filtered_data[r1] + 1e-9)
            pos = r1 + (threshold - self.filtered_data[r1]) / dr1
            
            dr2 = (self.filtered_data[r2 + 1] - self.filtered_data[r2] + 1e-9)
            npos = r2 + (threshold - self.filtered_data[r2]) / dr2
            
            df1 = (self.filtered_data[f1] - self.filtered_data[f1 + 1] + 1e-9)
            f_pos = f1 + (self.filtered_data[f1] - threshold) / df1
            
            blk_center = (f_pos + npos) / 2.0
            temp_black_positions.append(blk_center)
            
            # Duty比計算
            white_width = f_pos - pos
            pitch = npos - pos
            duty = (white_width / pitch * 100) if pitch != 0 else 50
            
            self.p_list.append({
                "id": i + 1,
                "pos": pos,
                "pitch": pitch,
                "freq": 1.0 / pitch if pitch != 0 else 0,
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
                "blk_pos_raw": blk_center,
                "duty": duty
            })
        
        # 全体統計と線形性計算
        if self.p_list:
            ap = np.mean([p["pitch"] for p in self.p_list])
            mh = np.max([p["high"] for p in self.p_list])
            ml = np.mean([p["low"] for p in self.p_list])
            
            if len(temp_black_positions) > 1:
                x_idxs = np.arange(len(temp_black_positions))
                y_centers = np.array(temp_black_positions)
                a, b = np.polyfit(x_idxs, y_centers, 1)
                ideals = a * x_idxs + b
                residuals = y_centers - ideals
            else:
                residuals = [0] * len(self.p_list)
            
            for i, p in enumerate(self.p_list):
                p["distortion"] = ((p["pitch"] - ap) / ap) * 100
                p["shading"] = (p["high"] / mh) * 100
                p["dark_shade"] = (p["low"] / ml * 100) if ml != 0 else 0
                p["jitter"] = abs(p["pitch"] - self.p_list[i - 1]["pitch"]) if i > 0 else 0
                p["linearity"] = residuals[i]
        
        return self.p_list, f"解析完了: {len(self.p_list)} ラインペア検出"
    
    def get_summary(self):
        """解析サマリーを取得"""
        if self.original_data is None:
            return "データが読み込まれていません"
        
        if not self.p_list:
            return f"ファイル: {self.file_name}\n検出数: 0"
        
        cnt = len(self.p_list)
        p_avg = np.mean([p["pitch"] for p in self.p_list])
        p_std = np.std([p["pitch"] for p in self.p_list])
        c_avg = np.mean([p["contrast"] for p in self.p_list])
        c_min = np.min([p["contrast"] for p in self.p_list])
        lin_max = np.max([abs(p["linearity"]) for p in self.p_list])
        lin_rms = np.sqrt(np.mean([p["linearity"]**2 for p in self.p_list]))
        
        txt = f"■ ファイル: {self.file_name}\n"
        txt += f"  データ点: {len(self.original_data):,}\n\n"
        txt += f"■ 解析サマリー (N={cnt})\n"
        txt += f"  ピッチ平均: {p_avg:.2f} px (σ={p_std:.3f})\n"
        txt += f"  コントラスト平均: {c_avg:.3f} (min={c_min:.3f})\n"
        txt += f"  黒位置直線性 Max: {lin_max:.3f} px\n"
        txt += f"  黒位置直線性 RMS: {lin_rms:.3f} px\n"
        
        return txt
    
    def export_csv(self):
        """解析結果をCSVとしてエクスポート"""
        if not self.p_list:
            return None
        
        df = pd.DataFrame(self.p_list)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as f:
            df.to_csv(f, index=False)
            return f.name


# グローバルなAnalyzerインスタンス
analyzer = LineAnalyzer()


# ==========================================
# Plotlyプロット関数群（ズーム・パン対応）
# ==========================================
def create_profile_plot(show_envelope=True, threshold=128):
    """輝度プロファイルグラフを生成（Plotly版）"""
    fig = go.Figure()
    
    if analyzer.original_data is None:
        fig.add_annotation(text="CSVファイルを読み込んでください", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x = np.arange(len(analyzer.filtered_data))
    
    # 元データ（薄いグレー）
    fig.add_trace(go.Scatter(
        x=x, y=analyzer.original_data,
        mode='lines', name='元データ',
        line=dict(color='silver', width=1),
        opacity=0.5
    ))
    
    # フィルタ後データ
    fig.add_trace(go.Scatter(
        x=x, y=analyzer.filtered_data,
        mode='lines', name='フィルタ後',
        line=dict(color='#1976d2', width=1.5)
    ))
    
    # しきい値線
    fig.add_hline(y=threshold, line_dash="dash", line_color="#d32f2f", 
                  annotation_text=f"しきい値: {threshold:.1f}")
    
    # 判定ガイド（エッジ位置）
    if show_envelope and analyzer.p_list:
        for p in analyzer.p_list:
            fig.add_vline(x=p["pos"], line_color="green", line_width=1, opacity=0.4)
    
    fig.update_layout(
        title="輝度プロファイル & 判定エッジ",
        xaxis_title="位置 [px]",
        yaxis_title="輝度",
        hovermode='x unified',
        dragmode='zoom',  # デフォルトでズームモード
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400
    )
    
    # ズーム・パン用のボタン追加
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{"dragmode": "zoom"}], label="🔍 ズーム", method="relayout"),
                    dict(args=[{"dragmode": "pan"}], label="✋ パン", method="relayout"),
                    dict(args=[{"xaxis.autorange": True, "yaxis.autorange": True}], 
                         label="🔄 リセット", method="relayout"),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    return fig


def create_diff_plot():
    """微分グラフを生成"""
    fig = go.Figure()
    
    if analyzer.filtered_data is None:
        fig.add_annotation(text="データなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x = np.arange(len(analyzer.filtered_data))
    diff_data = np.gradient(analyzer.filtered_data)
    
    fig.add_trace(go.Scatter(
        x=x, y=diff_data,
        mode='lines', name='微分値',
        line=dict(color='#00796b', width=1)
    ))
    
    fig.update_layout(
        title="輝度微分 (エッジ強度)",
        xaxis_title="位置 [px]",
        yaxis_title="dI/dx",
        hovermode='x unified',
        dragmode='zoom',
        height=300
    )
    
    return fig


def create_fft_plot():
    """FFTグラフを生成"""
    fig = go.Figure()
    
    if analyzer.filtered_data is None or len(analyzer.filtered_data) < 2:
        fig.add_annotation(text="データなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    n_f = len(analyzer.filtered_data)
    yf = np.abs(fft(analyzer.filtered_data - np.mean(analyzer.filtered_data)))[:n_f // 2]
    xf = fftfreq(n_f, 1)[:n_f // 2]
    
    fig.add_trace(go.Scatter(
        x=xf, y=yf,
        mode='lines', name='FFTスペクトル',
        line=dict(color='#6a1b9a', width=1)
    ))
    
    fig.update_layout(
        title="空間周波数解析 (FFT)",
        xaxis_title="空間周波数 [1/px]",
        yaxis_title="振幅",
        xaxis=dict(range=[0, 0.5]),
        yaxis_type="log",
        hovermode='x unified',
        dragmode='zoom',
        height=350
    )
    
    return fig


def create_mtf_plot():
    """MTFグラフを生成"""
    fig = go.Figure()
    
    if not analyzer.p_list:
        fig.add_annotation(text="データなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    freqs = [p["freq"] for p in analyzer.p_list]
    contrasts = [p["contrast"] for p in analyzer.p_list]
    ids = [p["id"] for p in analyzer.p_list]
    
    fig.add_trace(go.Scatter(
        x=freqs, y=contrasts,
        mode='markers', name='MTF',
        marker=dict(color='#e65100', size=8),
        text=[f"ID: {i}" for i in ids],
        hovertemplate="周波数: %{x:.5f}<br>コントラスト: %{y:.3f}<br>%{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title="MTF特性 (コントラスト vs 周波数)",
        xaxis_title="空間周波数 [1/px]",
        yaxis_title="コントラスト (MTF)",
        yaxis=dict(range=[0, 1.1]),
        hovermode='closest',
        dragmode='zoom',
        height=350
    )
    
    return fig


def create_trend_plot(metrics, show_trend_line=False):
    """トレンドグラフを生成"""
    fig = go.Figure()
    
    if not analyzer.p_list:
        fig.add_annotation(text="データなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    ids = [p["id"] for p in analyzer.p_list]
    
    # メトリクス定義（日本語）
    metrics_info = {
        "pitch": {"label": "ピッチ [px]", "color": "#d35400"},
        "duty": {"label": "Duty比 [%]", "color": "#27ae60"},
        "contrast": {"label": "コントラスト", "color": "#c0392b"},
        "ratio": {"label": "コントラスト比", "color": "#8e44ad"},
        "rise_px": {"label": "立上り幅 [px]", "color": "#2980b9"},
        "fall_px": {"label": "立下り幅 [px]", "color": "#9c27b0"},
        "slope": {"label": "エッジ傾斜度", "color": "#00acc1"},
        "asymmetry": {"label": "非対称性 [px]", "color": "#5e35b1"},
        "distortion": {"label": "歪曲偏差 [%]", "color": "#e67e22"},
        "shading": {"label": "明部相対輝度 [%]", "color": "#fbc02d"},
        "dark_shade": {"label": "暗部均一性 [%]", "color": "#455a64"},
        "overshoot": {"label": "オーバーシュート [%]", "color": "#e91e63"},
        "snr": {"label": "SNR [dB]", "color": "#00897b"},
        "jitter": {"label": "隣接誤差 [px]", "color": "#546e7a"},
        "high": {"label": "High輝度", "color": "#ff9800"},
        "low": {"label": "Low輝度", "color": "#757575"},
        "linearity": {"label": "黒位置ズレ [px]", "color": "#000000"}
    }
    
    for metric in metrics:
        if metric in metrics_info and metric in analyzer.p_list[0]:
            vals = [p[metric] for p in analyzer.p_list]
            info = metrics_info[metric]
            
            fig.add_trace(go.Scatter(
                x=ids, y=vals,
                mode='lines+markers', name=info["label"],
                line=dict(color=info["color"], width=1.5),
                marker=dict(size=5),
                hovertemplate=f"{info['label']}: %{{y:.3f}}<br>ID: %{{x}}<extra></extra>"
            ))
            
            if show_trend_line and len(ids) > 2:
                z = np.polyfit(ids, vals, 2)
                poly = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=ids, y=poly(ids),
                    mode='lines', name=f"{info['label']} (近似)",
                    line=dict(color=info["color"], width=1, dash='dash'),
                    opacity=0.5,
                    showlegend=False
                ))
    
    fig.update_layout(
        title="品質指標トレンド",
        xaxis_title="ラインペア ID",
        yaxis_title="値",
        hovermode='x unified',
        dragmode='zoom',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400
    )
    
    # ズーム・パン用ボタン
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{"dragmode": "zoom"}], label="🔍 ズーム", method="relayout"),
                    dict(args=[{"dragmode": "pan"}], label="✋ パン", method="relayout"),
                    dict(args=[{"xaxis.autorange": True, "yaxis.autorange": True}], 
                         label="🔄 リセット", method="relayout"),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    return fig


def create_linearity_plot():
    """直線性専用グラフを生成"""
    fig = go.Figure()
    
    if not analyzer.p_list:
        fig.add_annotation(text="データなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    ids = [p["id"] for p in analyzer.p_list]
    linearity = [p["linearity"] for p in analyzer.p_list]
    
    # 色分け（良/注意/警告）
    colors = ['#4caf50' if abs(v) < 0.5 else '#ff9800' if abs(v) < 1.0 else '#f44336' for v in linearity]
    
    fig.add_trace(go.Bar(
        x=ids, y=linearity,
        marker_color=colors,
        name='黒位置ズレ',
        hovertemplate="ID: %{x}<br>ズレ: %{y:.3f} px<extra></extra>"
    ))
    
    # 基準線
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", opacity=0.5,
                  annotation_text="±0.5 px", annotation_position="right")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="orange", opacity=0.5)
    
    fig.update_layout(
        title="直線性誤差 (黒ライン位置の理想直線からの偏差)",
        xaxis_title="ラインペア ID",
        yaxis_title="位置ズレ [px]",
        hovermode='x unified',
        dragmode='zoom',
        height=350
    )
    
    return fig


# ==========================================
# Gradio イベントハンドラ
# ==========================================
def on_file_upload(file):
    """ファイルアップロード時の処理"""
    if file is None:
        return 128, "ファイルが選択されていません", None, None, None, None, None, None
    
    threshold, msg = analyzer.load_csv(file)
    
    if threshold is None:
        return 128, msg, None, None, None, None, None, None
    
    # numpy型をPythonネイティブ型に変換
    threshold = float(threshold)
    
    # 初期解析
    analyzer.analyze("None", 10, threshold, 10, 90)
    
    return (
        threshold,
        msg,
        create_profile_plot(True, threshold),
        create_diff_plot(),
        create_fft_plot(),
        create_mtf_plot(),
        create_trend_plot(["pitch", "contrast"]),
        create_linearity_plot()
    )


def on_analyze(filter_type, filter_param, threshold, edge_low, edge_high, show_envelope, metrics, show_trend_line):
    """解析実行"""
    if analyzer.original_data is None:
        return "データが読み込まれていません", None, None, None, None, None, None, None
    
    result, msg = analyzer.analyze(filter_type, filter_param, threshold, edge_low, edge_high)
    summary = analyzer.get_summary()
    
    return (
        summary,
        create_profile_plot(show_envelope, threshold),
        create_diff_plot(),
        create_fft_plot(),
        create_mtf_plot(),
        create_trend_plot(metrics, show_trend_line),
        create_linearity_plot(),
        get_result_dataframe()
    )


def on_export():
    """CSVエクスポート"""
    path = analyzer.export_csv()
    if path:
        return path
    return None


def get_result_dataframe():
    """結果テーブル用DataFrameを取得"""
    if not analyzer.p_list:
        return None
    
    # 表示用に整形（日本語ヘッダー）
    display_data = []
    for p in analyzer.p_list:
        display_data.append({
            "ID": p["id"],
            "位置": f"{p['pos']:.1f}",
            "ピッチ": f"{p['pitch']:.2f}",
            "コントラスト": f"{p['contrast']:.3f}",
            "傾斜度": f"{p['slope']:.2f}",
            "オーバーシュート%": f"{p['overshoot']:.1f}",
            "暗部均一性%": f"{p['dark_shade']:.1f}",
            "黒位置ズレ": f"{p['linearity']:.3f}"
        })
    
    return pd.DataFrame(display_data)


def on_threshold_change(threshold):
    """しきい値変更時のプレビュー更新"""
    if analyzer.original_data is None:
        return None
    return create_profile_plot(True, threshold)


# ==========================================
# Gradio UI構築
# ==========================================
def create_gradio_app():
    """Gradioアプリケーションを構築"""
    
    with gr.Blocks(title="Line-Pair Optical Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📊 ラインペア光学解析ツール
        ラインスキャンカメラ画像の輝度プロファイルを解析し、MTF・コントラスト・直線性などを評価します。
        
        **💡 グラフ操作:** マウスドラッグでズーム、ダブルクリックでリセット、ボタンでズーム/パン切替
        """)
        
        with gr.Row():
            # ===== 左サイドバー =====
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📂 1. データ読み込み")
                file_input = gr.File(label="CSVファイルをアップロード", file_types=[".csv"])
                load_status = gr.Textbox(label="ステータス", value="ファイルを選択してください", interactive=False)
                
                gr.Markdown("### ⚙️ 2. 信号処理フィルタ")
                filter_type = gr.Dropdown(
                    choices=["None", "Lowpass", "Highpass", "Moving Average", "Median", "Gaussian"],
                    value="None",
                    label="フィルタ種類"
                )
                filter_param = gr.Slider(minimum=2, maximum=100, value=10, step=1, label="フィルタパラメータ [px]")
                
                gr.Markdown("### 📏 3. 解析パラメータ")
                threshold = gr.Slider(minimum=0, maximum=255, value=128, step=0.5, label="二値化しきい値")
                
                with gr.Row():
                    edge_low = gr.Number(value=10, label="Edge Low %", minimum=0, maximum=45)
                    edge_high = gr.Number(value=90, label="Edge High %", minimum=55, maximum=100)
                
                gr.Markdown("### 📈 4. 表示設定")
                show_envelope = gr.Checkbox(value=True, label="判定ガイドを表示")
                show_trend_line = gr.Checkbox(value=False, label="トレンド近似曲線を表示")
                
                metrics = gr.CheckboxGroup(
                    choices=[
                        ("ピッチ [px]", "pitch"),
                        ("Duty比 [%]", "duty"),
                        ("コントラスト", "contrast"),
                        ("コントラスト比", "ratio"),
                        ("立上り幅 [px]", "rise_px"),
                        ("立下り幅 [px]", "fall_px"),
                        ("エッジ傾斜度", "slope"),
                        ("非対称性 [px]", "asymmetry"),
                        ("歪曲偏差 [%]", "distortion"),
                        ("明部相対輝度 [%]", "shading"),
                        ("暗部均一性 [%]", "dark_shade"),
                        ("オーバーシュート [%]", "overshoot"),
                        ("SNR [dB]", "snr"),
                        ("隣接誤差 [px]", "jitter"),
                        ("High輝度", "high"),
                        ("Low輝度", "low"),
                        ("黒位置ズレ [px]", "linearity"),
                    ],
                    value=["pitch", "contrast"],
                    label="トレンドグラフ表示項目"
                )
                
                analyze_btn = gr.Button("🔄 解析実行", variant="primary")
                
                gr.Markdown("### 💾 5. データ出力")
                export_btn = gr.Button("📥 CSVエクスポート")
                export_file = gr.File(label="ダウンロード", interactive=False)
                
                gr.Markdown("### 📋 解析サマリー")
                summary_text = gr.Textbox(label="", lines=10, interactive=False)
            
            # ===== 右側グラフエリア =====
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("📊 プロファイル"):
                        profile_plot = gr.Plot(label="輝度プロファイル")
                        diff_plot = gr.Plot(label="微分波形")
                    
                    with gr.TabItem("📈 FFT / MTF"):
                        with gr.Row():
                            fft_plot = gr.Plot(label="FFTスペクトル")
                            mtf_plot = gr.Plot(label="MTF特性")
                    
                    with gr.TabItem("📉 トレンド"):
                        trend_plot = gr.Plot(label="品質指標トレンド")
                        linearity_plot = gr.Plot(label="直線性誤差")
                    
                    with gr.TabItem("📋 データテーブル"):
                        result_table = gr.Dataframe(label="解析結果一覧", interactive=False)
        
        # ===== イベント接続 =====
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[threshold, load_status, profile_plot, diff_plot, fft_plot, mtf_plot, trend_plot, linearity_plot]
        )
        
        analyze_btn.click(
            fn=on_analyze,
            inputs=[filter_type, filter_param, threshold, edge_low, edge_high, show_envelope, metrics, show_trend_line],
            outputs=[summary_text, profile_plot, diff_plot, fft_plot, mtf_plot, trend_plot, linearity_plot, result_table]
        )
        
        threshold.change(
            fn=on_threshold_change,
            inputs=[threshold],
            outputs=[profile_plot]
        )
        
        export_btn.click(
            fn=on_export,
            inputs=[],
            outputs=[export_file]
        )
        
        # 使い方ガイド
        gr.Markdown("""
        ---
        ### 📖 使い方
        1. **CSVファイルをアップロード**: 1列目に輝度データ（0-255等）が並んだCSVファイルを選択
        2. **しきい値を調整**: プロファイルグラフの赤い点線が波形と適切に交差するように調整
        3. **必要に応じてフィルタを適用**: ノイズが多い場合はLowpassなどを選択
        4. **解析実行ボタンをクリック**: 各種グラフと解析結果が更新されます
        5. **結果をエクスポート**: CSVエクスポートボタンで結果をダウンロード
        
        ### 📌 評価項目の説明
        | 項目 | 説明 | 理想値 |
        |------|------|--------|
        | **ピッチ** | 白黒ペアの幅 [px] | 全体で一定 |
        | **コントラスト** | 明暗の分離度 (0-1) | 1に近いほど良好 |
        | **黒位置ズレ** | 直線からの位置偏差 [px] | 0に近いほど良好 |
        | **傾斜度** | エッジの急峻さ | 大きいほどシャープ |
        """)
    
    return demo


# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()
