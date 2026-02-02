# 📊 Line-Pair Optical Analyzer

ラインスキャンカメラ画像の輝度プロファイルを解析し、MTF・コントラスト・直線性などを評価するツールです。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ 特徴

- **17種類の評価指標**: ピッチ、コントラスト、MTF、直線性など
- **信号処理フィルタ**: Lowpass, Highpass, Moving Average, Median, Gaussian
- **インタラクティブグラフ**: ズーム・パン操作対応（Plotly）
- **2つの実行環境**: デスクトップ版（Tkinter）& Web版（Gradio/Google Colab）

## 📁 ファイル構成

| ファイル | 説明 |
|----------|------|
| `camera_anal_v1_00.py` | デスクトップ版（Tkinter GUI） |
| `camera_anal_gradio.py` | Web版（Gradio/Plotly） |
| `Line_Pair_Optical_Analyzer_Colab.ipynb` | Google Colab用ノートブック |
| `Line-Pair Optical Analyzer.md` | 詳細仕様書・取扱説明書 |

## 🚀 クイックスタート

### デスクトップ版（Windows）

```bash
pip install pandas numpy matplotlib scipy
python camera_anal_v1_00.py
```

### Web版（ローカル）

```bash
pip install gradio plotly pandas numpy scipy
python camera_anal_gradio.py
```

### Google Colab

1. `Line_Pair_Optical_Analyzer_Colab.ipynb` をColabにアップロード
2. セルを順番に実行
3. 生成されたURLにアクセス

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Line-PairOpticalAnalyzer/blob/main/Line_Pair_Optical_Analyzer_Colab.ipynb)

## 📊 入力データ形式

- **形式**: CSVファイル（ヘッダーなし）
- **構造**: 1列目に輝度データ（0-255等の数値）

```csv
120
125
180
220
...
```

## 📈 評価指標

| 指標 | 説明 | 理想値 |
|------|------|--------|
| **ピッチ (Pitch)** | 白黒ペアの幅 [px] | 全体で一定 |
| **コントラスト (Contrast)** | マイケルソン・コントラスト | 1に近いほど良好 |
| **黒位置ズレ (Linearity)** | 直線からの位置偏差 [px] | 0に近いほど良好 |
| **傾斜度 (Slope)** | エッジの急峻さ | 大きいほどシャープ |
| **Duty比** | 白部の占有率 [%] | チャートによる（通常50%） |
| **オーバーシュート** | エッジ後の跳ね上がり [%] | 小さいほど良好 |
| **SNR** | 信号対雑音比 [dB] | 高いほど良好 |

## 🖼️ スクリーンショット

### デスクトップ版
- 輝度プロファイル表示
- FFT/MTF解析
- トレンドグラフ（複数ウィンドウ対応）
- データテーブル

### Web版（Gradio）
- タブ切り替えUI
- インタラクティブグラフ（ズーム/パン）
- CSVエクスポート

## 🔧 動作環境

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8以上

### 依存ライブラリ

```
pandas
numpy
matplotlib (デスクトップ版)
plotly (Web版)
scipy
gradio (Web版)
```

## 📖 詳細ドキュメント

詳細な仕様書・取扱説明書は `Line-Pair Optical Analyzer.md` を参照してください。

## 📝 ライセンス

MIT License

## 🤝 コントリビューション

Issue・Pull Requestを歓迎します。
