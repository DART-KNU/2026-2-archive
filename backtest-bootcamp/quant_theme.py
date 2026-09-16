"""
quant_theme.py
금융데이터과학회 산출물(차트/표) 공통 스타일 모듈

목적:
    백테스트 결과물(수익률 곡선, MDD, 성과 요약표 등)을 포함해
    동아리에서 만드는 모든 차트/표가 같은 레이아웃과 색상을 쓰도록
    통일하기 위한 모듈입니다.

사용법:
    from quant_theme import apply_theme, plot_equity_curve, plot_bar, render_table

    apply_theme()  # 스크립트 맨 위에서 한 번만 호출
    fig, ax = plt.subplots()
    plot_equity_curve(ax, dates, equity_values, title="Equity Curve")
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.table import Table
import pandas as pd
import numpy as np


def _apply_date_axis(ax):
    """날짜 x축 라벨이 겹치지 않도록 자동 간격 조정 (Equity/Drawdown 차트용)."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


# ------------------------------------------------------------------
# 1. 색상 & 폰트 팔레트 (여기 값만 바꾸면 전체 톤이 바뀝니다)
# ------------------------------------------------------------------
COLOR_MAIN = "#161C3C"    # 메인 테마 - 제목, 헤더 배경 등
COLOR_TEXT = "#111111"    # 기본 글자
COLOR_EMPH1 = "#C05621"   # 강조 1 - 주황 (주로 음수/하락/매도 등)
COLOR_EMPH2 = "#C9A227"   # 강조 2 - 골드 (주로 보조 지표/벤치마크)
COLOR_EMPH3 = "#4DA3FF"   # 강조 3 - 블루 (주로 양수/상승/매수 등)

COLOR_GRID = "#E5E5E5"
COLOR_ROW_ALT = "#F4F5F8"  # 표 짝수행 배경 (메인색을 아주 옅게)

# 산출물 언어를 영어로 통일하기로 했으므로 한글 폰트 폴백은 필요 없습니다.
# Calibri가 없는 PC를 대비해 Arial → 기본 폰트 순으로만 대체되게 해뒀습니다.
FONT_FAMILY = ["Calibri", "Arial", "DejaVu Sans"]

# 다중 시리즈(예: 여러 전략 비교) 그릴 때 순서대로 쓰는 팔레트
PALETTE = [COLOR_MAIN, COLOR_EMPH3, COLOR_EMPH1, COLOR_EMPH2]


def apply_theme():
    """rcParams를 한 번에 세팅. 스크립트/노트북 맨 위에서 한 번만 호출하면 됩니다."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.edgecolor": COLOR_TEXT,
        "axes.titlecolor": COLOR_MAIN,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


# ------------------------------------------------------------------
# 2. 차트 헬퍼
# ------------------------------------------------------------------
def plot_equity_curve(ax, dates, series_dict, title="Equity Curve"):
    """
    자산 곡선(들)을 그립니다.
    series_dict: {"전략명": [값...], "벤치마크": [값...]} 형태로 여러 개 비교 가능
    """
    for i, (name, values) in enumerate(series_dict.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(dates, values, label=name, color=color, linewidth=1.8)

    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _apply_date_axis(ax)
    if len(series_dict) > 1:
        ax.legend(loc="upper left")
    ax.margins(x=0.01)
    return ax


def plot_drawdown(ax, dates, drawdown_pct, title="Drawdown"):
    """
    drawdown_pct: 0 이하 값의 시계열 (예: -0.12 = -12%)
    음수 영역을 강조색1(주황)로 채워서 위험 구간을 직관적으로 보여줍니다.
    """
    ax.fill_between(dates, drawdown_pct, 0, color=COLOR_EMPH1, alpha=0.35, linewidth=0)
    ax.plot(dates, drawdown_pct, color=COLOR_EMPH1, linewidth=1.2)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _apply_date_axis(ax)
    ax.margins(x=0.01)
    return ax


def plot_bar(ax, labels, values, title="", highlight_sign=True):
    """
    막대그래프. highlight_sign=True면 양수는 블루(EMPH3), 음수는 주황(EMPH1)으로 자동 채색.
    (월별 수익률, 종목별 기여도 등에 사용)
    """
    if highlight_sign:
        colors = [COLOR_EMPH3 if v >= 0 else COLOR_EMPH1 for v in values]
    else:
        colors = COLOR_MAIN

    ax.bar(labels, values, color=colors, width=0.7)
    ax.axhline(0, color=COLOR_TEXT, linewidth=0.8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    return ax


# ------------------------------------------------------------------
# 3. 표 헬퍼 (성과 요약표, 매매 로그 등)
# ------------------------------------------------------------------
def render_table(ax, df: pd.DataFrame, title=None, col_widths=None):
    """
    DataFrame을 matplotlib Table로 그려서 차트와 같은 톤의 이미지로 뽑습니다.
    (PPT/보고서에 이미지로 바로 붙여넣기 좋습니다)
    """
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", pad=12)

    n_rows, n_cols = df.shape
    tbl = Table(ax, bbox=[0, 0, 1, 1])

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    row_height = 1.0 / (n_rows + 1)

    # 헤더 행
    for j, col_name in enumerate(df.columns):
        cell = tbl.add_cell(
            0, j, col_widths[j], row_height,
            text=str(col_name), loc="center", facecolor=COLOR_MAIN,
        )
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    # 데이터 행 (짝수/홀수 행 배경 다르게)
    for i in range(n_rows):
        row_color = COLOR_ROW_ALT if i % 2 == 0 else "white"
        for j, col_name in enumerate(df.columns):
            value = df.iloc[i, j]
            cell = tbl.add_cell(
                i + 1, j, col_widths[j], row_height,
                text=str(value), loc="center", facecolor=row_color,
            )
            cell.get_text().set_color(COLOR_TEXT)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.add_table(tbl)
    return tbl


# ------------------------------------------------------------------
# 4. 백테스트 결과 리포트 한 장으로 묶어주는 조합 함수
# ------------------------------------------------------------------
def build_backtest_report(dates, equity, drawdown_pct, monthly_labels, monthly_returns,
                           summary_df, suptitle="Backtest Report"):
    """
    자산곡선 + 낙폭 + 월별수익률 + 성과요약표를 2x2 레이아웃 한 장으로 만듭니다.
    실전 데모에서는 이 함수 하나만 호출하면 완성된 리포트가 나오도록 설계했습니다.
    """
    apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", color=COLOR_MAIN, x=0.02, ha="left")

    plot_equity_curve(axes[0, 0], dates, {"전략": equity})
    plot_drawdown(axes[0, 1], dates, drawdown_pct)
    plot_bar(axes[1, 0], monthly_labels, monthly_returns, title="Monthly Returns")
    render_table(axes[1, 1], summary_df, title="Performance Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
