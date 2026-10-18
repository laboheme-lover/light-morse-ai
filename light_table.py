"""光のモールス信号で学習・認識する文字(クラス)と、その符号。

文字と符号の基本の表は、音のモールス(morse-ai)と同じ morse_table.py(63文字)をそのまま使う。
光では、それに括弧「（」「）」を加えた65文字を学習する(実際の文章の動画に括弧が送られていたため)。
    （  -.--.-
    ）  .-..-.
    (符号は電気通信術の音源と、光のテスト動画の両方で確かめたもの)
テスト動画のラベルの「」」(区切り)は、符号が句点「。」と同じ(.-.-..)なので「。」のクラスとして扱う。
"""

from __future__ import annotations

from morse_table import MORSE_CODE_TABLE, char_to_units

# 符号表に加える文字(括弧)
EXTRA_CODES = {"（": "-.--.-", "）": ".-..-."}
# 学習するクラス(文字 → 符号)。並びの順がモデルのクラスの番号になる
CLASS_CODES: dict[str, str] = {**MORSE_CODE_TABLE, **EXTRA_CODES}
# ラベル → 学習するクラス(符号が同じものは1つにまとめる)
LABEL_TO_CLASS = {"」": "。"}


def class_for_label(label: str) -> str | None:
    """テスト動画のラベルを、学習するクラスにする。どのクラスにも当たらなければ None。"""
    label = LABEL_TO_CLASS.get(label, label)
    return label if label in CLASS_CODES else None


def code_for_label(label: str) -> str | None:
    """ラベルのモールス符号(短点 '.'、長点 '-')。分からなければ None。"""
    cls = class_for_label(label)
    return CLASS_CODES[cls] if cls is not None else None


def units_for_char(char: str) -> list[str]:
    """1文字を送るときの符号の列(濁音は「基本仮名+濁点」の2つ、括弧は1つ)。"""
    if char in EXTRA_CODES:
        return [EXTRA_CODES[char]]
    return char_to_units(char)
