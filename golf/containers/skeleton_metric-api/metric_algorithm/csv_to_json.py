#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → JSON 변환 유틸리티

개념
- 프론트엔드는 보통 JSON을 더 쉽게 소비합니다. 분석 단계에서 생성된 CSV들을
  그대로 전달하기보다, 각 행을 객체(record)로 갖는 JSON 배열로 변환해 주면
  클라이언트(웹/앱)에서 파싱 없이 바로 사용하기 편합니다.
- 이 스크립트는 폴더 단위로 CSV 파일들을 찾아 같은 이름의 .json 파일을 생성합니다.
- NaN/NaT 등 결측치는 JSON의 null 로 직렬화하여 프론트에서 안전하게 처리되도록 합니다.

사용 예
  # 폴더 내 모든 CSV를 JSON으로 변환 (동일 폴더에 .json 생성)
  python src/csv_to_json.py -i /path/to/summary

  # 단일 파일 변환 및 출력 폴더 지정
  python src/csv_to_json.py -i /path/to/metrics.csv -o /path/to/out_json

옵션
- --orient: JSON 구조를 선택(records|split|index|columns|values|table). 기본 records
- --pattern: 디렉터리 입력 시 매칭 패턴 (기본 *.csv)

analyze.yaml 연동
- -c/--config 로 analyze.yaml을 넘기면, 다음 키들을 통해 경로를 지정할 수 있습니다.
    json_export:
        input: "/path/to/csv_or_dir"
        output_dir: "/path/to/save_json"
        pattern: "*_metrics.csv"   # 폴더 입력 시 필터
        orient: "records"          # records|split|index|columns|values|table
        indent: 2                   # records 모드 들여쓰기
        ensure_ascii: false         # true면 ASCII로 이스케이프

        # 여러 CSV → 하나의 JSON 병합 출력(옵션)
        merge_output: "/path/to/merged.json"   # 지정 시 병합 JSON도 생성
        merge_mode: "by-file"                   # by-file | concat
        # by-file: {"<csv_stem>": [records...], ...}
        # concat:  [ {source: "<csv_stem>", ...record}, ... ]
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
try:
    import yaml
except Exception:
    yaml = None

try:
    # 프로젝트 공용 유틸 (폴더 생성 등)
    from utils_io import ensure_dir
except Exception:
    def ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)


ALLOWED_ORIENTS = {"records", "split", "index", "columns", "values", "table"}


def convert_one(csv_path: Path, out_dir: Path | None = None, orient: str = "records", indent: int = 2, ensure_ascii: bool = False) -> Path:
    if orient not in ALLOWED_ORIENTS:
        raise ValueError(f"unsupported orient: {orient}")

    df = pd.read_csv(csv_path)
    # NaN/NaT → None (JSON null)
    df = df.where(pd.notna(df), None)

    if out_dir is None:
        out_dir = csv_path.parent
    ensure_dir(out_dir)
    out_path = out_dir / (csv_path.stem + ".json")

    # orient 별 직렬화
    if orient == "records":
        payload = df.to_dict(orient="records")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent)
    else:
        # DataFrame.to_json은 NaN을 null로 직렬화함
        txt = df.to_json(orient=orient, force_ascii=ensure_ascii)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(txt)

    return out_path


def iter_csvs_in_dir(d: Path, pattern: str = "*.csv") -> Iterable[Path]:
    yield from sorted(d.glob(pattern))


def read_as_records(csv_path: Path) -> list[dict]:
    """CSV를 records(list[dict])로 로드(NaN→None). 병합용.
    """
    df = pd.read_csv(csv_path)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser(description="CSV → JSON 변환기 (폴더/단일 파일 지원)")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"), help="analyze.yaml 경로")
    ap.add_argument("-i", "--input", default=None, help="CSV 파일 경로 또는 CSV들이 있는 폴더 경로 (미지정 시 analyze.yaml의 json_export.input 사용)")
    ap.add_argument("-o", "--output-dir", default=None, help="출력 JSON을 저장할 폴더 (미지정 시 입력과 동일 폴더 또는 analyze.yaml의 json_export.output_dir)")
    ap.add_argument("--orient", default=None, choices=sorted(ALLOWED_ORIENTS), help="JSON 구조 유형 (기본 records 또는 analyze.yaml의 json_export.orient)")
    ap.add_argument("--pattern", default=None, help="디렉터리 입력 시 CSV 매칭 패턴 (기본 *.csv 또는 analyze.yaml의 json_export.pattern)")
    ap.add_argument("--indent", type=int, default=None, help="JSON 들여쓰기 (records 모드에 적용). 기본 2 또는 analyze.yaml의 json_export.indent")
    ap.add_argument("--ensure-ascii", action="store_true", help="ASCII만 사용하여 이스케이프(기본: 한글 유지). analyze.yaml에서 true 설정 가능")
    # 병합 출력 옵션
    ap.add_argument("-m", "--merge-output", default=None, help="여러 CSV를 하나의 JSON으로 병합해 저장할 경로")
    ap.add_argument("--merge-mode", default=None, choices=["by-file","concat"], help="병합 방식: by-file|concat (기본 by-file)")
    args = ap.parse_args()

    # analyze.yaml 로드 (있을 때만)
    cfg = None
    cfg_path = Path(args.config) if args.config else None
    if cfg_path and cfg_path.exists() and cfg_path.suffix.lower() in (".yml", ".yaml") and yaml is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = None
    jexp = (cfg.get("json_export") if isinstance(cfg, dict) else None) or {}

    # 우선순위: CLI > analyze.yaml > 기본값
    in_val = args.input or jexp.get("input")
    out_val = args.output_dir or jexp.get("output_dir")
    orient = args.orient or jexp.get("orient") or "records"
    pattern = args.pattern or jexp.get("pattern") or "*.csv"
    indent = (args.indent if args.indent is not None else jexp.get("indent", 2))
    ensure_ascii = bool(args.ensure_ascii or jexp.get("ensure_ascii", False))
    merge_output = args.merge_output or jexp.get("merge_output")
    merge_mode = args.merge_mode or jexp.get("merge_mode", "by-file")

    if not in_val:
        raise FileNotFoundError("입력 경로가 지정되지 않았습니다. -i 또는 analyze.yaml의 json_export.input을 설정하세요.")

    in_path = Path(in_val)
    out_dir = Path(out_val) if out_val else None

    print(f"📥 입력: {in_path}")
    if out_dir:
        print(f"📤 출력 폴더: {out_dir}")
    print(f"⚙️ 옵션: orient={orient}, pattern={pattern}, indent={indent}, ensure_ascii={ensure_ascii}")
    if merge_output:
        print(f"🔗 병합 출력: {merge_output} (mode={merge_mode})")

    generated = []
    if in_path.is_file():
        p = convert_one(in_path, out_dir, orient=orient, indent=indent, ensure_ascii=ensure_ascii)
        generated.append(p)
    elif in_path.is_dir():
        for csvf in iter_csvs_in_dir(in_path, pattern=pattern):
            p = convert_one(csvf, out_dir or in_path, orient=orient, indent=indent, ensure_ascii=ensure_ascii)
            generated.append(p)
    else:
        raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {in_path}")

    # 병합 JSON 생성(옵션): 폴더 입력일 때 주로 사용, 파일 입력도 단일 병합으로 저장 가능
    if merge_output:
        merge_out_path = Path(merge_output)
        merge_out_path.parent.mkdir(parents=True, exist_ok=True)
        # 대상 CSV 목록
        targets: list[Path]
        if in_path.is_file():
            targets = [in_path]
        else:
            targets = list(iter_csvs_in_dir(in_path, pattern=pattern))

        if merge_mode == "concat":
            merged: list[dict] = []
            for csvp in targets:
                recs = read_as_records(csvp)
                stem = csvp.stem
                for r in recs:
                    if isinstance(r, dict) and "source" not in r:
                        r["source"] = stem
                merged.extend(recs)
            with merge_out_path.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=ensure_ascii, indent=indent)
        else:  # by-file
            merged: dict[str, list[dict]] = {}
            for csvp in targets:
                recs = read_as_records(csvp)
                merged[csvp.stem] = recs
            with merge_out_path.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=ensure_ascii, indent=indent)
        print(f"[SAVE] 병합 JSON: {merge_out_path}")

    if generated:
        print("\n[SAVE] JSON 생성 완료:")
        for p in generated:
            print(f"  - {p}")
    else:
        print("변환할 CSV가 없습니다.")


if __name__ == "__main__":
    main()
