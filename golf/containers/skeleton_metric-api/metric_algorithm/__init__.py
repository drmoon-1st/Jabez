"""Metric algorithms package helper.

This module exposes the available metric submodules and provides small helpers
so external code (for example `controller.py`) can discover and import
metric implementations easily.

Usage examples:
	from metric_algorithm import list_metrics, get_metric_module
	print(list_metrics())
	m = get_metric_module('com_speed')
	# then call m.compute_com_points_3d(...) as appropriate
"""
from . import com_speed, head_speed, shoulder_sway, swing_speed, xfactor, utils_io
import os
import boto3
from pathlib import Path

__all__ = [
	'com_speed', 'head_speed', 'shoulder_sway', 'swing_speed', 'xfactor', 'csv_to_json', 'utils_io',
	'list_metrics', 'get_metric_module', 'METRIC_MODULES'
]

METRIC_MODULES = {
	'com_speed': com_speed,
	'head_speed': head_speed,
	'shoulder_sway': shoulder_sway,
	'swing_speed': swing_speed,
	'xfactor': xfactor,
}


def list_metrics():
	"""Return available metric module names."""
	return list(METRIC_MODULES.keys())


def get_metric_module(name: str):
	"""Return the metric module object for `name`, or None if not found."""
	return METRIC_MODULES.get(name)


def run_metrics_from_context(ctx: dict, dest_dir: str, job_id: str, dimension: str = '3d'):
	"""Discover tidy DataFrames (df_2d/df_3d) in ctx, convert to wide form and run selected metrics.

	This is a best-effort runner: it will try to call several metric functions and collect their
	outputs into a single JSON saved to dest_dir/<job_id>_metrics.json. It will not raise on metric
	failures; errors are captured per-metric.
	"""
	import traceback as _tb
	from pathlib import Path as _Path
	import json as _json
	import numpy as _np
	from . import utils_io as _utils

	out = {'job_id': job_id, 'dimension': dimension, 'metrics': {}}
	dest_dir = _Path(dest_dir)
	dest_dir.mkdir(parents=True, exist_ok=True)

	# find tidy DataFrames
	df2 = ctx.get('df_2d') if 'df_2d' in ctx else None
	df3 = ctx.get('df_3d') if 'df_3d' in ctx else None

	# convert tidy -> wide using utils_io helper
	wide2 = None
	wide3 = None
	try:
		if df2 is not None:
			wide2 = _utils.tidy_to_wide(df2, dimension='2d')
		if df3 is not None:
			wide3 = _utils.tidy_to_wide(df3, dimension='3d')
	except Exception:
		out['metrics']['conversion_error'] = _tb.format_exc()

	# --- Inject per-job CSVs and a minimal analyze config in dest_dir so metric
	# modules that read files or configs will use absolute paths under dest_dir.
	try:
		# ensure dest_dir is absolute
		dest_dir = dest_dir.resolve()
		# prepare csv/config paths
		overlay_csv = dest_dir / f"{job_id}_overlay.csv"
		metrics_csv = dest_dir / f"{job_id}_metrics_input.csv"
		config_path = dest_dir / f"{job_id}_analyze.yaml"

		# write wide2/wide3 to CSVs when available (use absolute paths)
		if wide2 is not None:
			try:
				wide2.to_csv(str(overlay_csv), index=False)
			except Exception:
				out['metrics']['overlay_csv_write_error'] = _tb.format_exc()
		if wide3 is not None:
			try:
				wide3.to_csv(str(metrics_csv), index=False)
			except Exception:
				out['metrics']['metrics_csv_write_error'] = _tb.format_exc()

		# For 2D runs many metric modules expect a single overlay CSV as their input.
		# Make the overlay_csv also available as `metrics_csv` for 2D so downstream
		# expansion logic can inline per-frame data even if modules don't return their own CSV paths.
		if str(dimension).lower() == '2d':
			# if metrics_csv wasn't written, point it to overlay_csv for expansion
			try:
				if not metrics_csv.exists():
					# copy overlay_csv to metrics_csv path for consistency
					if overlay_csv.exists():
						try:
							import shutil as _sh
							_sh.copy2(str(overlay_csv), str(metrics_csv))
						except Exception:
							# if copy fails, attempt to create a tiny symlink-like file by writing header
							try:
								with metrics_csv.open('w', encoding='utf-8') as mf:
									mf.write(overlay_csv.read_text(encoding='utf-8'))
							except Exception:
								pass
			except Exception:
				# non-fatal
				pass

		# build an absolute-path config that metric CLIs will accept
		cfg_obj = {
			"overlay_csv_path": str(overlay_csv),
			"metrics_csv_path": str(metrics_csv),
			"overlay_mp4": str((dest_dir / f"{job_id}_overlay.mp4").resolve()),
			"img_dir": str(dest_dir),
		}
		# try to dump as YAML if possible, otherwise write simple key: value pairs
		try:
			import yaml as _yaml
			with config_path.open('w', encoding='utf-8') as cf:
				_yaml.safe_dump(cfg_obj, cf, default_flow_style=False, allow_unicode=True)
		except Exception:
			# fallback plain text
			with config_path.open('w', encoding='utf-8') as cf:
				for k, v in cfg_obj.items():
					cf.write(f"{k}: \"{v}\"\n")

		out['injected_config'] = str(config_path)
		out['injected_overlay_mp4'] = cfg_obj['overlay_mp4']
	except Exception:
		out['config_injection_error'] = _tb.format_exc()

	# Prepare a lightweight context for metric modules
	module_ctx = {
		'wide2': wide2,
		'wide3': wide3,
		'overlay_csv': str(overlay_csv) if 'overlay_csv' in locals() else None,
		'metrics_csv': str(metrics_csv) if 'metrics_csv' in locals() else None,
		# point img_dir to dest/img where controller copies OpenPose-rendered images
		'img_dir': str((dest_dir / 'img').resolve()),
		'overlay_mp4_template': str(dest_dir / f"{job_id}_{{metric}}_overlay.mp4"),
		'job_id': job_id,
		'dest_dir': str(dest_dir),
		'dimension': dimension,
	}

	# Forward a small set of commonly-used optional context keys from the
	# original ctx so modules that expect fps/codec/draw/etc still work when
	# controller supplied them. Use None when not present so modules can fall
	# back to their defaults.
	for _k in ('fps', 'codec', 'draw', 'landmarks', 'm_per_px_2d', 'calibration_2d', 'subject'):
		# Only add the key if the original ctx explicitly provided a non-None value.
		# This preserves module-level default fallbacks like ctx.get('fps', 30).
		try:
			if _k in ctx and ctx.get(_k) is not None:
				module_ctx[_k] = ctx.get(_k)
		except Exception:
			# ignore failures; leave key absent so modules use their defaults
			pass

	# Run each metric module's standardized runner if present
	for name, mod in METRIC_MODULES.items():
		try:
			# Skip xfactor for pure 2D runs: xfactor needs 3D metrics for correct results
			if name == 'xfactor' and str(dimension).lower() == '2d':
				out['metrics'][name] = {'skipped': True, 'reason': 'xfactor requires 3D metrics; skipped for 2D run'}
				continue
			if hasattr(mod, 'run_from_context') and callable(getattr(mod, 'run_from_context')):
				try:
					res = mod.run_from_context(module_ctx)
					# If this is a 2D run, ensure modules have a metrics_csv to allow frame_data expansion.
					if isinstance(res, dict) and str(dimension).lower() == '2d':
						try:
							# if module didn't provide any csv-like keys as valid paths, attach the generated
							# metrics_csv path so the expansion step can read per-frame values. Use a safe
							# detection helper to avoid evaluating DataFrames (which cause ambiguous truth).
							def _has_valid_csv_value(v):
								if v is None:
									return False
								if isinstance(v, str):
									return len(v.strip()) > 0
								if isinstance(v, (list, tuple)):
									return any(isinstance(x, str) and len(x.strip()) > 0 for x in v)
								# other types (DataFrame, dict, etc) are not treated as valid CSV path values
								return False
							has_csv_key = any('csv' in k.lower() and _has_valid_csv_value(res.get(k)) for k in res.keys())
							# Only inject the fallback metrics_csv when the module did not error and did not
							# explicitly skip. Mark injected fallback so expansion step can ignore it.
							if not has_csv_key and not res.get('error') and not res.get('skipped') and metrics_csv.exists():
								res['metrics_csv'] = str(metrics_csv)
								res['_metrics_csv_injected'] = True
						except Exception:
							pass
					# expect res to be JSON-serializable (dict/list/primitive)
					# Normalize overlay paths: if module returned overlay_mp4, ensure file is moved into dest_dir/mp4
					if isinstance(res, dict) and res.get('overlay_mp4'):
						try:
							ov = res.get('overlay_mp4')
							ovp = _Path(ov)
							mp4_dir = dest_dir / 'mp4'
							mp4_dir.mkdir(parents=True, exist_ok=True)
							if ovp.exists():
								try:
									# prefer atomic move
									ovp.replace(mp4_dir / ovp.name)
								except Exception:
									import shutil as _sh
									try:
										_sh.move(str(ovp), str(mp4_dir / ovp.name))
									except Exception:
										try:
											_sh.copy2(str(ovp), str(mp4_dir / ovp.name))
										except Exception:
											# give up copying
											pass
							# update returned path to be relative into mp4/
							res['overlay_mp4'] = str(_Path('mp4') / (mp4_dir / ovp.name).name)
						except Exception:
							# non-fatal normalization failure
							pass
					out['metrics'][name] = res
				except Exception:
					out['metrics'][name] = {'error': _tb.format_exc()}
			else:
				# module does not provide standardized runner
				out['metrics'][name] = {'error': f"module {name} has no run_from_context"}
		except Exception:
			out['metrics'][name] = {'error': _tb.format_exc()}

	# Attempt to upload any overlay mp4s produced by metric modules to the result S3 bucket
	# NOTE: Disabled by default to avoid uploading metric overlays at top-level.
	# To enable overlay uploads from metric runner set env var METRIC_UPLOAD_OVERLAYS=1
	try:
		if os.environ.get('METRIC_UPLOAD_OVERLAYS', '0') == '1':
			bucket = os.environ.get('S3_RESULT_BUCKET_NAME') or os.environ.get('RESULT_S3_BUCKET')
			if bucket:
				s3 = boto3.client('s3')
				uploaded = []
				for name in out['metrics'].keys():
					try:
						# metric may include overlay path
						m = out['metrics'].get(name) or {}
						local_overlay = None
						if isinstance(m, dict) and 'overlay_mp4' in m:
							local_overlay = m.get('overlay_mp4')
						else:
							# try conventional name
							cand = dest_dir / f"{job_id}_{name}_overlay.mp4"
							if cand.exists():
								local_overlay = str(cand)
						if local_overlay and Path(local_overlay).exists():
							key = f"{job_id}_{name}_overlay.mp4"
							s3.upload_file(str(local_overlay), bucket, key)
							uploaded.append({'local': str(local_overlay), 'bucket': bucket, 'key': key})
							# record S3 location in out metrics
							if isinstance(m, dict):
								m['overlay_s3'] = {'bucket': bucket, 'key': key}
					except Exception:
						# non-fatal
						continue
				if uploaded:
					out['uploaded_overlays'] = uploaded
	except Exception:
		out['overlay_upload_error'] = _tb.format_exc()

	# Expand any metrics CSVs produced by modules into JSON structures so the
	# combined metrics JSON contains frame-wise data inline (helps downstream clients).
	# Convert CSVs into a single `frame_data` dict per module (frame -> {col: val}).
	try:
		import csv as _csv
		import pandas as _pd

		def _parse_val(s):
			# attempt to coerce to int/float, otherwise leave as string
			if s is None:
				return None
			s = str(s)
			if s == '':
				return None
			try:
				if '.' in s:
					f = float(s)
					return int(f) if f.is_integer() else f
			except Exception:
				pass
			try:
				return int(s)
			except Exception:
				return s

		for name, m in list(out.get('metrics', {}).items()):
			try:
				if not isinstance(m, dict):
					continue

				# Collect candidate CSV paths reported by module. Only consider values that
				# are explicit string paths (or lists of strings). Skip the runner-injected
				# fallback when present (marker '_metrics_csv_injected').
				csv_paths = []
				for key in ('metrics_csv', 'metrics_csv_path', 'metrics_csvs', 'metrics'):
					if key not in m:
						continue
					# skip fallback-injected CSVs: expansion should only process CSVs the module
					# genuinely returned, not runner fallbacks.
					if key == 'metrics_csv' and m.get('_metrics_csv_injected'):
						continue
					val = m.get(key)
					if val is None:
						continue
					if isinstance(val, (list, tuple)):
						for it in val:
							if isinstance(it, str) and it.strip():
								csv_paths.append(it)
					elif isinstance(val, str) and val.strip():
						csv_paths.append(val)

				# Normalize paths and attempt to read them into dataframes
				dfs = []
				for p in csv_paths:
					if not p:
						continue
					p_path = _Path(p)
					if not p_path.exists():
						# try relative to dest_dir
						cand = _Path(m.get('metrics_path') or _Path(dest_dir) / f"{job_id}_metric_result.json").parent / p_path.name
						if cand.exists():
							p_path = cand
						else:
							# skip missing
							continue
					try:
						df = _pd.read_csv(p_path)
						dfs.append(df)
					except Exception:
						# try csv module fallback
						try:
							with p_path.open('r', encoding='utf-8') as cf:
								rdr = _csv.DictReader(cf)
								rows = list(rdr)
							if rows:
								df = _pd.DataFrame(rows)
								dfs.append(df)
						except Exception:
							continue

				if not dfs:
					continue

				# Merge dataframes on 'frame' if present, else by index
				for i, df in enumerate(dfs):
					if 'frame' in df.columns:
						dfs[i] = df.set_index('frame')
					else:
						df.index.name = 'frame'

				from functools import reduce
				def _join(a, b):
					return a.join(b, how='outer', lsuffix='_l', rsuffix='_r')

				merged = reduce(_join, dfs) if len(dfs) > 1 else dfs[0]

				# Build frame_data: map frame index -> dict of columns (None for NaN)
				frame_data = {}
				for idx, row in merged.iterrows():
					try:
						fkey = int(idx)
					except Exception:
						fkey = str(idx)
					rd = {}
					for col, val in row.items():
						if _pd.isna(val):
							rd[col] = None
						else:
							# convert numpy scalar to python type when possible
							try:
								if hasattr(val, 'item'):
									val = val.item()
							except Exception:
								pass
							rd[col] = val
					frame_data[str(fkey)] = rd

				# attach frame_data and remove csv path keys from module result
				m['frame_data'] = frame_data
				for k in ('metrics_csv', 'metrics_csvs', 'metrics_csv_path', 'metrics'):
					if k in m:
						try:
							del m[k]
						except Exception:
							pass
			except Exception:
				m.setdefault('metrics_expand_error', _tb.format_exc())
	except Exception:
		# non-fatal; record top-level error
		out['metrics_expand_error'] = _tb.format_exc()

	# Save combined metrics JSON (one file per job)
	try:
		# remove injected_* debug/config paths before saving final JSON
		for k in list(out.keys()):
			if str(k).startswith('injected_'):
				del out[k]
		metrics_path = dest_dir / f"{job_id}_metric_result.json"
		metrics_path.write_text(_json.dumps(out, indent=2), encoding='utf-8')
		out['metrics_path'] = str(metrics_path)
	except Exception:
		out['metrics_save_error'] = _tb.format_exc()

	return out

