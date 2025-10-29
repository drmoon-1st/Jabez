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
from . import com_speed, head_speed, shoulder_sway, swing_speed, xfactor, csv_to_json, utils_io

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
	# This prevents overlays being written to unexpected relative paths.
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
				# non-fatal: record but continue
				out['metrics']['overlay_csv_write_error'] = _tb.format_exc()
		if wide3 is not None:
			try:
				wide3.to_csv(str(metrics_csv), index=False)
			except Exception:
				out['metrics']['metrics_csv_write_error'] = _tb.format_exc()

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
		# do not fail metrics runner for config injection issues
		out['config_injection_error'] = _tb.format_exc()

	# Now run metrics (best-effort)
	# 3D metrics (requires wide3)
	if dimension == '3d' and wide3 is not None and len(wide3) > 0:
		# If wide3 contains only NaNs for numeric columns, skip metrics and note it
		try:
			_numeric_cols = [c for c in wide3.columns if c != 'frame']
			if len(_numeric_cols) == 0 or wide3[_numeric_cols].isna().all().all():
				out['metrics']['note'] = 'no_valid_3d_points'
				# still write empty metrics JSON and return early
				metrics_path = dest_dir / f"{job_id}_metrics.json"
				metrics_path.write_text(_json.dumps(out, indent=2), encoding='utf-8')
				out['metrics_path'] = str(metrics_path)
				return out
		except Exception:
			# proceed to try metrics and capture warnings if any
			pass
		try:
			# COM speed
			com_pts = com_speed.compute_com_points_3d(wide3)
			com_v, unit = com_speed.speed_3d(com_pts, fps=None)
			out['metrics']['com_speed'] = {'mean': float(_np.nanmean(com_v)), 'max': float(_np.nanmax(com_v)), 'unit': unit}
		except Exception:
			out['metrics']['com_speed_error'] = _tb.format_exc()

		try:
			# Head speed (if Nose exists)
			if 'Nose' in wide3.columns.str.replace('__x','').tolist() or True:
				pts, head_speed_arr, head_dev, stability, head_unit = head_speed.compute_head_speed_3d(wide3, 'Nose', fps=None)
				out['metrics']['head_speed'] = {'mean': float(_np.nanmean(head_speed_arr)), 'max': float(_np.nanmax(head_speed_arr)), 'unit': head_unit, 'stability': stability}
		except Exception:
			out['metrics']['head_speed_error'] = _tb.format_exc()

		try:
			# Swing grip / wrist speed
			grip, R, L, R_speed, L_speed = swing_speed.compute_grip_points_3d(wide3, 'RWrist', 'LWrist')
			out['metrics']['swing'] = {'grip_mean_x': float(_np.nanmean(grip[:,0])), 'R_wrist_mean_speed': float(_np.nanmean(R_speed)), 'L_wrist_mean_speed': float(_np.nanmean(L_speed))}
		except Exception:
			out['metrics']['swing_error'] = _tb.format_exc()

		try:
			# X-factor
			# default joints
			sL, sR, hL, hR = 'LShoulder', 'RShoulder', 'LHip', 'RHip'
			xf, sa, ha = xfactor.compute_xfactor_series(wide3, sL, sR, hL, hR)
			out['metrics']['xfactor'] = {'mean': float(_np.nanmean(xf)), 'max': float(_np.nanmax(xf))}
		except Exception:
			out['metrics']['xfactor_error'] = _tb.format_exc()

	# 2D metrics (best-effort) - many modules expect 3D but some visual overlays use 2D
	if dimension == '2d' and wide2 is not None and len(wide2) > 0:
		try:
			# shoulder_sway uses overlay data and returns no numeric summary; we note availability
			out['metrics']['2d_available'] = True
		except Exception:
			out['metrics']['2d_error'] = _tb.format_exc()

	# Save metrics JSON
	try:
		metrics_path = dest_dir / f"{job_id}_metrics.json"
		metrics_path.write_text(_json.dumps(out, indent=2), encoding='utf-8')
		out['metrics_path'] = str(metrics_path)
	except Exception:
		out['metrics_save_error'] = _tb.format_exc()

	return out

