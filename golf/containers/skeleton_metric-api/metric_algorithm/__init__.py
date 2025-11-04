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
		'img_dir': str(dest_dir),
		'overlay_mp4_template': str(dest_dir / f"{job_id}_{{metric}}_overlay.mp4"),
		'job_id': job_id,
		'dest_dir': str(dest_dir),
		'dimension': dimension,
	}

	# Run each metric module's standardized runner if present
	for name, mod in METRIC_MODULES.items():
		try:
			if hasattr(mod, 'run_from_context') and callable(getattr(mod, 'run_from_context')):
				try:
					res = mod.run_from_context(module_ctx)
					# expect res to be JSON-serializable (dict/list/primitive)
					out['metrics'][name] = res
				except Exception:
					out['metrics'][name] = {'error': _tb.format_exc()}
			else:
				# module does not provide standardized runner
				out['metrics'][name] = {'error': f"module {name} has no run_from_context"}
		except Exception:
			out['metrics'][name] = {'error': _tb.format_exc()}

	# Attempt to upload any overlay mp4s produced by metric modules to the result S3 bucket
	try:
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

	# Save combined metrics JSON (one file per job)
	try:
		metrics_path = dest_dir / f"{job_id}_metric_result.json"
		metrics_path.write_text(_json.dumps(out, indent=2), encoding='utf-8')
		out['metrics_path'] = str(metrics_path)
	except Exception:
		out['metrics_save_error'] = _tb.format_exc()

	return out

