head_disp_addr': pre['head_disp'],
        'head_disp_pct': pre['head_disp_pct'],
    })
    
    ensure_dir(out_csv.parent)
    metrics.to_csv(out_csv, index=False)
    # 저장 로그 출력 생략 (요청에 따라 콘솔은 최소화)

    # 4) 비디오 오버레이 (이전 동작 유지)
    # 2D 스무딩 적용 가능
    draw_cfg = cfg.get('draw', {}) or {}
    smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'ema')
        window = int(smooth_cfg.get('window', 5))
        alpha = float(smooth_cfg.get('alpha', 0.2))
        gaussian_sigma = smooth_cfg.get('gaussian_sigma')
        hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
        oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0)
        oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007)
        oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
        df_overlay_sm = smooth_df_2d(
            df_overlay,
            prefer_2d=True,
            method=method,
            window=window,
            alpha=alpha,
            fps=fps,
            gaussian_sigma=gaussian_sigma,
            hampel_sigma=hampel_sigma,
            oneeuro_min_cutoff=oneeuro_min_cutoff,
            oneeuro_beta=oneeuro_beta,
            oneeuro_d_cutoff=oneeuro_d_cutoff,
        )
    else:
        df_overlay_sm = df_overlay

    # 오버레이에 필요한 최소 메트릭 계산(함수 시그니처 충족)
    head_pts, head_speed, head_deviations, stability_metrics, head_unit = compute_head_speed_3d(df_metrics, head_name, fps)
    overlay_head_video(img_dir, df_overlay_sm, head_pts, head_speed, head_deviations,
                       stability_metrics, head_unit, head_name, out_mp4, fps, codec)