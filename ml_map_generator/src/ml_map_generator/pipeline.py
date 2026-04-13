from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

from .generator import generate_maps
from .reporting import build_variant_label, create_report_bundle
from .runner import RunnerConfig, find_start_script, run_generated_maps


def run_pipeline(
    project_root: Path,
    output_dir_name: str,
    total_maps: int,
    civilian_count: int,
    seed: int | None,
    startup_timeout_seconds: int,
    skip_run: bool,
    agent_command: str,
) -> int:
    test_config_dir = project_root / 'ml_map_generator' / 'config'
    maps_root = project_root / 'rcrs-server' / 'maps'
    rcrs_server_dir = project_root / 'rcrs-server'
    output_dir = project_root / output_dir_name

    if not test_config_dir.exists():
        raise FileNotFoundError(f'Config dir not found: {test_config_dir}')
    if not maps_root.exists():
        raise FileNotFoundError(f'Maps dir not found: {maps_root}')
    if not rcrs_server_dir.exists():
        raise FileNotFoundError(f'RCRS dir not found: {rcrs_server_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    batch = generate_maps(
        base_config_dir=test_config_dir,
        maps_root=maps_root,
        output_root=output_dir,
        total_maps=total_maps,
        civilian_count=civilian_count,
        seed=seed,
    )

    print(f'Generation seed: {batch.request_seed}')
    print(f'Available maps for run: {len(batch.all_maps)} in {output_dir}')
    print(f'Newly generated maps: {len(batch.newly_generated)}')

    run_results: list[dict] = []
    table_report_path: str | None = None
    table_session_dir: str | None = None
    session_label = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant_label = build_variant_label(project_root)

    if not skip_run:
        script_path = find_start_script(rcrs_server_dir)
        cfg = RunnerConfig(
            project_root=project_root,
            rcrs_server_dir=rcrs_server_dir,
            script_path=script_path,
            agent_command=agent_command,
            startup_timeout_seconds=startup_timeout_seconds,
            line_wait_seconds=1,
            variant_label=variant_label,
        )
        results = run_generated_maps([item.output_dir for item in batch.all_maps[:total_maps]], cfg)
        run_results = [
            {
                'map_name': item.map_name,
                'size_group': item.size_group,
                'base_map': item.base_map,
                'family': item.family,
                'finished': item.finished,
                'server_return_code': item.server_return_code,
                'agent_return_code': item.agent_return_code,
                'status_path': str(item.status_path),
            }
            for item in results
        ]
        finished_count = sum(1 for item in results if item.finished)
        print(f'Completed simulation runs: {finished_count}/{len(results)}')
        report_bundle = create_report_bundle(project_root, results, variant_label, session_label)
        table_report_path = str(report_bundle.report_xlsx_path)
        table_session_dir = str(report_bundle.session_dir)
        print(f'Benchmark table saved to: {report_bundle.report_xlsx_path}')

    summary = {
        'total_maps_requested': total_maps,
        'generation_seed': batch.request_seed,
        'maps_available': len(batch.all_maps),
        'newly_generated': len(batch.newly_generated),
        'variant_label': variant_label,
        'run_results': run_results,
        'table_report_path': table_report_path,
        'table_session_dir': table_session_dir,
        'postprocessing_skipped': True,
        'training_skipped': True,
    }
    (output_dir / 'pipeline_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return 0
