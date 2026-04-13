from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import time


TICK_RE = re.compile(r'"tick"\s*:\s*(\d+)|[Тт]ик[=:](\d+)')
LOG_SUFFIXES = {'.jsonl', '.log', '.txt', '.out', '.err', '.json'}


@dataclass(frozen=True)
class RunnerConfig:
    project_root: Path
    rcrs_server_dir: Path
    script_path: Path
    agent_command: str
    startup_timeout_seconds: int
    line_wait_seconds: int
    server_timeout_seconds: int = 3600
    stall_timeout_seconds: int = 90
    settle_after_finish_seconds: int = 3
    variant_label: str = 'BaseRescueAgent'


@dataclass(frozen=True)
class RunResult:
    map_name: str
    size_group: str
    base_map: str
    family: str
    generated_dir: Path
    run_dir: Path
    status_path: Path
    server_stdout_path: Path
    agent_stdout_path: Path
    agent_run_id: str
    raw_log_paths: list[Path]
    debug_log_paths: list[Path]
    text_log_paths: list[Path]
    server_return_code: int | None
    agent_return_code: int | None
    finished: bool
    failure_reason: str | None
    max_tick_seen: int
    expected_timesteps: int | None
    wall_runtime_sec: float


def find_start_script(rcrs_server_dir: Path) -> Path:
    candidates = [rcrs_server_dir / 'start-comprun.sh', rcrs_server_dir / 'scripts' / 'start-comprun.sh', rcrs_server_dir / 'boot' / 'start-comprun.sh']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = list(rcrs_server_dir.rglob('start-comprun.sh'))
    if found:
        return found[0]
    raise FileNotFoundError(f'start-comprun.sh not found under {rcrs_server_dir}')


def find_kill_script(rcrs_server_dir: Path) -> Path:
    candidates = [rcrs_server_dir / 'kill.sh', rcrs_server_dir / 'scripts' / 'kill.sh', rcrs_server_dir / 'boot' / 'kill.sh']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = list(rcrs_server_dir.rglob('kill.sh'))
    if found:
        return found[0]
    raise FileNotFoundError(f'kill.sh not found under {rcrs_server_dir}')


def _relative(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir)


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _terminate(proc: subprocess.Popen[str] | None, timeout: int = 10) -> None:
    if proc is None or proc.poll() is not None:
        return
    _kill_process_tree(proc)
    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _wait_for_agent_prompt(proc: subprocess.Popen[str], log_fp, startup_timeout_seconds: int, line_wait_seconds: int) -> bool:
    deadline = time.time() + startup_timeout_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        line = ''
        if proc.stdout is not None:
            try:
                line = proc.stdout.readline()
            except Exception:
                line = ''
        if line:
            log_fp.write(line)
            log_fp.flush()
            if 'Start your agents' in line:
                return True
        else:
            time.sleep(line_wait_seconds)
    return False


def _start_agent(command: str, project_root: Path, log_path: Path, env_overrides: dict[str, str] | None = None) -> subprocess.Popen[str]:
    env = os.environ.copy()
    agent_root = project_root / 'BaseRescueAgent'
    src_path = agent_root / 'src'
    existing = env.get('PYTHONPATH', 'src')
    env['PYTHONPATH'] = str(src_path) if not existing else f"{src_path}:{existing}"
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})
    log_file = log_path.open('w', encoding='utf-8')
    args = shlex.split(command)
    proc = subprocess.Popen(args, cwd=str(agent_root), stdout=log_file, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid, env=env)
    proc._ml_log_file = log_file
    return proc


def _close_agent_log(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    log_file = getattr(proc, '_ml_log_file', None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass


def _cleanup_rcrs(kill_script: Path, rcrs_server_dir: Path, wait_seconds: int = 2) -> None:
    subprocess.run(['bash', str(kill_script)], cwd=str(rcrs_server_dir), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait_seconds)


def _read_expected_timesteps(config_dir: Path) -> Optional[int]:
    for path in [config_dir / 'kernel.cfg', config_dir / 'config.cfg', *config_dir.rglob('kernel.cfg')]:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for line in text.splitlines():
            if 'kernel.timesteps' in line:
                m = re.search(r'(\d+)', line)
                if m:
                    return int(m.group(1))
    return None


def _snapshot_log_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.resolve()) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in LOG_SUFFIXES}


def _collect_new_log_files(root: Path, before: set[str]) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in LOG_SUFFIXES and str(p.resolve()) not in before:
            files.append(p)
    files.sort()
    return files


def _extract_max_tick(paths: list[Path]) -> int:
    best = 0
    for path in paths:
        try:
            with path.open('r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    m = TICK_RE.search(line)
                    if not m:
                        continue
                    value = m.group(1) or m.group(2)
                    if value:
                        best = max(best, int(value))
        except Exception:
            continue
    return best


def _safe_load_meta(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _copy_logs(paths: list[Path], destination_root: Path) -> list[Path]:
    copied: list[Path] = []
    if not paths:
        return copied
    destination_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination_root / path.name
        stem = target.stem
        suffix = target.suffix
        counter = 1
        while target.exists():
            target = destination_root / f'{stem}_{counter}{suffix}'
            counter += 1
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def run_generated_maps(generated_dirs: List[Path], cfg: RunnerConfig) -> List[RunResult]:
    script_dir = cfg.script_path.parent
    kill_script = find_kill_script(cfg.rcrs_server_dir)
    results: List[RunResult] = []
    raw_logs_root = cfg.project_root / 'BaseRescueAgent' / 'runtime' / 'raw_logs'
    my_logs_root = cfg.project_root / 'BaseRescueAgent' / 'runtime' / 'my_logs'
    debug_logs_root = cfg.project_root / 'BaseRescueAgent' / 'runtime' / 'log'

    for generated_dir in generated_dirs:
        meta = _safe_load_meta(generated_dir / 'meta.json')
        map_name = generated_dir.name
        size_group = str(meta.get('size_group') or 'unknown')
        base_map = str(meta.get('base_map') or '')
        family = str(meta.get('family') or '')

        map_arg = _relative(script_dir, generated_dir / 'map')
        generated_config_dir = generated_dir / 'config'
        config_dir = generated_config_dir if generated_config_dir.exists() else (cfg.project_root / 'test_maps' / 'config')
        config_arg = _relative(script_dir, config_dir)
        cmd = ['bash', str(cfg.script_path), '-m', map_arg, '-c', config_arg]

        run_dir = generated_dir / 'run_logs'
        run_dir.mkdir(exist_ok=True)
        stdout_path = run_dir / 'server_stdout.log'
        agent_stdout_path = run_dir / 'agent_stdout.log'
        status_path = run_dir / 'run_status.json'
        agent_runtime_dir = run_dir / 'agent_runtime_logs'
        raw_copy_dir = agent_runtime_dir / 'raw'
        debug_copy_dir = agent_runtime_dir / 'debug'
        text_copy_dir = agent_runtime_dir / 'text'

        start_epoch = time.time()
        agent_proc: subprocess.Popen[str] | None = None
        proc: subprocess.Popen[str] | None = None
        finished = False
        failure_reason: Optional[str] = None
        before_raw = _snapshot_log_files(raw_logs_root)
        before_my = _snapshot_log_files(my_logs_root)
        before_debug = _snapshot_log_files(debug_logs_root)
        expected_timesteps = _read_expected_timesteps(config_dir)
        max_tick_seen = 0
        last_progress = time.time()
        agent_run_id = f'{map_name}__{time.strftime("%Y%m%d_%H%M%S")}'

        with stdout_path.open('w', encoding='utf-8') as log_fp:
            log_fp.write('Pre-run cleanup with kill.sh\n')
            log_fp.flush()
            _cleanup_rcrs(kill_script, cfg.rcrs_server_dir)
            proc = subprocess.Popen(cmd, cwd=str(script_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid, bufsize=1)
            try:
                prompt_seen = _wait_for_agent_prompt(proc, log_fp, cfg.startup_timeout_seconds, cfg.line_wait_seconds)
                if not prompt_seen:
                    failure_reason = 'server_prompt_timeout'
                else:
                    agent_proc = _start_agent(
                        cfg.agent_command,
                        cfg.project_root,
                        agent_stdout_path,
                        env_overrides={
                            'RRS_BENCHMARK_RUN_ID': agent_run_id,
                            'RRS_BENCHMARK_MAP_NAME': map_name,
                            'RRS_BENCHMARK_VARIANT': cfg.variant_label,
                        },
                    )
                    time.sleep(4)
                    if agent_proc.poll() is not None:
                        failure_reason = 'agent_exited_immediately'
                    else:
                        while True:
                            if proc.poll() is not None:
                                finished = proc.returncode == 0
                                break
                            if agent_proc.poll() is not None:
                                time.sleep(cfg.settle_after_finish_seconds)
                                finished = True
                                break
                            new_logs = (
                                _collect_new_log_files(raw_logs_root, before_raw)
                                + _collect_new_log_files(my_logs_root, before_my)
                                + _collect_new_log_files(debug_logs_root, before_debug)
                            )
                            max_tick_seen = max(max_tick_seen, _extract_max_tick(new_logs))
                            if max_tick_seen > 0:
                                last_progress = time.time()
                            if expected_timesteps is not None and max_tick_seen >= expected_timesteps:
                                finished = True
                                failure_reason = None
                                log_fp.write(f'Reached expected timesteps: {expected_timesteps}\n')
                                log_fp.flush()
                                break
                            if time.time() - last_progress > cfg.stall_timeout_seconds:
                                failure_reason = f'stalled_at_tick_{max_tick_seen}'
                                break
                            if time.time() - start_epoch > cfg.server_timeout_seconds:
                                failure_reason = f'server_timeout_at_tick_{max_tick_seen}'
                                break
                            time.sleep(1)
            finally:
                _terminate(agent_proc)
                _close_agent_log(agent_proc)
                _cleanup_rcrs(kill_script, cfg.rcrs_server_dir)
                _terminate(proc, timeout=5)
                if proc and proc.stdout is not None:
                    try:
                        tail = proc.stdout.read() or ''
                    except Exception:
                        tail = ''
                    if tail:
                        log_fp.write(tail)
                        log_fp.flush()

        wall_runtime_sec = max(time.time() - start_epoch, 0.0)
        new_raw_logs = _collect_new_log_files(raw_logs_root, before_raw)
        new_text_logs = _collect_new_log_files(my_logs_root, before_my)
        new_debug_logs = _collect_new_log_files(debug_logs_root, before_debug)
        copied_raw_logs = _copy_logs(new_raw_logs, raw_copy_dir)
        copied_text_logs = _copy_logs(new_text_logs, text_copy_dir)
        copied_debug_logs = _copy_logs(new_debug_logs, debug_copy_dir)

        status = {
            'map_name': map_name,
            'size_group': size_group,
            'base_map': base_map,
            'family': family,
            'command': ' '.join(shlex.quote(part) for part in cmd),
            'agent_command': cfg.agent_command,
            'script_dir': str(script_dir),
            'started_at_epoch': start_epoch,
            'finished_at_epoch': time.time(),
            'server_return_code': None if proc is None else proc.returncode,
            'agent_return_code': None if agent_proc is None else agent_proc.returncode,
            'agent_started': agent_proc is not None,
            'agent_run_id': agent_run_id,
            'finished': finished,
            'failure_reason': failure_reason,
            'max_tick_seen': max_tick_seen,
            'expected_timesteps': expected_timesteps,
            'wall_runtime_sec': round(wall_runtime_sec, 3),
            'copied_raw_logs': [str(path) for path in copied_raw_logs],
            'copied_debug_logs': [str(path) for path in copied_debug_logs],
            'copied_text_logs': [str(path) for path in copied_text_logs],
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append(
            RunResult(
                map_name=map_name,
                size_group=size_group,
                base_map=base_map,
                family=family,
                generated_dir=generated_dir,
                run_dir=run_dir,
                status_path=status_path,
                server_stdout_path=stdout_path,
                agent_stdout_path=agent_stdout_path,
                agent_run_id=agent_run_id,
                raw_log_paths=copied_raw_logs,
                debug_log_paths=copied_debug_logs,
                text_log_paths=copied_text_logs,
                server_return_code=None if proc is None else proc.returncode,
                agent_return_code=None if agent_proc is None else agent_proc.returncode,
                finished=finished,
                failure_reason=failure_reason,
                max_tick_seen=max_tick_seen,
                expected_timesteps=expected_timesteps,
                wall_runtime_sec=wall_runtime_sec,
            )
        )

    return results
