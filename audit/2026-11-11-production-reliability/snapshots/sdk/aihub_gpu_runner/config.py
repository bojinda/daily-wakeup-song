"""Trusted local configuration; clients cannot choose resources, URLs or paths."""
import hashlib
from pathlib import Path
from .core import Admission, AtomicStore, Runner, check_id, encode, strict_json
from .backends import Target, HTTPTransport, NvidiaMemoryProbe, NvidiaUUIDMemoryProbe, OllamaBackend, ComfyBackend, physical_gpu_uuid

def load_config(path):
    path = Path(path).resolve()
    value = strict_json(path.read_bytes())
    if not isinstance(value, dict) or set(value) - {"state_dir", "lock_paths", "targets",
                                                   "listen", "auth_token_env", "workers", "host_stages"}:
        raise ValueError("invalid_configuration")
    host = value.get("host_stages", {})
    if (not isinstance(host, dict) or set(host) - {"whisperx"} or
            any(not isinstance(policy, dict) or set(policy) - {"idle_memory_mb", "cleanup_samples", "cleanup_timeout"}
                for policy in host.values())):
        raise ValueError("invalid_host_stage_policy")
    if set(value.get("lock_paths", {})) != {"gpu0", "gpu1"}:
        raise ValueError("both_physical_locks_required")
    state_dir = Path(value["state_dir"])
    if not state_dir.is_absolute():
        state_dir = path.parent / state_dir
    lock_paths = {}
    for name, raw in value["lock_paths"].items():
        p = Path(raw)
        lock_paths[name] = p if p.is_absolute() else path.parent / p
    targets = {}
    for name, settings in value.get("targets", {}).items():
        check_id(name)
        settings = dict(settings)
        if settings.get('lifecycle') == 'on_demand' and not {'startup_timeout', 'shutdown_timeout'} <= settings.keys():
            raise ValueError('explicit_wallpaper_startup_and_shutdown_deadlines_required')
        kind = settings["kind"]
        required = (("gpu0", "gpu1") if kind == "ollama" and settings.get('mode_switch') is not None
                    else ("gpu1",) if kind == "ollama"
                    else ("gpu0", "gpu1") if kind == "comfy" and name == "wallpaper" and
                         settings.get("conditioning_idle_memory_mb") is not None
                    else ("gpu0",) if kind == "comfy" else None)
        if required is None or tuple(settings.get("resources", ())) != required:
            raise ValueError("routine_resource_policy_violation")
        workflow_path = settings.pop("workflow_template", None)
        if workflow_path:
            p = Path(workflow_path)
            if not p.is_absolute():
                p = path.parent / p
            workflow_bytes = p.read_bytes()
            expected = settings.pop("workflow_sha256")
            if hashlib.sha256(workflow_bytes).hexdigest() != expected:
                raise ValueError("workflow_template_hash_mismatch")
            settings["workflow"] = strict_json(workflow_bytes)
        if kind == "ollama" and not settings.get("models"):
            raise ValueError("model_allowlist_required")
        settings["resources"] = tuple(settings["resources"])
        settings["models"] = tuple(settings.get("models", ()))
        targets[name] = Target(name=name, **settings)
        if kind == "comfy" and (targets[name].output_node not in targets[name].workflow or
                                targets[name].media_type not in ("image/png", "audio/mpeg")):
            raise ValueError("intended_comfy_output_required")
    if not targets or type(value.get("workers", 4)) is not int or not 2 <= value.get("workers", 4) <= 32:
        raise ValueError("invalid_targets_or_workers")
    for target in targets.values():
        if target.mode_switch is not None:
            normal = [t for t in targets.values() if t.kind == 'ollama' and t.resources == ('gpu1',)
                      and t.base_url.rstrip('/') == target.base_url.rstrip('/')]
            if len(normal) != 1:
                raise ValueError('dual_target_requires_one_routine_ollama_backend')
    wallpaper = targets.get('wallpaper')
    if wallpaper is not None and wallpaper.conditioning_idle_memory_mb is not None:
        uuids = wallpaper.conditioning_gpu_uuids()
        for target in targets.values():
            if target.mode_switch is not None:
                existing = {r: physical_gpu_uuid(u) for r, u in target.mode_switch['gpu_uuids'].items()}
                if existing != uuids:
                    raise ValueError('wallpaper_gpu_uuids_conflict_with_existing_resource_contract')
    token_env = value.get("auth_token_env", "AIHUB_GPU_RUNNER_TOKEN")
    if not isinstance(token_env, str) or not token_env or "=" in token_env:
        raise ValueError("invalid_token_environment_name")
    scope = hashlib.sha256(encode({"config": value,
        "resolved_locks": {r: str(p.resolve()) for r, p in lock_paths.items()},
        "resolved_templates": {n: t.workflow for n, t in targets.items()}})).hexdigest()
    return value, targets, state_dir.resolve(), lock_paths, scope

def admission_from_config(path):
    value, targets, state, locks, scope = load_config(path)
    return value, targets, Admission(AtomicStore(state), locks, scope, require_existing_locks=True)

def configured_backends(targets):
    backends = {}
    for name, target in targets.items():
        if target.mode_switch is not None:
            from .dual_gpu import SessionOnlyBackend
            backends[name] = SessionOnlyBackend()
            continue
        transport = HTTPTransport(target.base_url)
        if target.kind == "ollama":
            backends[name] = OllamaBackend(target, transport)
        elif target.conditioning_idle_memory_mb is not None:
            uuids = target.conditioning_gpu_uuids()
            backends[name] = ComfyBackend(target, transport,
                probe=NvidiaUUIDMemoryProbe(uuids["gpu0"]),
                conditioning_probe=NvidiaUUIDMemoryProbe(uuids["gpu1"]))
        else:
            backends[name] = ComfyBackend(target, transport, probe=NvidiaMemoryProbe(0))
    return backends


def make_runner(path):
    value, targets, admission = admission_from_config(path)
    return value, Runner(admission, targets, configured_backends(targets), workers=value.get("workers", 4))
