"""Explicit idle-only local scope transition; preserve ownership/history and locks."""
import hashlib
import os
from pathlib import Path
import sys
from .config import load_config
from .core import Admission, AtomicStore, NativeLock, RecoveryRequired, FINAL, encode, lock_identity, strict_json, timestamp


def transition_scope(old_config, new_config, evidence, *, rollback=False):
    old, old_targets, old_state, old_locks, old_scope = load_config(old_config)
    new, new_targets, new_state, new_locks, new_scope = load_config(new_config)
    larger, smaller = (old, new) if rollback else (new, old)
    larger_targets, smaller_targets = (old_targets, new_targets) if rollback else (new_targets, old_targets)
    check = dict(larger, targets={n: v for n, v in larger['targets'].items() if n in smaller['targets']})
    additions = set(larger_targets) - set(smaller_targets)
    if (check != smaller or not additions or any(larger_targets[n].mode_switch is None for n in additions) or
            old_state != new_state or old_locks != new_locks or old_scope == new_scope):
        raise ValueError('only_additive_dual_target_scope_transition_permitted')
    if (not evidence.get('approval_ref') or evidence.get('old_scope') != old_scope or
            evidence.get('new_scope') != new_scope or any(evidence.get(k) is not True for k in
            ('callers_excluded', 'backend_quiescent', 'cleanup_verified', 'gpu1_restored', 'backend_ready',
             'placement_verified'))):
        raise ValueError('scope_transition_evidence_required')
    if not (old_state / 'ownership.json').is_file():
        raise RecoveryRequired('existing_ownership_required_no_bootstrap')
    store = AtomicStore(old_state)
    supervisor = NativeLock(store.path('supervisor.lock'))
    if not supervisor.acquire(0):
        raise RecoveryRequired('scope_transition_supervisor_active')
    locks = []
    try:
        with NativeLock(store.path('jobs.lock')).held(), NativeLock(store.path('admission.lock')).held():
            state = store.read('ownership.json')
            if not isinstance(state, dict):
                raise RecoveryRequired('existing_ownership_required_no_bootstrap')
            current_scope = state.get('scope')
            if current_scope not in (old_scope, new_scope):
                raise RecoveryRequired('scope_transition_baseline_changed')
            # Avoid constructor initialization: never create/reset ownership here.
            gate = Admission.__new__(Admission)
            gate.store, gate.scope, gate.lock_paths = store, current_scope, old_locks
            gate._validate(state)
            if not state['ready'] or state['owners'] or any(j['state'] not in FINAL for j in store.jobs()):
                raise RecoveryRequired('scope_transition_requires_resolved_idle_journal')
            for resource, path in sorted(old_locks.items()):
                if not path.is_file() or lock_identity(path.stat()) != state['lock_identity'][resource]:
                    raise RecoveryRequired('physical_lock_identity_changed')
                lock = NativeLock(path, create=False)
                if not lock.acquire(0):
                    raise RecoveryRequired('scope_transition_resource_busy')
                locks.append(lock)
                if lock_identity(os.fstat(lock.fd)) != state['lock_identity'][resource]:
                    raise RecoveryRequired('physical_lock_identity_changed_during_acquisition')
            if current_scope == new_scope:
                previous = state.get('scope_transitions', [])
                if not previous or any(previous[-1].get(k) != evidence[k]
                                       for k in ('old_scope', 'new_scope', 'approval_ref')):
                    raise RecoveryRequired('scope_transition_outcome_requires_review')
                return {'scope': new_scope, 'already_recorded': True, 'journal_preserved': True}
            if rollback:
                previous = state.get('scope_transitions', [])
                if (not previous or previous[-1].get('new_scope') != old_scope or
                        previous[-1].get('old_scope') != new_scope or
                        evidence.get('rollback_of') != previous[-1].get('approval_ref')):
                    raise ValueError('exact_previous_scope_transition_rollback_required')
            state.setdefault('scope_transitions', []).append(dict(evidence, rollback=rollback,
                                                                  transitioned_at=timestamp()))
            state['scope'] = new_scope
            store.write('ownership.json', state)
            return {'scope': new_scope, 'already_recorded': False, 'journal_preserved': True}
    finally:
        for lock in reversed(locks):
            lock.close()
        supervisor.close()


WALLPAPER_ATTESTATIONS = (
    'operator_approved_wallpaper_workflow_transition', 'callers_excluded',
    'supervisor_stopped', 'automatic_wallpaper_paused', 'backend_quiescent', 'cleanup_verified',
)


def _wallpaper_delta(old, new, old_targets, new_targets, rollback):
    full, reduced = (new_targets, old_targets) if rollback else (old_targets, new_targets)
    if ('wallpaper' not in full or 'wallpaper' not in reduced or
            full['wallpaper'].kind != 'comfy' or reduced['wallpaper'].kind != 'comfy'):
        raise ValueError('wallpaper_comfy_target_required')
    before, after = full['wallpaper'].workflow, reduced['wallpaper'].workflow
    if (before.get('95', {}).get('class_type') != 'LayerUtility: PurgeVRAM' or
            encode({n: v for n, v in before.items() if n != '95'}) != encode(after)):
        raise ValueError('only_wallpaper_purge95_removal_permitted')
    def references_95(value):
        if isinstance(value, list):
            return (len(value) == 2 and value[0] == '95' and type(value[1]) is int or
                    any(references_95(v) for v in value))
        return isinstance(value, dict) and any(references_95(v) for v in value.values())
    if any(references_95(node.get('inputs', {})) for node in after.values()):
        raise ValueError('removed_purge95_still_referenced')
    old_wallpaper, new_wallpaper = old['targets']['wallpaper'], new['targets']['wallpaper']
    keys = ('workflow_template', 'workflow_sha256')
    if any(not isinstance(w.get(k), str) or not w[k] for w in (old_wallpaper, new_wallpaper) for k in keys):
        raise ValueError('hash_bound_wallpaper_templates_required')
    expected = dict(old, targets=dict(old['targets'], wallpaper=dict(old_wallpaper,
                    **{k: new_wallpaper[k] for k in keys})))
    if encode(expected) != encode(new):
        raise ValueError('only_wallpaper_template_binding_may_change')


CONDITIONING_ATTESTATIONS = (
    'operator_approved_wallpaper_conditioning_transition', *WALLPAPER_ATTESTATIONS[1:],
    'both_gpus_quiescent', 'physical_gpu_uuids_verified', 'staged_workflow_bytes_verified',
)


def _wallpaper_conditioning_delta(old, new, old_targets, new_targets, rollback):
    cpu, dual = (new, old) if rollback else (old, new)
    cpu_targets, dual_targets = (new_targets, old_targets) if rollback else (old_targets, new_targets)
    if 'wallpaper' not in cpu_targets or 'wallpaper' not in dual_targets:
        raise ValueError('wallpaper_comfy_target_required')
    before, after = cpu_targets['wallpaper'], dual_targets['wallpaper']
    if (before.kind != 'comfy' or before.resources != ('gpu0',) or
            after.kind != 'comfy' or after.resources != ('gpu0', 'gpu1') or
            before.conditioning_idle_memory_mb is not None or after.conditioning_idle_memory_mb is None):
        raise ValueError('exact_cpu_to_dual_wallpaper_resources_required')
    uuids = after.conditioning_gpu_uuids()
    node = before.workflow.get('3', {})
    inputs = node.get('inputs', {})
    if (node.get('class_type') != 'CLIPLoader' or
            set(inputs) != {'clip_name', 'type', 'device'} or
            inputs.get('clip_name') != 'qwen_2.5_vl_7b_fp8_scaled.safetensors' or
            inputs.get('type') != 'qwen_image' or inputs.get('device') != 'cpu' or
            '95' in before.workflow or
            before.workflow.get('7', {}).get('class_type') != 'QwenImageIntegratedKSampler' or
            before.workflow['7'].get('inputs', {}).get('clip') != ['3', 0]):
        raise ValueError('approved_cpu_wallpaper_loader_graph_required')
    expected_node = dict(node, class_type='AIHubQwenGPU1CLIPLoader', inputs={
        'clip_name': inputs['clip_name'], 'gpu0_uuid': uuids['gpu0'], 'gpu1_uuid': uuids['gpu1']})
    if encode(dict(before.workflow, **{'3': expected_node})) != encode(after.workflow):
        raise ValueError('only_wallpaper_gpu1_loader_change_permitted')
    cpu_raw, dual_raw = cpu['targets']['wallpaper'], dual['targets']['wallpaper']
    keys = ('workflow_template', 'workflow_sha256')
    if (any(not isinstance(w.get(k), str) or not w[k] for w in (cpu_raw, dual_raw) for k in keys) or
            'conditioning_idle_memory_mb' in cpu_raw):
        raise ValueError('hash_bound_cpu_and_dual_wallpaper_templates_required')
    expected = dict(cpu, targets=dict(cpu['targets'], wallpaper=dict(cpu_raw,
        resources=['gpu0', 'gpu1'], conditioning_idle_memory_mb=after.conditioning_idle_memory_mb,
        **{k: dual_raw[k] for k in keys})))
    if encode(expected) != encode(dual):
        raise ValueError('only_wallpaper_resources_conditioning_calibration_template_may_change')


def transition_wallpaper_conditioning(old_config, new_config, evidence, *, rollback=False):
    """Exact CPU-wallpaper <-> GPU1-conditioning transition; no backend/config actions."""
    return _transition_wallpaper(old_config, new_config, evidence,
                                 kind='wallpaper_gpu1_conditioning', rollback=rollback)


def _wallpaper_on_demand_delta(old, new, old_targets, new_targets, rollback):
    # The current checkpoint is CPU/GPU0 -> dual conditioning + on-demand.
    # Reuse the exact reviewed graph/resource validator; only these three extra
    # lifecycle fields may accompany that same transition, never model changes.
    cpu, demand = (new, old) if rollback else (old, new)
    cpu_targets, demand_targets = (new_targets, old_targets) if rollback else (old_targets, new_targets)
    if (cpu_targets['wallpaper'].lifecycle != 'always_on' or
            demand_targets['wallpaper'].lifecycle != 'on_demand'):
        raise ValueError('exact_cpu_to_dual_on_demand_wallpaper_required')
    keys = ('lifecycle', 'startup_timeout', 'shutdown_timeout')
    if any(k in cpu['targets']['wallpaper'] for k in keys) or not all(
            k in demand['targets']['wallpaper'] for k in keys):
        raise ValueError('explicit_new_lifecycle_fields_only')
    stripped = dict(demand, targets=dict(demand['targets'], wallpaper={
        k: v for k, v in demand['targets']['wallpaper'].items() if k not in keys}))
    _wallpaper_conditioning_delta(cpu, stripped, cpu_targets, demand_targets, False)


def transition_wallpaper_on_demand(old_config, new_config, evidence, *, rollback=False):
    return _transition_wallpaper(old_config, new_config, evidence,
                                 kind='wallpaper_gpu1_on_demand', rollback=rollback)


def transition_wallpaper_workflow(old_config, new_config, evidence, *, rollback=False):
    """Idle-only removal of node 95; commit journal, never activate config or call a backend."""
    return _transition_wallpaper(old_config, new_config, evidence,
                                 kind='wallpaper_remove_purge95', rollback=rollback)


def _transition_wallpaper(old_config, new_config, evidence, *, kind, rollback):
    # Two fixed validators share the existing transaction; no arbitrary-scope interface.
    if kind == 'wallpaper_remove_purge95':
        validate_delta, attestations = _wallpaper_delta, WALLPAPER_ATTESTATIONS
    elif kind == 'wallpaper_gpu1_conditioning':
        validate_delta, attestations = _wallpaper_conditioning_delta, CONDITIONING_ATTESTATIONS
    elif kind == 'wallpaper_gpu1_on_demand':
        validate_delta = _wallpaper_on_demand_delta
        attestations = ('operator_approved_wallpaper_on_demand_transition',
                        *CONDITIONING_ATTESTATIONS[1:], 'wallpaper_stopped')
    else:
        raise ValueError('unsupported_wallpaper_transition')
    if not sys.platform.startswith('linux'):
        raise RecoveryRequired('wallpaper_transition_requires_linux')
    configs = tuple(Path(p) for p in (old_config, new_config))
    frozen = tuple(p.read_bytes() for p in configs)
    old, old_targets, state_dir, locks, old_scope = load_config(old_config)
    new, new_targets, new_state, new_locks, new_scope = load_config(new_config)
    validate_delta(old, new, old_targets, new_targets, rollback)
    if (state_dir != new_state or
            {r: p.resolve() for r, p in locks.items()} != {r: p.resolve() for r, p in new_locks.items()} or
            old_scope == new_scope):
        raise ValueError('wallpaper_transition_storage_or_scope_changed')
    digest = lambda data: hashlib.sha256(data).hexdigest()
    bindings = dict(old_scope=old_scope, new_scope=new_scope,
                    old_config_sha256=digest(frozen[0]), new_config_sha256=digest(frozen[1]),
                    old_workflow_sha256=old['targets']['wallpaper']['workflow_sha256'],
                    new_workflow_sha256=new['targets']['wallpaper']['workflow_sha256'])
    references = ('approval_ref', 'exclusion_evidence_ref', 'backend_evidence_ref', 'ownership_sha256')
    if kind in ('wallpaper_gpu1_conditioning', 'wallpaper_gpu1_on_demand'):
        dual = (old_targets if rollback else new_targets)['wallpaper']
        bindings.update(physical_gpu_uuids=dual.conditioning_gpu_uuids(),
            calibrated_memory_mb={'gpu0': dual.idle_memory_mb, 'gpu1': dual.conditioning_idle_memory_mb},
            media_workflow_sha256=bindings['new_workflow_sha256'])
        references += ('workflow_staging_evidence_ref', 'calibration_evidence_ref')
    if (not isinstance(evidence, dict) or
            any(not isinstance(evidence.get(k), str) or not evidence[k].strip() for k in references) or
            any(evidence.get(k) is not True for k in attestations) or
            any(encode(evidence.get(k)) != encode(v) for k, v in bindings.items()) or
            (not rollback and 'rollback_of' in evidence)):
        raise ValueError('wallpaper_transition_exact_operator_evidence_required')
    if not (state_dir / 'ownership.json').is_file():
        raise RecoveryRequired('existing_ownership_required_no_bootstrap')
    store, held = AtomicStore(state_dir), []
    try:
        for name in ('maintenance.lock', 'supervisor.lock', 'jobs.lock', 'admission.lock'):
            lock = NativeLock(store.path(name))
            if not lock.acquire(0):
                raise RecoveryRequired('wallpaper_transition_maintenance_or_supervisor_busy')
            held.append(lock)
        state_bytes = store.path('ownership.json').read_bytes()
        state = strict_json(state_bytes)
        gate = Admission.__new__(Admission)
        gate.store, gate.scope, gate.lock_paths = store, state.get('scope') if isinstance(state, dict) else None, locks
        gate._validate(state)
        if kind == 'wallpaper_gpu1_on_demand' and state.get('recovery_policy'):
            raise RecoveryRequired('lifecycle_scope_change_requires_separate_recovery_policy_migration')
        if gate.scope not in (old_scope, new_scope):
            raise RecoveryRequired('wallpaper_transition_baseline_changed')
        if not state['ready'] or state['owners'] or any(j.get('state') not in FINAL for j in store.jobs()):
            raise RecoveryRequired('wallpaper_transition_requires_resolved_idle_journal')
        gate._check_physical_identities(state)
        for resource, path in sorted(locks.items()):
            lock = NativeLock(path, create=False)
            if not lock.acquire(0):
                raise RecoveryRequired('wallpaper_transition_resource_busy')
            held.append(lock)
            if lock_identity(os.fstat(lock.fd)) != state['lock_identity'][resource]:
                raise RecoveryRequired('physical_lock_identity_changed_during_acquisition')
        # Freeze both file bindings and resolved workflow contents, then recheck pathname identities.
        if (tuple(p.read_bytes() for p in configs) != frozen or
                load_config(old_config)[-1] != old_scope or load_config(new_config)[-1] != new_scope):
            raise RecoveryRequired('wallpaper_transition_inputs_changed')
        gate._check_physical_identities(state)
        history = state.get('scope_transitions', [])
        if not isinstance(history, list):
            raise RecoveryRequired('wallpaper_transition_history_invalid')
        evidence_hash = digest(encode(evidence))
        if gate.scope == new_scope:
            if (not history or history[-1].get('kind') != kind or
                    history[-1].get('rollback') is not rollback or
                    history[-1].get('evidence_sha256') != evidence_hash or
                    any(history[-1].get(k) != v for k, v in bindings.items())):
                raise RecoveryRequired('wallpaper_transition_outcome_requires_review')
            # Complete a possibly interrupted directory fsync; never append twice.
            for path, flags in ((store.path('ownership.json'), os.O_RDONLY),
                                (state_dir, os.O_RDONLY | os.O_DIRECTORY)):
                fd = os.open(path, flags)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return {'scope': new_scope, 'already_recorded': True, 'journal_preserved': True}
        if digest(state_bytes) != evidence['ownership_sha256']:
            raise RecoveryRequired('wallpaper_transition_journal_evidence_changed')
        if any(row.get('approval_ref') == evidence['approval_ref'] for row in history):
            raise ValueError('wallpaper_transition_fresh_approval_required')
        if rollback and (not history or history[-1].get('kind') != kind or
                history[-1].get('rollback') is not False or
                history[-1].get('old_scope') != new_scope or history[-1].get('new_scope') != old_scope or
                evidence.get('rollback_of') != history[-1].get('approval_ref')):
            raise ValueError('exact_previous_wallpaper_transition_rollback_required')
        state['scope_transitions'] = history + [dict(evidence, kind=kind, rollback=rollback,
            evidence_sha256=evidence_hash, transitioned_at=timestamp(), provenance='local_operator_attestation')]
        state['scope'] = new_scope
        store.write('ownership.json', state)
        return {'scope': new_scope, 'already_recorded': False, 'journal_preserved': True}
    finally:
        for lock in reversed(held):
            lock.close()
