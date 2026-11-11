"""Local missing-lock migration; retain one unresolved GPU0 Comfy owner.

Never provision physical files, change configs, recover jobs, or call backends.
The operator activates the prepared config separately after this journal commit.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys

from .config import load_config
from .core import Admission, AtomicStore, NativeLock, RecoveryRequired, FINAL, encode, lock_identity, strict_json, timestamp


ATTESTATIONS = (
    'operator_approved_physical_lock_migration', 'callers_excluded',
    'backend_quiescent', 'original_execution_terminated',
    'original_gpu_context_absent', 'no_conflicting_gpu_work',
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _pending_jobs(store, job_id, owner):
    pending = {}
    waiting_fields = {'request_id', 'request_hash', 'target', 'operation', 'payload', 'state',
                      'created_at', 'updated_at', 'error_code', 'resources', 'phase', 'lease_id',
                      'backend_detail', 'completion_verified', 'host_stage', 'dual_gpu_session'}
    for path in sorted(store.path('jobs').glob('*.json')):
        raw = path.read_bytes()
        job = strict_json(raw)
        if job.get('state') in FINAL or job.get('request_id') == job_id:
            continue
        # Only unadmitted API wallpaper work may accompany the retained owner.
        # It remains untouched here; ordinary Runner startup fails it without replay.
        if (set(job) - waiting_fields or job.get('state') != 'waiting' or job.get('target') != owner['target'] or
                job.get('resources') != ['gpu0'] or job.get('request_id') != path.stem or
                not isinstance(job.get('request_hash'), str) or not job['request_hash'] or
                job.get('host_stage') is True or job.get('dual_gpu_session') is True or
                any(job.get(k) is not None for k in
                    ('phase', 'lease_id', 'backend_detail', 'completion_verified', 'error_code'))):
            raise RecoveryRequired('migration_other_active_or_uncertain_job')
        pending[path.stem] = _digest(raw)
    return pending


def _absent(paths):
    # lstat also rejects dangling symlinks; missing files have no inspected identity.
    for path in paths.values():
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RecoveryRequired('migration_old_lock_not_absent')


def _new_identities(state_dir, paths):
    for directory in (state_dir, state_dir / 'physical-locks'):
        try:
            value = directory.lstat()
        except OSError:
            raise RecoveryRequired('migration_private_directory_missing') from None
        if (not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o700 or
                (value.st_uid, value.st_gid) != (os.geteuid(), os.getegid())):
            raise RecoveryRequired('migration_private_directory_invalid')
    identities = {}
    for resource, path in paths.items():
        try:
            value = path.lstat()
        except OSError:
            raise RecoveryRequired('migration_new_lock_missing') from None
        if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size != 0 or
                stat.S_IMODE(value.st_mode) != 0o600 or
                (value.st_uid, value.st_gid) != (os.geteuid(), os.getegid())):
            raise RecoveryRequired('migration_new_lock_invalid')
        identities[resource] = lock_identity(value)
    if len({(v['device'], v['inode']) for v in identities.values()}) != 2:
        raise RecoveryRequired('migration_new_locks_not_distinct')
    return identities


def migrate_physical_locks(old_config, new_config, job_id, evidence):
    """One journal commit, with config activation deliberately outside the operation.

    Explicit operator attestations cover exclusion and backend termination: neither
    an absent file nor an acquired new flock proves an unlinked old FD is inactive.
    """
    if not sys.platform.startswith('linux'):
        raise RecoveryRequired('physical_lock_migration_requires_linux')
    old_bytes, new_bytes = Path(old_config).read_bytes(), Path(new_config).read_bytes()
    old, targets, state_dir, old_paths, old_scope = load_config(old_config)
    new, _, new_state, new_paths, new_scope = load_config(new_config)
    expected_paths = {r: state_dir / 'physical-locks' / (r + '.lock') for r in ('gpu0', 'gpu1')}
    if (dict(old, lock_paths=new['lock_paths']) != new or state_dir != new_state or
            new_paths != expected_paths or old_scope == new_scope or
            set(old_paths.values()) & set(new_paths.values())):
        raise ValueError('migration_only_persistent_lock_path_change_permitted')
    if (not isinstance(evidence, dict) or
            not isinstance(evidence.get('approval_ref'), str) or not evidence['approval_ref'].strip() or
            any(evidence.get(k) is not True for k in ATTESTATIONS) or
            any(not isinstance(evidence.get(k), str) or not evidence[k].strip() for k in
                ('exclusion_evidence_ref', 'backend_evidence_ref', 'gpu_evidence_ref')) or
            evidence.get('old_scope') != old_scope or evidence.get('new_scope') != new_scope or
            evidence.get('old_config_sha256') != _digest(old_bytes) or
            evidence.get('new_config_sha256') != _digest(new_bytes)):
        raise ValueError('migration_operator_evidence_required')
    if not (state_dir / 'ownership.json').is_file():
        raise RecoveryRequired('migration_existing_journal_required')
    store = AtomicStore(state_dir)
    locks = []
    try:
        # Same supervisor -> jobs -> admission ordering as the existing scope tool.
        for name in ('maintenance.lock', 'supervisor.lock', 'jobs.lock', 'admission.lock'):
            lock = NativeLock(store.path(name))
            if not lock.acquire(0):
                raise RecoveryRequired('migration_maintenance_or_journal_busy')
            locks.append(lock)
        state_bytes = store.path('ownership.json').read_bytes()
        state = strict_json(state_bytes)
        gate = Admission.__new__(Admission)
        gate.scope, gate.lock_paths = state.get('scope'), new_paths
        gate._validate(state)
        if gate.scope not in (old_scope, new_scope) or not state['ready']:
            raise RecoveryRequired('migration_scope_or_readiness_changed')
        if set(state['owners']) != {'gpu0'}:
            raise RecoveryRequired('migration_single_retained_gpu0_owner_required')
        owner = state['owners']['gpu0']
        if owner != evidence.get('owner') or owner['job_id'] != job_id:
            raise RecoveryRequired('migration_owner_or_job_evidence_changed')
        job_path = store.path(store.job_name(job_id))
        job_bytes = job_path.read_bytes()
        job = strict_json(job_bytes)
        target = targets.get(owner['target'])
        if (owner['phase'] not in ('reconciling', 'cleanup_pending') or
                job.get('state') not in ('reconciling', 'cleanup_pending') or
                job.get('request_id') != job_id or job.get('resources') != ['gpu0'] or
                any(job.get(k) != owner[k] for k in ('lease_id', 'request_hash', 'target')) or
                job.get('dual_gpu_session') is True or job.get('host_stage') is True or
                target is None or target.kind != 'comfy' or target.resources != ('gpu0',) or
                evidence.get('job_sha256') != _digest(job_bytes)):
            raise RecoveryRequired('migration_owner_or_job_evidence_changed')
        if _pending_jobs(store, job_id, owner) != evidence.get('pending_job_sha256'):
            raise RecoveryRequired('migration_pending_job_evidence_changed')
        _absent(old_paths)
        identities = _new_identities(state_dir, new_paths)
        if identities != evidence.get('new_lock_identity'):
            raise RecoveryRequired('migration_new_lock_identity_changed')
        for resource, path in sorted(new_paths.items()):
            lock = NativeLock(path, create=False)
            try:
                acquired = lock.acquire(0)
            except OSError:
                raise RecoveryRequired('migration_new_lock_open_failed') from None
            if not acquired:
                raise RecoveryRequired('migration_new_lock_busy')
            locks.append(lock)
            if lock_identity(os.fstat(lock.fd)) != identities[resource]:
                raise RecoveryRequired('migration_new_lock_identity_changed_during_acquisition')
        history = state.get('physical_lock_migrations', [])
        if not isinstance(history, list):
            raise RecoveryRequired('migration_history_invalid')
        old_identity = evidence.get('old_lock_identity')
        if (not isinstance(old_identity, dict) or set(old_identity) != {'gpu0', 'gpu1'} or
                any(not isinstance(v, dict) or set(v) != {'device', 'inode', 'mode', 'uid', 'gid'} or
                    any(type(x) is not int or x < 0 for x in v.values())
                    for v in old_identity.values())):
            raise ValueError('migration_recorded_old_identity_required')
        evidence_hash = _digest(encode(evidence))
        if gate.scope == old_scope:
            if (evidence.get('ownership_sha256') != _digest(state_bytes) or
                    evidence.get('old_lock_identity') != state['lock_identity']):
                raise RecoveryRequired('migration_old_journal_evidence_changed')
        else:
            if (not history or history[-1].get('evidence_sha256') != evidence_hash or
                    history[-1].get('old_scope') != old_scope or
                    history[-1].get('new_scope') != new_scope or
                    history[-1].get('old_lock_identity') != evidence.get('old_lock_identity') or
                    state['lock_identity'] != identities):
                raise RecoveryRequired('migration_recorded_outcome_changed')
        # Recheck path and frozen config bindings after all physical locks are held.
        _absent(old_paths)
        if (_new_identities(state_dir, new_paths) != identities or
                Path(old_config).read_bytes() != old_bytes or Path(new_config).read_bytes() != new_bytes):
            raise RecoveryRequired('migration_inputs_changed_during_operation')
        if gate.scope == new_scope:
            # Finish a possibly interrupted directory fsync; no duplicate audit entry.
            for path, flags in ((store.path('ownership.json'), os.O_RDONLY),
                                (state_dir, os.O_RDONLY | os.O_DIRECTORY)):
                fd = os.open(path, flags)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return {'scope': new_scope, 'already_recorded': True, 'owners_retained': True}
        record = dict(evidence, evidence_sha256=evidence_hash, transitioned_at=timestamp(),
                      old_identity_source='persisted_ownership_only',
                      old_file_observations={r: 'absent' for r in old_paths},
                      provenance='local_operator_attestation')
        state['physical_lock_migrations'] = history + [record]
        state['scope'], state['lock_identity'] = new_scope, identities
        store.write('ownership.json', state)
        return {'scope': new_scope, 'already_recorded': False, 'owners_retained': True}
    finally:
        for lock in reversed(locks):
            lock.close()
