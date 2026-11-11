"""Local, explicitly reviewed contract transition for one interrupted paired lease.

No backend controls. Until finish() commits policy + paired release + verified
boot in one AtomicStore write, the original contract and boot refusal remain.
"""
import copy
import os
import time

from .core import NativeLock, RecoveryRequired, timestamp
from .recovery import (RecoveryManager, RecoveryObserver, PersistenceUncertain,
                       binding, digest, remaining, valid_boot, validate_policy)


RESOURCES = ('gpu0', 'gpu1')
REVIEW_FLAGS = {'complete_configuration_reviewed', 'gpu_exposure_reviewed',
                'mounts_reviewed', 'executable_settings_reviewed',
                'operator_approved_contract_transition', 'participating_callers_excluded'}
BINDINGS = {'job_id', 'request_hash', 'lease_id', 'target', 'previous_boot', 'current_boot',
            'scope', 'policy_id', 'old_contract_sha256', 'new_contract_sha256',
            'container_id', 'image', 'gpu_uuids', 'lock_identity', 'reviewed_metadata_sha256'}


def require(condition, reason):
    if not condition:
        raise RecoveryRequired('reboot_contract_' + reason)


def refusal(state):
    return {k: copy.deepcopy(state.get(k)) for k in (
        'boot_recovery_status', 'boot_recovery_error', 'boot_recovery_attempt_boot')}


def reviewed_approval(evidence, reviewed):
    require(isinstance(evidence, dict) and set(evidence) == BINDINGS | REVIEW_FLAGS | {'approval_ref'},
            'explicit_approval_shape_required')
    require(all(evidence[k] is True for k in REVIEW_FLAGS) and
            isinstance(evidence['approval_ref'], str) and bool(evidence['approval_ref'].strip()),
            'explicit_operator_review_required')
    require(isinstance(reviewed, dict) and digest(reviewed) == evidence['reviewed_metadata_sha256'],
            'reviewed_metadata_changed')
    require(RecoveryObserver.contract(reviewed) == evidence['new_contract_sha256'] and
            reviewed.get('Id') == evidence['container_id'] and reviewed.get('Image') == evidence['image'],
            'reviewed_contract_binding_changed')
    require(evidence['old_contract_sha256'] != evidence['new_contract_sha256'], 'distinct_contracts_required')
    require(valid_boot(evidence['previous_boot']) and valid_boot(evidence['current_boot']) and
            evidence['previous_boot'] != evidence['current_boot'], 'genuine_reboot_required')
    require(isinstance(evidence['gpu_uuids'], dict) and set(evidence['gpu_uuids']) == set(RESOURCES),
            'paired_gpu_bindings_required')
    # Exact explicit UUID envelopes only. Docker's CDI driver is used on AI-HUB;
    # keep its null capability/options fields bound, rather than normalizing them.
    # Privileged/direct-device/all-device forms still need separate work.
    host = reviewed['HostConfig']
    nvidia = [{'Driver': 'nvidia', 'Count': 0,
        'DeviceIDs': [evidence['gpu_uuids'].get(r) for r in RESOURCES],
        'Capabilities': [['gpu']], 'Options': {}}]
    cdi = [{'Driver': 'cdi', 'Count': 0,
        'DeviceIDs': ['nvidia.com/gpu=' + evidence['gpu_uuids'][r] for r in RESOURCES],
        'Capabilities': None, 'Options': None}]
    require(host.get('Privileged') is False and not host.get('Devices') and
            host.get('DeviceRequests') in (nvidia, cdi), 'explicit_paired_gpu_exposure_required')


def validate_pending(manager, state, evidence):
    a, store = manager.a, manager.store
    installed = state['recovery_policy']
    require(state['ready'] is True and state['scope'] == evidence['scope'] == a.scope,
            'scope_changed')
    require(a.boot_reader() == evidence['current_boot'] and
            state.get('verified_boot') == evidence['previous_boot'], 'boot_changed')
    require(installed['policy_id'] == evidence['policy_id'] == digest(installed['policy']),
            'policy_changed')
    p = installed['policy']; wall = p['services']['wallpaper']
    require(manager.lifecycle is not None and manager.targets['wallpaper'].resources == RESOURCES and
            p['boot_startup'] == 'stopped_idle', 'paired_on_demand_policy_required')
    require(wall['identity_mode'] == 'exact' and wall['id'] == evidence['container_id'] == p['wallpaper'] and
            wall['image'] == evidence['image'] and wall['contract_sha256'] == evidence['old_contract_sha256'],
            'original_contract_binding_changed')
    require(p['gpu_uuids'] == evidence['gpu_uuids'] and set(p['gpu_uuids']) == set(RESOURCES) and
            state['lock_identity'] == evidence['lock_identity'], 'physical_binding_changed')
    a._check_physical_identities(state)
    require(set(state['owners']) == set(RESOURCES), 'exact_paired_ownership_required')
    expected = {k: evidence[k] for k in ('job_id', 'request_hash', 'lease_id', 'target')}
    expected['boot_id'] = evidence['previous_boot']
    require(evidence['target'] == 'wallpaper' and
            all(binding(o) == expected for o in state['owners'].values()), 'lease_changed')
    job = store.job(evidence['job_id'])
    require(job.get('request_id') == evidence['job_id'] and
            all(job.get(k) == expected[k] for k in ('request_hash', 'lease_id', 'target', 'boot_id')) and
            job.get('resources') == list(RESOURCES) and job.get('state') == 'failed' and
            job.get('error_code') == 'guest_reboot_interrupted' and not job.get('dual_gpu_session'),
            'interrupted_job_binding_changed')
    execution = job.get('execution_container', {})
    lifecycle = job.get('wallpaper_lifecycle', {})
    require(execution.get('id') == wall['id'] and
            execution.get('contract_sha256') == wall['contract_sha256'] and
            lifecycle.get('phase') == 'submit_intent' and
            lifecycle.get('identity') == dict(expected, resources=list(RESOURCES),
                policy_id=evidence['policy_id'], container_id=wall['id'], physical_gpu_uuids=p['gpu_uuids']),
            'verified_submission_binding_required')
    require(not any(j['state'] not in ('success', 'failed', 'cancelled') for j in store.jobs()),
            'nonterminal_jobs_present')
    require(refusal(state) == {
        'boot_recovery_status': 'operator_required',
        'boot_recovery_error': 'container_identity_or_supervision_changed',
        'boot_recovery_attempt_boot': evidence['current_boot']} and
        evidence['current_boot'] not in state.get('boot_startups', {}), 'original_boot_refusal_required')
    return job


def commit_transition(manager, state, transition, proof):
    """Called only inside finish()'s admission transaction with physical locks held."""
    evidence = transition['audit']['approval']
    validate_pending(manager, state, evidence)
    require(state['recovery_policy'] == transition['audit']['previous_recovery_policy'] and
            refusal(state) == transition['audit']['original_refusal'],
            'authority_changed_before_commit')
    remaining(transition['deadline'])
    require(manager.observer.container('wallpaper', transition['deadline']) ==
            transition['startup_record']['running_container'], 'container_changed_before_commit')
    require(manager.a.boot_reader() == proof['verified_boot'] == evidence['current_boot'] and
            proof.get('original_execution_gone') is True and proof.get('backend_stopped') is True,
            'fresh_stopped_boot_proof_required')
    # No externally supplied policy: only this reviewed hash may change.
    candidate = copy.deepcopy(state['recovery_policy']['policy'])
    candidate['services']['wallpaper']['contract_sha256'] = evidence['new_contract_sha256']
    require(candidate == manager.p and digest(candidate) == manager.policy_id,
            'only_wallpaper_contract_may_change')
    state['recovery_policy'] = {'policy': candidate, 'policy_id': manager.policy_id,
        'approval_ref': evidence['approval_ref'], 'authorized_at': timestamp()}
    state.setdefault('boot_startups', {})[evidence['current_boot']] = transition['startup_record']
    state.update(verified_boot=evidence['current_boot'], boot_recovery_status='ready', boot_readiness=proof)
    state.pop('boot_recovery_error', None)
    # Original attempt/error/policy survive in the same recovery audit as release.


def confirm_receipt(admission, approval_id):
    """A lost final directory-fsync response must not become a false durable ACK.

    Reread and flush the existing receipt under admission authority; no rewrite,
    rollback, backend control or second release. Other admission users cannot
    replace ownership.json between its file and directory flushes.
    """
    with admission.mutex.held():
        require(any(r.get('operator_contract_transition', {}).get('approval_id') == approval_id
                    for r in admission.snapshot().get('recoveries', [])), 'receipt_missing')
        try:
            fd = os.open(admission.store.path('ownership.json'), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if os.name != 'nt':
                fd = os.open(admission.store.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except OSError as exc:
            raise PersistenceUncertain() from exc


def recover_reboot_contract(admission, targets, backends, evidence, reviewed, *, observer=None):
    """Operator CLI only; excludes runner and other recovery, never controls Docker.

    Injected observer is for deterministic offline tests, like authorize_policy().
    """
    reviewed_approval(evidence, reviewed)
    store = admission.store
    with NativeLock(store.path('maintenance.lock')).held(0), \
            NativeLock(store.path('supervisor.lock')).held(0), \
            NativeLock(store.path('recovery.lock')).held(0):
        manager = RecoveryManager(admission, targets, backends)
        state = admission.snapshot()
        approval_id = digest(evidence)
        audits = [r for r in state.get('recoveries', []) if
                  r.get('operator_contract_transition', {}).get('approval_id') == approval_id]
        if audits:
            audit = audits[0]['operator_contract_transition']
            expected_policy = copy.deepcopy(audit['previous_recovery_policy']['policy'])
            expected_policy['services']['wallpaper']['contract_sha256'] = evidence['new_contract_sha256']
            require(len(audits) == 1 and
                    audits[0]['operator_contract_transition']['approval'] == evidence and
                    admission.boot_reader() == evidence['current_boot'] and
                    state.get('verified_boot') == evidence['current_boot'], 'completed_approval_changed')
            admission._check_physical_identities(state)
            require(state['lock_identity'] == evidence['lock_identity'] and
                    state['scope'] == evidence['scope'] and
                    manager.p == expected_policy and manager.policy_id == digest(expected_policy) and
                    not any(o['lease_id'] == evidence['lease_id'] for o in state['owners'].values()),
                    'completed_transition_binding_changed')
            confirm_receipt(admission, approval_id)
            return {'recorded': 'recover-reboot-contract', 'already_recorded': True,
                    'approval_id': approval_id, 'backend_action_performed': False}
        job = validate_pending(manager, state, evidence)
        previous_policy = copy.deepcopy(state['recovery_policy'])
        # Candidate authority remains process-local until the atomic final write.
        manager.p = copy.deepcopy(manager.p)
        manager.p['services']['wallpaper']['contract_sha256'] = evidence['new_contract_sha256']
        validate_policy(manager.p, admission, targets)
        manager.policy_id = digest(manager.p)
        manager.observer = observer or RecoveryObserver(manager.p, targets, backends)
        if observer is not None:
            require(observer.p == manager.p, 'test_observer_policy_changed')
        owner = state['owners']['gpu0']
        with manager.locks(RESOURCES, boot_owners=state['owners']):
            validate_pending(manager, admission.snapshot(), evidence)
            deadline = time.monotonic() + manager.p['observation_timeout']
            current = manager.observer.container('wallpaper', deadline)
            require(not current['running'] and current['pid'] == 0 and
                    current['started_at'] == job['execution_container']['started_at'],
                    'stopped_original_generation_required')
            # The genuine boot transition proves the old address space ended.
            # Also reject any surviving matching PID/start identity conservatively.
            manager.observer.gone(job.get('execution_processes', {}))
            record = {'phase': 'ready', 'identity': {'boot_id': evidence['current_boot'],
                'previous_verified_boot': evidence['previous_boot'], 'policy_id': manager.policy_id,
                'container': {'id': evidence['container_id'], 'contract_sha256': evidence['new_contract_sha256']},
                'physical_gpu_uuids': evidence['gpu_uuids']}, 'running_container': current,
                'created_at': timestamp(), 'operator_approval_id': approval_id}
            # Reuse boot readiness/generation checks without publishing a ready record.
            proof = manager.boot_readiness(RESOURCES, deadline, startup_record=record)
            manager.observer.gone(job.get('execution_processes', {}))
            record['proof'] = proof
            audit = {'approval_id': approval_id, 'approval': copy.deepcopy(evidence),
                     'previous_recovery_policy': previous_policy,
                     'previous_boot_readiness': copy.deepcopy(state.get('boot_readiness')),
                     'original_refusal': refusal(state)}
            transition = {'audit': audit, 'startup_record': record, 'deadline': deadline}
            incident = manager.incident(owner, RESOURCES, 'previous_boot')
            require(incident['phase'] in ('observing', 'verified'), 'incident_not_resumable')
            # finish() retains the failed job, adds proof, and atomically publishes
            # policy + verified boot + audit + both removals through shared authority.
            manager.finish(incident, owner, RESOURCES, proof, operator_transition=transition)
        return {'recorded': 'recover-reboot-contract', 'already_recorded': False,
                'approval_id': approval_id, 'backend_action_performed': False}
