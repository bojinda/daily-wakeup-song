"""Music GPU0 handoff proof under its original lease and cleanup deadline.

No scheduler, recovery action, unload transport, or independent locking authority.
Only the existing Comfy cleanup may send /free, once after durable intent.
"""
import copy
import csv
import io
import time

from .backends import CleanupPending, physical_gpu_uuid, remaining
from .core import FINAL, RecoveryRequired, timestamp
from .recovery import binding, digest, process_birth, valid_boot


# Exact literals only. Never format arbitrary exception text into public state.
_HARD_FAILURES = frozenset('''
music_release_scope_or_policy_invalid music_release_owner_boot_or_policy_changed
music_release_job_binding_changed music_release_context_changed
music_cleanup_intent_exists_no_replay music_release_execution_changed
music_release_wallpaper_not_stopped music_release_invalid_compute_capture
music_release_prompt_missing music_release_generation_changed
music_release_process_birth_changed music_cleanup_ack_without_unique_intent
music_release_deadline_changed music_release_free_ack_unverified
invalid_expected_queue_music unexplained_backend_activity_music
backend_readiness_unverified_music music_release_proof_missing_or_changed
music_release_no_longer_idle music_release_service_not_running
music_release_observation_scope_invalid music_release_unknown_or_reused_compute_process
service_process_identity_unverified ambiguous_service_process_identity
unexplained_gpu_process_or_memory gpu_process_observation_invalid
container_process_disappeared container_process_identity_unverified
container_process_stat_invalid container_main_process_unverified
container_identity_or_supervision_changed container_name_identity_changed
container_inspect_invalid container_mounts_invalid
container_mount_destination_invalid container_mount_destinations_ambiguous
physical_gpu_identity_unverified_gpu0 invalid_gpu_health_gpu0
gpu_health_requires_operator_gpu0 wallpaper_not_stopped_for_idle_cleanup
service_exceeds_calibrated_idle_baseline_ollama
service_exceeds_calibrated_idle_baseline_wallpaper
kernel_gpu_driver_fault_requires_operator kernel_health_log_requires_operator
kernel_health_boot_unverified kernel_health_log_boot_unverified
kernel_health_log_invalid kernel_health_log_unavailable
physical_lock_identity_changed physical_lock_identity_changed_during_acquisition
physical_lock_not_regular_file
'''.split())


def failure_code(exc):
    reason = exc.args[0] if isinstance(exc, RecoveryRequired) and len(exc.args) == 1 else None
    return reason if type(reason) is str and reason in _HARD_FAILURES else None


def result_released(job, state):
    """Read-only recognition of existing release records, never recovery.

    Local recover commits its flat audit BEFORE the CLI's terminal-failed write.
    Thus recovery can release a failed result, never substitute for a success
    receipt. Automatic recovery has a different, incident-bound audit schema.
    """
    try:
        context = job['music_cleanup_context']
        owner = {k: job[j] for k, j in (('job_id','request_id'), ('request_hash','request_hash'),
                 ('lease_id','lease_id'), ('target','target'), ('boot_id','boot_id'))}
        if (owner['target'] != 'music' or not valid_boot(owner['boot_id']) or
                any(not isinstance(v, str) or not v for v in owner.values()) or
                job['resources'] != ['gpu0'] or context['owner'] != owner or
                context['resources'] != ['gpu0']):
            return False
        policy_id, uuids = context['policy_id'], context['physical_gpu_uuids']
        proof = job.get('cleanup_proof', {}).get('music_release_idle')
        receipt = state.get('music_release_receipts', {}).get(owner['lease_id'], {})
        if (proof and job.get('music_release_idle') == proof and
                proof.get('schema') == 1 and proof.get('cleanup_verified') is True and
                proof.get('policy') == 'music_release_idle' and
                proof['owner'] == owner and proof['resources'] == ['gpu0'] and
                proof['policy_id'] == policy_id and proof['context_sha256'] == digest(context) and
                receipt.get('schema') == 1 and receipt.get('owner') == owner and
                receipt.get('resources') == ['gpu0'] and receipt.get('policy_id') == policy_id and
                receipt.get('proof_sha256') == digest(proof) and
                receipt.get('released_at') and
                receipt.get('final_observation', {}).get('context_sha256') == digest(context)):
            return True
        # Missing owner / ready:true / terminal success alone is never proof.
        if job['state'] != 'failed' or not job.get('error_code'):
            return False
        for audit in state.get('recoveries', []):
            if not isinstance(audit, dict) or not audit.get('recovered_at'):
                continue
            if audit.get('provenance') == 'runner_observations':
                identity = dict(owner, resources=['gpu0'], physical_gpu_uuids=uuids, policy_id=policy_id)
                incident_id = digest(identity)
                incident = state.get('automatic_recovery', {}).get(incident_id, {})
                recovery_proof = audit.get('proof', {})
                readiness = recovery_proof.get('readiness', recovery_proof)
                if (audit.get('identity') == identity and audit.get('incident_id') == incident_id and
                        job.get('recovery_incident') == incident_id and job.get('recovery_status') == 'verified' and
                        incident.get('id') == incident_id and incident.get('identity') == identity and
                        incident.get('kind') == 'previous_boot' and incident.get('phase') == 'released' and
                        incident.get('released_at') and valid_boot(audit.get('observed_boot')) and
                        audit['observed_boot'] != owner['boot_id'] and
                        incident.get('observed_boot') == audit['observed_boot'] and
                        recovery_proof == incident.get('proof') == job.get('recovery_proof') and
                        recovery_proof.get('cleanup_verified') is True and
                        recovery_proof.get('verified_boot') == audit['observed_boot'] and
                        readiness.get('cleanup_verified') is True and
                        (readiness.get('idle_ready') is True or readiness.get('backend_ready') is True)):
                    return True
            # Admission.recover stores the supplied flat local attestation; it
            # does not add runner provenance, incident, boot or resource fields.
            elif (job['error_code'] == 'operator_recovered_no_automatic_resubmission' and
                    audit.get('provenance') in (None, 'local_operator_attestation') and
                    'identity' not in audit and 'incident_id' not in audit and
                    all(audit.get(k) == owner[k] for k in ('job_id','request_hash','lease_id','target')) and
                    isinstance(audit.get('approval_ref'), str) and audit['approval_ref'].strip() and
                    all(audit.get(k) is True for k in ('operator_approved_backend_recovery',
                        'backend_quiescent','cleanup_verified')) and
                    all(k not in audit or audit[k] == v for k,v in
                        (('boot_id',owner['boot_id']), ('resources',['gpu0']),
                         ('policy_id',policy_id), ('physical_gpu_uuids',uuids)))):
                return True
        return False
    except (KeyError, TypeError, AttributeError, ValueError):
        return False  # Malformed evidence is never permission to publish.


def capture_execution(observer, deadline):
    identity = observer.container('music', deadline)
    if not identity['running'] or identity['pid'] <= 0:
        raise RecoveryRequired('music_release_service_not_running')
    processes = observer.processes(identity, deadline)
    if observer.container('music', deadline) != identity:
        raise RecoveryRequired('music_release_generation_changed')
    return {'identity': identity, 'processes': processes}


def music_guard(admission, target, owner, observer, record):
    if target.name != 'music' or target.kind != 'comfy':
        return None
    installed = admission.snapshot().get('recovery_policy')
    # Legacy standalone scopes have no accepted stopped-Wallpaper calibration.
    if not installed or 'stopped_idle_memory_mb' not in installed['policy']:
        return None
    return MusicReleaseGuard(admission, target, owner, observer, record)


class MusicReleaseGuard:
    def __init__(self, admission, target, owner, observer, record):
        self.a, self.target, self.owner = admission, target, copy.deepcopy(owner)
        self.observer, self.record = observer, record
        installed = admission.snapshot()['recovery_policy']
        self.p, self.policy_id = copy.deepcopy(installed['policy']), installed['policy_id']
        if (tuple(target.resources) != ('gpu0',) or target.lifecycle != 'always_on' or
                observer is None or observer.p != self.p or digest(self.p) != self.policy_id or
                self.p['scope'] != admission.scope or not valid_boot(owner.get('boot_id')) or
                not {'music', 'wallpaper'} <= set(self.p['services'])):
            raise RecoveryRequired('music_release_scope_or_policy_invalid')
        self.context = None
        self.deadline = None
        self.proof = None
        self.sequence = 0
        self.final_observation = None

    def bound(self, *, terminal=False):
        state = self.a.snapshot()
        self.a._check_physical_identities(state)
        job = self.a.store.job(self.owner['job_id'])
        if (state.get('ready') is not True or state.get('scope') != self.p['scope'] or
                state.get('verified_boot') != self.owner['boot_id'] or
                self.a.boot_reader() != self.owner['boot_id'] or
                state.get('recovery_policy', {}).get('policy_id') != self.policy_id or
                state.get('recovery_policy', {}).get('policy') != self.p or self.observer.p != self.p or
                binding(state['owners'].get('gpu0', {})) != binding(self.owner) or
                {r for r,o in state['owners'].items() if o.get('lease_id') == self.owner['lease_id']} != {'gpu0'}):
            raise RecoveryRequired('music_release_owner_boot_or_policy_changed')
        if (job.get('request_id') != self.owner['job_id'] or job.get('target') != 'music' or
                any(job.get(k) != self.owner[k] for k in ('request_hash', 'lease_id', 'boot_id')) or
                job.get('resources') != ['gpu0'] or job.get('host_stage') or
                job.get('completion_verified') is not True or
                job.get('state') not in (FINAL if terminal else ('cleanup_pending',))):
            raise RecoveryRequired('music_release_job_binding_changed')
        if self.context is not None and (
                job.get('music_cleanup_context') != self.context or
                job.get('backend_detail', {}).get('prompt_id') != self.context['prompt_id'] or
                job.get('music_execution') != self.context['execution']):
            raise RecoveryRequired('music_release_context_changed')
        return job

    def prepare(self, deadline):
        remaining(deadline, CleanupPending)
        job = self.bound()
        if any(k in job for k in ('music_cleanup_context', 'music_free_intent', 'music_free_ack')):
            raise RecoveryRequired('music_cleanup_intent_exists_no_replay')
        original = job.get('music_execution')
        if not original or capture_execution(self.observer, deadline) != original:
            raise RecoveryRequired('music_release_execution_changed')
        identities = {s:self.observer.container(s, deadline) for s in self.p['services']}
        if identities['wallpaper']['running'] or identities['wallpaper']['pid'] != 0:
            raise RecoveryRequired('music_release_wallpaper_not_stopped')
        # Bind other services only where they currently have GPU0 contexts.
        # A GPU1-only Ollama child may start/exit during this capture without
        # causing Music to wait for its process birth or idleness.
        raw = self.observer.command([self.p['nvidia_smi'],
            '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], deadline)
        gpu0_pids = set()
        for row in csv.reader(io.StringIO(raw)):
            if not row:
                continue
            if len(row) != 2 or not row[1].strip().isdecimal() or int(row[1]) <= 0:
                raise RecoveryRequired('music_release_invalid_compute_capture')
            if physical_gpu_uuid(row[0].strip()) == self.p['gpu_uuids']['gpu0']:
                pid = str(int(row[1]))
                if pid in gpu0_pids:
                    raise RecoveryRequired('music_release_invalid_compute_capture')
                gpu0_pids.add(pid)
        processes = {'music':self.observer.processes(identities['music'], deadline)}
        for service, ident in identities.items():
            if service == 'music':
                continue
            processes[service] = {}
            if ident['running']:
                raw = self.observer.command([self.p['docker'], 'top', ident['id'], '-eo', 'pid'], deadline)
                lines = raw.strip().splitlines()
                if not lines or lines[0].strip() != 'PID':
                    raise RecoveryRequired('service_process_identity_unverified')
                for line in lines[1:]:
                    pid = str(int(line.strip()))
                    if pid in gpu0_pids:
                        processes[service][pid] = process_birth(pid)
        if (identities['music'] != original['identity'] or processes['music'] != original['processes']):
            raise RecoveryRequired('music_release_execution_changed')
        prompt = job.get('backend_detail', {}).get('prompt_id')
        if not isinstance(prompt, str) or not prompt:
            raise RecoveryRequired('music_release_prompt_missing')
        self.deadline = deadline
        context = {'schema':1, 'owner':binding(self.owner), 'resources':['gpu0'],
                   'policy_id':self.policy_id, 'physical_gpu_uuids':{'gpu0':self.p['gpu_uuids']['gpu0']},
                   'prompt_id':prompt, 'execution':original, 'identities':identities,
                   'processes':processes, 'captured_at':timestamp(),
                   'cleanup_deadline_monotonic':deadline, 'cleanup_timeout_seconds':self.target.cleanup_timeout}
        self.record(music_cleanup_context=context)
        self.context = context
        self.bound()
        remaining(deadline, CleanupPending)

    def identities(self, deadline):
        for service in ('music', 'wallpaper'):
            if self.observer.container(service, deadline) != self.context['identities'][service]:
                raise RecoveryRequired('music_release_generation_changed')
        if self.observer.processes(self.context['identities']['music'], deadline) != self.context['processes']['music']:
            raise RecoveryRequired('music_release_process_birth_changed')

    def intent(self, deadline):
        self.bound()
        self.identities(deadline)
        if self.a.store.job(self.owner['job_id']).get('music_free_intent'):
            raise RecoveryRequired('music_cleanup_intent_exists_no_replay')
        remaining(deadline, CleanupPending)
        self.record(music_free_intent={'at':timestamp(), 'context_sha256':digest(self.context)})
        # A failed/ambiguous write never permits the effect.
        self.bound()
        remaining(deadline, CleanupPending)

    def acknowledged(self, deadline):
        self.bound()
        job = self.a.store.job(self.owner['job_id'])
        if not job.get('music_free_intent') or job.get('music_free_ack'):
            raise RecoveryRequired('music_cleanup_ack_without_unique_intent')
        self.record(music_free_ack={'at':timestamp(), 'context_sha256':digest(self.context)})
        remaining(deadline, CleanupPending)

    def observe(self, deadline, *, terminal=False, persist=True):
        if deadline != self.deadline:
            raise RecoveryRequired('music_release_deadline_changed')
        remaining(deadline, CleanupPending)
        job = self.bound(terminal=terminal)
        for key in ('music_free_intent', 'music_free_ack'):
            if job.get(key, {}).get('context_sha256') != digest(self.context):
                raise RecoveryRequired('music_release_free_ack_unverified')
        self.identities(deadline)
        # Only Music's queue is relevant to this GPU0 handoff. Ollama GPU1 is
        # neither polled nor unloaded. Same completed prompt may unwind; a new
        # prompt is a hard error even if memory is still settling.
        transport = self.observer.backends['music'].transport
        queue = transport.json('GET', '/queue', timeout=remaining(deadline))
        if any(not isinstance(queue.get(k), list) for k in ('queue_running','queue_pending')):
            raise RecoveryRequired('invalid_expected_queue_music')
        entries = queue['queue_running'] + queue['queue_pending']
        if any(not isinstance(e,list) or len(e)<2 or e[1]!=self.context['prompt_id'] for e in entries):
            raise RecoveryRequired('unexplained_backend_activity_music')
        stats = transport.json('GET', '/system_stats', timeout=remaining(deadline))
        if not isinstance(stats.get('system'),dict):
            raise RecoveryRequired('backend_readiness_unverified_music')
        sample = self.observer.gpu_sample(('gpu0',), deadline, stopped=True, music_context=self.context)
        self.identities(deadline)
        self.bound(terminal=terminal)
        remaining(deadline, CleanupPending)
        self.sequence += 1
        observation = {'at':timestamp(), 'sequence':self.sequence, 'queue_empty':not entries,
                       'sample':sample, 'context_sha256':digest(self.context)}
        if persist:
            self.record(music_cleanup_observation=observation)
        remaining(deadline, CleanupPending)
        return observation

    def verify(self, deadline):
        stable = []
        while True:
            observation = self.observe(deadline)
            if observation['queue_empty'] and not observation['sample']['music_settling']:
                stable.append(observation)
            else:
                stable = []
            if len(stable) >= max(3, self.p['samples']):
                proof = {'schema':1, 'cleanup_verified':True, 'policy':'music_release_idle',
                         'context_sha256':digest(self.context), 'owner':binding(self.owner),
                         'resources':['gpu0'], 'policy_id':self.policy_id, 'stable_samples':len(stable),
                         'observations':stable, 'verified_at':timestamp()}
                self.record(music_release_idle=proof)
                self.proof = proof
                self.bound()
                remaining(deadline, CleanupPending)
                return copy.deepcopy(proof)
            time.sleep(min(self.observer.poll_interval, remaining(deadline, CleanupPending)))

    def recheck(self, deadline, *, terminal=False):
        job = self.bound(terminal=terminal)
        if self.proof is None or job.get('music_release_idle') != self.proof:
            raise RecoveryRequired('music_release_proof_missing_or_changed')
        observation = self.observe(deadline, terminal=terminal)
        if not observation['queue_empty'] or observation['sample']['music_settling']:
            raise RecoveryRequired('music_release_no_longer_idle')
        self.record(music_release_commit_check=observation)
        self.bound(terminal=terminal)
        self.identities(deadline)
        remaining(deadline, CleanupPending)

    def observe_release(self, deadline):
        # Last full observation outside the shared ownership mutex. No waiting
        # or retry: rebound after certification retains ownership immediately.
        job = self.bound(terminal=True)
        if job.get('music_release_idle') != self.proof or not job.get('music_release_commit_check'):
            raise RecoveryRequired('music_release_proof_missing_or_changed')
        observation = self.observe(deadline, terminal=True, persist=False)
        if not observation['queue_empty'] or observation['sample']['music_settling']:
            raise RecoveryRequired('music_release_no_longer_idle')
        self.final_observation = observation

    def release_binding(self, deadline):
        # Called under the EXISTING ownership mutex. Local identity/journal
        # reads only; no Docker/HTTP/GPU probes, sleeps or jobs-lock inversion.
        job = self.bound(terminal=True)
        if (job.get('music_release_idle') != self.proof or self.final_observation is None or
                self.final_observation['context_sha256'] != digest(self.context)):
            raise RecoveryRequired('music_release_proof_missing_or_changed')
        for pid, birth in self.context['processes']['music'].items():
            if process_birth(pid) != birth:
                raise RecoveryRequired('music_release_process_birth_changed')
        remaining(deadline, CleanupPending)
        return {'schema':1, 'owner':binding(self.owner), 'resources':['gpu0'],
                'policy_id':self.policy_id, 'proof_sha256':digest(self.proof),
                'final_observation':self.final_observation, 'released_at':timestamp()}
