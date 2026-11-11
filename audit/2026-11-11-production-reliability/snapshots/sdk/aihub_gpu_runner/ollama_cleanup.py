"""Physical idle proof under an existing Ollama lease. No inference or control effects."""
import copy
import csv
import errno
import functools
import io
import math
import time

from .backends import CleanupPending, Uncertain, physical_gpu_uuid, remaining
from .core import RecoveryRequired, timestamp
from .recovery import (ProcessDisappeared, RecoveryObserver, binding, digest,
                       process_birth, valid_boot)


class _MemoryPending(Exception):
    """Only verified, owned unload memory may consume the bounded polling budget."""


class _ProcessPending(Exception):
    """Discard turnover/raced samples; process exit itself is never idle proof."""


class _UtilizationPending(Exception):
    """A complete safe observation is still above the accepted idle utilization."""


# Never persist arbitrary exception strings, paths, commands, response bodies or
# user data. These are fixed runner-owned reason codes, not a message regex.
_SAFE_REASONS = {
    'backend_deadline', 'ollama_cleanup_uncertain', 'ollama_unload_not_confirmed',
    'ollama_cleanup_utilization_pending',
    'ollama_cleanup_process_settling_pending', 'ollama_cleanup_process_turnover_limit',
    'ollama_cleanup_calibration_or_scope_unverified', 'ollama_cleanup_owner_boot_or_policy_changed',
    'ollama_cleanup_job_binding_unverified', 'ollama_cleanup_host_completion_unverified',
    'ollama_cleanup_completion_unverified', 'ollama_cleanup_on_demand_wallpaper_must_be_stopped',
    'ollama_cleanup_service_not_running', 'ollama_cleanup_main_process_unverified',
    'ollama_cleanup_generation_changed', 'ollama_cleanup_invalid_gpu_number',
    'ollama_cleanup_context_missing_or_changed', 'ollama_cleanup_ambiguous_process',
    'ollama_cleanup_new_or_reused_process', 'ollama_cleanup_invalid_compute_observation',
    'ollama_cleanup_unleased_gpu_activity', 'ollama_cleanup_unknown_or_duplicate_gpu_process',
    'ollama_cleanup_physical_gpu_unverified', 'ollama_cleanup_gpu_health_unverified',
    'ollama_cleanup_gpu_activity_unexplained', 'ollama_cleanup_process_birth_changed',
    'ollama_cleanup_unowned_memory_residency', 'ollama_cleanup_unverified_process_disappearance',
    'ollama_cleanup_no_longer_idle',
    'container_process_identity_unverified', 'container_process_stat_invalid',
    'container_main_process_unverified', 'container_identity_or_supervision_changed',
    'container_name_identity_changed', 'container_inspect_invalid',
    'kernel_gpu_driver_fault_requires_operator', 'kernel_health_log_requires_operator',
    'kernel_health_boot_unverified', 'kernel_health_log_boot_unverified',
    'kernel_health_log_invalid', 'kernel_health_log_unavailable',
    'physical_lock_identity_changed', 'recovery_control_or_observation_uncertain',
}
_SAFE_REASONS.update(prefix + s for s in ('ollama', 'music', 'wallpaper') for prefix in
    ('service_exceeds_calibrated_idle_baseline_', 'unexplained_backend_activity_',
     'invalid_expected_queue_', 'backend_readiness_unverified_'))
_PHASES = {'factory', 'capture', 'binding', 'container_identity', 'service_processes', 'context_persistence',
           'owned_unload', 'queues', 'driver_health', 'compute_apps', 'gpu_health',
           'process_birth_recheck', 'observation_persistence', 'limits', 'deadline'}


def cleanup_failure(exc, phase, *, service=None, pid=None):
    """Small sanitized diagnostic, suitable for the caller's durable job update."""
    phase = phase if phase in _PHASES else 'factory'
    reason = str(exc) if isinstance(exc, (RecoveryRequired, CleanupPending, Uncertain)) else None
    if reason not in _SAFE_REASONS:
        if isinstance(exc, ProcessDisappeared):
            reason = 'ollama_cleanup_unverified_process_disappearance'
        elif isinstance(exc, OSError) and phase in ('context_persistence', 'observation_persistence'):
            reason = 'ollama_cleanup_persistence_error'
        elif isinstance(exc, PermissionError):
            reason = 'ollama_cleanup_permission_denied'
        elif isinstance(exc, OSError):
            reason = 'ollama_cleanup_os_error'
        elif isinstance(exc, RecoveryRequired):
            reason = 'ollama_cleanup_recovery_required'
        elif isinstance(exc, (ValueError, IndexError, KeyError, UnicodeError)):
            reason = 'ollama_cleanup_invalid_observation'
        else:
            reason = 'ollama_cleanup_observation_error'
    number = getattr(exc, 'errno', None)
    pid = getattr(exc, 'pid', pid)
    return {'schema': 1, 'reason': reason, 'phase': phase,
            'exception_type': type(exc).__name__ if type(exc).__name__ in {
                'ProcessDisappeared', 'RecoveryRequired', 'CleanupPending', 'Uncertain',
                'FileNotFoundError', 'ProcessLookupError', 'PermissionError', 'OSError',
                'ValueError', 'IndexError', 'KeyError', 'UnicodeDecodeError', 'TypeError'} else 'Exception',
            'errno': number if type(number) is int and 0 < number < 4096 else None,
            'service': service if service in ('ollama', 'music', 'wallpaper') else None,
            'pid': str(pid) if str(pid).isdecimal() and 0 < int(pid) < 2**31 else None,
            'observed_at': timestamp()}


def _report_errors(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.report_failure(exc)
            raise
    return call


def cleanup_guard(admission, target, owner, backend, *, observer=None, record=None):
    # Legacy scopes have no accepted physical calibration. Dual sessions already
    # verify both physical GPUs during their separate restoration contract.
    if target.kind != 'ollama' or target.mode_switch is not None:
        return None
    installed = admission.snapshot().get('recovery_policy')
    if installed is None or 'ollama' not in installed['policy']['services']:
        return None
    if observer is None:
        observer = RecoveryObserver(installed['policy'], {target.name: target}, {target.name: backend})
    return OllamaCleanupGuard(admission, target, owner, observer, record=record)


class OllamaCleanupGuard:
    def __init__(self, admission, target, owner, observer, *, record=None, birth_reader=process_birth):
        self.a, self.target, self.owner, self.observer = admission, target, dict(owner), observer
        self.resources = tuple(target.resources)
        installed = admission.snapshot()['recovery_policy']
        self.p, self.policy_id = copy.deepcopy(installed['policy']), installed['policy_id']
        if ('ollama' not in self.p['services'] or self.p['scope'] != admission.scope or
                digest(self.p) != self.policy_id or observer.p != self.p or
                not valid_boot(self.owner.get('boot_id')) or
                not all(isinstance(self.owner.get(k), str) and self.owner[k] for k in
                        ('job_id', 'request_hash', 'lease_id', 'target')) or
                not self.resources or not set(self.resources) <= set(self.p['gpu_uuids'])):
            raise RecoveryRequired('ollama_cleanup_calibration_or_scope_unverified')
        self.record = record or (lambda **fields: None)
        self.birth_reader = birth_reader
        self.context = None
        self.exited = {}
        # Never add these children to the immutable pre-unload request context.
        # They are untrusted turnover: no observed GPU compute, and positive
        # disappearance is required before any fresh idle sample can count.
        self.settling = {}
        self.pending_exits = {}
        self.sample_exits = set()
        self.phase, self.service, self.pid = 'factory', None, None

    def mark_phase(self, phase, service=None, pid=None):
        self.phase, self.service, self.pid = phase, service, pid

    def failure(self, exc):
        return cleanup_failure(exc, self.phase, service=self.service, pid=self.pid)

    def report_failure(self, exc):
        detail = self.failure(exc)
        try:
            self.record(ollama_cleanup_failure=detail)
        except Exception:
            # A diagnostic write cannot override the original failure or make
            # an uncertain journal safe. The outer caller also tries its update.
            pass
        return detail

    def bound(self):
        state = self.a.snapshot()
        self.a._check_physical_identities(state)
        installed = state.get('recovery_policy', {})
        if (state.get('ready') is not True or state.get('scope') != self.p['scope'] or installed.get('policy_id') != self.policy_id or
                self.observer.p != self.p or self.owner['target'] != self.target.name or
                installed.get('policy') != self.p or digest(self.p) != self.policy_id or
                self.a.boot_reader() != self.owner.get('boot_id') or
                state.get('verified_boot') != self.owner.get('boot_id') or
                any(binding(state['owners'].get(r, {})) != binding(self.owner) for r in self.resources) or
                {r for r, o in state['owners'].items() if o['lease_id'] == self.owner['lease_id']} != set(self.resources)):
            raise RecoveryRequired('ollama_cleanup_owner_boot_or_policy_changed')
        job = self.a.store.job(self.owner['job_id'])
        if (any(job.get(k) != self.owner[k] for k in ('request_hash', 'lease_id', 'boot_id')) or
                job.get('request_id') != self.owner['job_id'] or job.get('target') != self.target.name or
                tuple(job.get('resources', ())) != self.resources or
                job.get('state') not in (('success', 'failed') if getattr(self, 'terminal', False) else ('cleanup_pending',))):
            raise RecoveryRequired('ollama_cleanup_job_binding_unverified')
        if job.get('host_stage'):
            requests = job.get('host_requests', [])
            if (type(job.get('host_exit_code')) is not int or job['host_exit_code'] < 0 or
                    any(q.get('state') != 'completed' for q in requests)):
                raise RecoveryRequired('ollama_cleanup_host_completion_unverified')
            return bool(requests)
        if job.get('completion_verified') is not True:
            raise RecoveryRequired('ollama_cleanup_completion_unverified')
        return True

    def identities(self, deadline):
        return {s: self.observer.container(s, deadline) for s in self.p['services']}

    @_report_errors
    def prepare(self, deadline):
        self.mark_phase('capture')
        remaining(deadline, CleanupPending)
        owned_work = self.bound()
        identities = self.identities(deadline)
        wallpaper = identities['wallpaper']
        if 'stopped_idle_memory_mb' in self.p and (wallpaper['running'] or wallpaper['pid'] != 0):
            raise RecoveryRequired('ollama_cleanup_on_demand_wallpaper_must_be_stopped')
        ident = identities['ollama']
        if not ident['running'] or ident['pid'] <= 0:
            raise RecoveryRequired('ollama_cleanup_service_not_running')
        self.mark_phase('service_processes', 'ollama')
        processes = self.observer.processes(ident, deadline)
        if str(ident['pid']) not in processes:
            raise RecoveryRequired('ollama_cleanup_main_process_unverified')
        if self.identities(deadline) != identities:
            raise RecoveryRequired('ollama_cleanup_generation_changed')
        self.context = {'binding': binding(self.owner), 'policy_id': self.policy_id,
                        'resources': list(self.resources), 'identities': identities,
                        'ollama_processes': processes, 'owned_work_completed': owned_work,
                        'captured_at': timestamp()}
        self.mark_phase('context_persistence')
        self.record(ollama_cleanup_context=self.context)
        remaining(deadline, CleanupPending)

    @staticmethod
    def number(value):
        try:
            result = float(value)
        except (ValueError, TypeError):
            raise RecoveryRequired('ollama_cleanup_invalid_gpu_number') from None
        if not math.isfinite(result) or result < 0:
            raise RecoveryRequired('ollama_cleanup_invalid_gpu_number')
        return result

    def absent(self, pid):
        try:
            birth = self.birth_reader(pid)
        except ProcessDisappeared as exc:
            if exc.pid != pid or exc.errno not in (errno.ENOENT, errno.ESRCH):
                raise RecoveryRequired('ollama_cleanup_unverified_process_disappearance') from None
            return
        expected = self.context['ollama_processes'].get(pid, self.settling.get(pid))
        if birth != expected:
            raise RecoveryRequired('ollama_cleanup_process_birth_changed')
        raise RecoveryRequired('ollama_cleanup_unverified_process_disappearance')

    def expected_exit(self, pid):
        ident = self.context['identities']['ollama']
        expected = self.context['ollama_processes'].get(pid, self.settling.get(pid))
        if (not self.context['owned_work_completed'] or pid == str(ident['pid']) or
                expected is None):
            raise RecoveryRequired('ollama_cleanup_unverified_process_disappearance')
        self.absent(pid)  # Positive ENOENT/ESRCH for this captured child only.
        self.exited[pid] = expected
        self.sample_exits.add(pid)
        return True

    def settling_exit(self, pid):
        """Docker omission is not exit proof for a quarantined noncompute child.

        Only an already tracked post-unload lifetime may wait. Keep it untrusted
        and continue the full compute/health/queue checks in this observation;
        it cannot count toward idle stability until positive disappearance.
        Captured model workers and on_missing confirmation remain strict.
        """
        try:
            birth = self.birth_reader(pid)
        except ProcessDisappeared as exc:
            if exc.pid != pid or exc.errno not in (errno.ENOENT, errno.ESRCH):
                raise RecoveryRequired('ollama_cleanup_unverified_process_disappearance') from None
            self.expected_exit(pid)  # Independent positive confirmation; no PID reuse.
            return
        if birth != self.settling[pid]:
            raise RecoveryRequired('ollama_cleanup_process_birth_changed')
        self.pending_exits[pid] = birth
        self.sample_turnover = True

    def track_processes(self, processes, owned_work):
        """Quarantine new same-container lifetimes; never infer request origin."""
        for pid, birth in processes.items():
            self.mark_phase('service_processes', 'ollama', pid)
            expected = self.context['ollama_processes'].get(pid, self.settling.get(pid))
            if pid in self.exited or (expected is not None and birth != expected):
                raise RecoveryRequired('ollama_cleanup_new_or_reused_process')
            if expected is None:
                if not owned_work:
                    raise RecoveryRequired('ollama_cleanup_new_or_reused_process')
                if len(self.settling) >= 64:
                    raise RecoveryRequired('ollama_cleanup_process_turnover_limit')
                self.settling[pid] = birth
                self.sample_turnover = True
        for pid in self.settling:
            if pid not in processes and pid not in self.exited:
                self.mark_phase('process_birth_recheck', 'ollama', pid)
                self.settling_exit(pid)
        for pid in self.context['ollama_processes']:
            if pid not in processes and pid not in self.exited:
                self.mark_phase('process_birth_recheck', 'ollama', pid)
                self.expected_exit(pid)
        if set(processes).intersection(self.settling):
            self.sample_turnover = True

    def sample(self, deadline):
        self.sample_exits = set()
        self.sample_turnover = False
        self.pending_exits = {}
        self.mark_phase('binding')
        owned_work = self.bound()
        if not self.context or owned_work != self.context['owned_work_completed']:
            raise RecoveryRequired('ollama_cleanup_context_missing_or_changed')
        self.mark_phase('container_identity')
        identities = self.identities(deadline)
        if identities != self.context['identities']:
            raise RecoveryRequired('ollama_cleanup_generation_changed')
        # This requires empty residency and valid backend readiness. Only targets
        # intersecting the owned resources are checked; GPU0 Music can keep working
        # while a routine GPU1 lease converges. Stopped wallpaper is not contacted.
        self.mark_phase('queues')
        self.observer.queues(self.resources, deadline, skip_wallpaper=not identities['wallpaper']['running'])
        self.mark_phase('driver_health')
        driver = self.observer.driver_health(deadline)
        for pid in self.exited:
            self.mark_phase('process_birth_recheck', 'ollama', pid)
            self.absent(pid)
        known = {}
        ollama_processes = None
        for service, ident in identities.items():
            self.mark_phase('service_processes', service)
            # Other-service helpers on GPU0 are not a prerequisite for GPU1
            # cleanup. Enumerate their PIDs, but birth-bind only compute rows
            # on our leased UUIDs below. Unknown GPU1 compute remains fatal.
            if service == 'ollama':
                processes = self.observer.processes(ident, deadline, on_missing=self.expected_exit)
            else:
                processes = self.observer.processes(ident, deadline, pids_only=True)
            if service == 'ollama':
                self.track_processes(processes, owned_work)
                ollama_processes = processes
            for pid, birth in processes.items():
                if pid in known:
                    raise RecoveryRequired('ollama_cleanup_ambiguous_process')
                known[pid] = (service, birth)
        uuid_resources = {self.p['gpu_uuids'][r]: r for r in self.resources}
        usage = {s: {r: 0 for r in self.resources} for s in identities}
        seen, observed_processes = set(), {}
        foreign_compute = False
        compute_pids = set()
        self.mark_phase('compute_apps')
        raw = self.observer.command([self.p['nvidia_smi'],
            '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], deadline)
        for row in csv.reader(io.StringIO(raw)):
            self.mark_phase('compute_apps')
            if not row:
                continue
            if len(row) != 3:
                raise RecoveryRequired('ollama_cleanup_invalid_compute_observation')
            uuid = physical_gpu_uuid(row[0].strip())
            pid = str(int(row[1].strip()))
            compute_pids.add(pid)
            if pid in self.settling:
                self.mark_phase('compute_apps', 'ollama', pid)
                raise RecoveryRequired('ollama_cleanup_new_or_reused_process')
            resource = uuid_resources.get(uuid)
            if resource is None:
                if pid in self.context['ollama_processes'] or pid in known and known[pid][0] == 'ollama':
                    raise RecoveryRequired('ollama_cleanup_unleased_gpu_activity')
                continue
            if (uuid, pid) in seen:
                raise RecoveryRequired('ollama_cleanup_unknown_or_duplicate_gpu_process')
            seen.add((uuid, pid))
            memory = self.number(row[2])
            if pid not in known:
                if pid in self.context['ollama_processes']:
                    self.mark_phase('process_birth_recheck', 'ollama', pid)
                    self.expected_exit(pid)
                    continue  # Stale compute row: discard, never count this sample.
                raise RecoveryRequired('ollama_cleanup_unknown_or_duplicate_gpu_process')
            service, birth = known[pid]
            if service != 'ollama':
                birth = self.birth_reader(pid)
                known[pid] = (service, birth)
            foreign_compute |= service != 'ollama'
            if service == 'ollama' and self.context['ollama_processes'].get(pid) != birth:
                raise RecoveryRequired('ollama_cleanup_new_or_reused_process')
            observed_processes[pid] = birth
            usage[service][resource] += memory
        gpus = {}
        self.mark_phase('gpu_health')
        for resource, uuid in ((r, self.p['gpu_uuids'][r]) for r in self.resources):
            raw = self.observer.command([self.p['nvidia_smi'], '--id=' + uuid,
                '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total,retired_pages.pending',
                '--format=csv,noheader,nounits'], deadline)
            rows = list(csv.reader(io.StringIO(raw)))
            if len(rows) != 1 or len(rows[0]) != 6 or physical_gpu_uuid(rows[0][0].strip()) != uuid:
                raise RecoveryRequired('ollama_cleanup_physical_gpu_unverified')
            _, memory, utilization, temperature, ecc, retired = [x.strip() for x in rows[0]]
            memory, utilization, temperature = map(self.number, (memory, utilization, temperature))
            if (utilization > 100 or temperature > self.p['max_temperature_c'] or ecc not in ('0', '[N/A]', 'N/A') or
                    retired not in ('No', 'no', '0', '[N/A]', 'N/A')):
                raise RecoveryRequired('ollama_cleanup_gpu_health_unverified')
            gpus[resource] = {'uuid': uuid, 'memory_mb': memory, 'utilization_pct': utilization,
                              'temperature_c': temperature}
        for pid, birth in observed_processes.items():
            self.mark_phase('process_birth_recheck', known[pid][0], pid)
            try:
                current = self.birth_reader(pid)
            except ProcessDisappeared as exc:
                if known[pid][0] != 'ollama' or exc.pid != pid:
                    raise
                self.expected_exit(pid)
                continue
            if current != birth:
                raise RecoveryRequired('ollama_cleanup_process_birth_changed')
            if known[pid][0] != 'ollama':
                # Deferring unrelated birth reads must not attribute a reused
                # foreign PID to a stale Docker census. Confirm membership
                # again around the leased-GPU lifetime, without reading the
                # births of independent helpers on the other physical GPU.
                service = known[pid][0]
                if (pid not in self.observer.processes(identities[service], deadline, pids_only=True) or
                        self.birth_reader(pid) != birth):
                    raise RecoveryRequired('ollama_cleanup_process_birth_changed')
        for pid in self.exited:
            self.mark_phase('process_birth_recheck', 'ollama', pid)
            self.absent(pid)
        self.mark_phase('container_identity')
        if self.identities(deadline) != identities:
            raise RecoveryRequired('ollama_cleanup_generation_changed')
        # Re-enumerate before counting a sample. A child arriving during GPU
        # observations invalidates it, including a newly identified compute PID.
        self.mark_phase('service_processes', 'ollama')
        final_processes = self.observer.processes(identities['ollama'], deadline,
                                                 on_missing=self.expected_exit)
        self.track_processes(final_processes, owned_work)
        unexpected_compute = set(self.settling).intersection(compute_pids)
        if unexpected_compute:
            self.mark_phase('compute_apps', 'ollama', min(unexpected_compute, key=int))
            raise RecoveryRequired('ollama_cleanup_new_or_reused_process')
        if final_processes != ollama_processes:
            self.sample_turnover = True
        if self.settling:
            self.mark_phase('queues')
            self.observer.queues(self.resources, deadline, skip_wallpaper=not identities['wallpaper']['running'])
        self.mark_phase('container_identity')
        if self.identities(deadline) != identities:
            raise RecoveryRequired('ollama_cleanup_generation_changed')
        self.bound()
        observation = {'observed_at': timestamp(), 'physical_gpus': gpus,
                       'service_memory_mb': usage, 'driver_health': driver,
                       'exited_ollama_processes': dict(self.exited),
                       'process_snapshot_discarded': bool(self.sample_exits or self.sample_turnover)}
        if self.settling:
            observation['settling_ollama_processes'] = dict(self.settling)
        if self.pending_exits:
            observation['pending_ollama_process_exits'] = dict(self.pending_exits)
        # Persist measured values before classifying a breach. Storage failure
        # propagates; it cannot be converted into cleanup success or a retry.
        self.mark_phase('observation_persistence')
        self.record(ollama_cleanup_observation=observation)
        self.mark_phase('limits')
        pending = False
        for service, values in usage.items():
            if any(v > self.p['services'][service]['idle_memory_mb'][r] for r, v in values.items()):
                if service != 'ollama' or not owned_work:
                    raise RecoveryRequired('service_exceeds_calibrated_idle_baseline_' + service)
                pending = True
        stopped = not identities['wallpaper']['running'] and identities['wallpaper']['pid'] == 0
        ceilings = self.p['stopped_idle_memory_mb'] if stopped and 'stopped_idle_memory_mb' in self.p else self.p['idle_memory_mb']
        if any(g['memory_mb'] > ceilings[r] for r, g in gpus.items()):
            if not owned_work:
                raise RecoveryRequired('ollama_cleanup_unowned_memory_residency')
            pending = True
        # Only after every identity/queue/driver/health check and the durable
        # observation write may normal post-unload utilization consume the
        # existing budget. Foreign compute on an owned GPU remains unexplained;
        # independent GPU0 Music is outside this GPU1 observation as before.
        if any(g['utilization_pct'] > self.p['idle_utilization_pct'] for g in gpus.values()):
            if not owned_work or foreign_compute:
                raise RecoveryRequired('ollama_cleanup_gpu_activity_unexplained')
            raise _UtilizationPending()
        remaining(deadline, CleanupPending)
        if self.sample_exits or self.sample_turnover:
            raise _ProcessPending()
        if pending:
            raise _MemoryPending()
        return observation

    @_report_errors
    def verify(self, deadline):
        stable = 0
        pending = None
        try:
            while True:
                self.mark_phase('deadline')
                remaining(deadline, CleanupPending)
                delay = self.observer.poll_interval
                try:
                    observation = self.sample(deadline)
                except _UtilizationPending:
                    stable = 0
                    pending = 'ollama_cleanup_utilization_pending'
                    delay = 1.0
                except (_MemoryPending, _ProcessPending) as exc:
                    stable = 0
                    pending = ('ollama_cleanup_process_settling_pending'
                               if isinstance(exc, _ProcessPending) and self.settling else None)
                else:
                    stable += 1
                    required = max(3, self.p['samples']) if self.settling else self.p['samples']
                    if stable >= required:
                        remaining(deadline, CleanupPending)
                        return {'policy_id': self.policy_id, 'binding': binding(self.owner),
                                'stable_samples': stable, 'observation': observation,
                                'policy': 'owned_unload_then_calibrated_physical_idle'}
                self.mark_phase('deadline')
                time.sleep(min(delay, remaining(deadline, CleanupPending)))
        except CleanupPending as exc:
            if pending and exc.code == 'backend_deadline':
                raise CleanupPending(pending) from None
            raise

    @_report_errors
    def commit_check(self, deadline, *, terminal=False):
        """One fresh observation under the original budget; no settling/replay."""
        self.terminal = terminal
        try:
            remaining(deadline, CleanupPending)
            self.sample(deadline)
            remaining(deadline, CleanupPending)
        except (_MemoryPending, _ProcessPending, _UtilizationPending):
            raise RecoveryRequired('ollama_cleanup_no_longer_idle') from None
        finally:
            self.terminal = False

    def release_binding(self, deadline):
        """Local-only last checks under the existing ownership mutex."""
        self.terminal = True
        try:
            self.bound()
            remaining(deadline, CleanupPending)
        finally:
            self.terminal = False
