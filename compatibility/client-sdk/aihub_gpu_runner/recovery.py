"""Opt-in, journaled recovery. No inference, global Docker control or lock deletion.

Action intents precede effects. Ambiguous intents are observed, never reissued.
Ownership removal and its unique audit entry share one atomic journal write.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import copy
import errno
import hashlib
import io
import math
import os
from pathlib import Path
import posixpath
import re
import subprocess
import threading
import time
from uuid import UUID

from .backends import CleanupPending, Uncertain, physical_gpu_uuid, remaining
from .core import (FINAL, NativeLock, RecoveryRequired, encode, guest_boot_id,
                   lock_identity, strict_json, timestamp)


class PersistenceUncertain(Exception):
    """Do not overwrite an outcome-uncertain durable write with a stale incident."""


class ProcessDisappeared(RecoveryRequired):
    """One exact /proc PID read returned ENOENT/ESRCH; not permission to skip it."""
    def __init__(self, pid, error_number):
        super().__init__('container_process_disappeared')
        self.pid, self.errno = str(pid), error_number


def process_birth(pid):
    """Read a numeric kernel start time, retaining exact missing-PID evidence."""
    pid = str(pid)
    if not pid.isdecimal() or int(pid) <= 0:
        raise RecoveryRequired('container_process_identity_unverified')
    try:
        raw = Path('/proc/' + pid + '/stat').read_text()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            raise ProcessDisappeared(pid, exc.errno) from None
        raise
    try:
        birth = raw.rsplit(')', 1)[1].split()[19]
        if not raw.startswith(pid + ' (') or not birth.isdecimal():
            raise ValueError()
    except (IndexError, ValueError):
        raise RecoveryRequired('container_process_stat_invalid') from None
    return birth


def valid_boot(value):
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def seconds(value):
    return datetime.fromisoformat(value).timestamp()


def now():
    return datetime.now(timezone.utc).timestamp()


def on_demand(targets):
    return getattr(targets.get('wallpaper'), 'lifecycle', 'always_on') == 'on_demand'


def validate_policy(policy, admission, targets):
    keys = {'scope', 'docker', 'nvidia_smi', 'journalctl', 'wallpaper', 'gpu_uuids', 'services', 'boot_startup',
            'idle_memory_mb', 'max_temperature_c', 'idle_utilization_pct',
            'samples', 'observation_timeout', 'grace_seconds', 'allow_force',
            'max_incidents', 'window_seconds', 'exclusive_wallpaper_access'}
    if on_demand(targets):
        keys.add('stopped_idle_memory_mb')
    if not isinstance(policy, dict) or set(policy) != keys or policy['scope'] != admission.scope:
        raise ValueError('recovery_policy_scope_or_shape_invalid')
    for key in ('docker', 'nvidia_smi', 'journalctl'):
        if not isinstance(policy[key], str) or not Path(policy[key]).is_absolute():
            raise ValueError('absolute_recovery_control_paths_required')
    if (policy['boot_startup'] != ('stopped_idle' if on_demand(targets) else 'verified_reboot_once') or
            type(policy['allow_force']) is not bool or policy['exclusive_wallpaper_access'] is not True or
            type(policy['samples']) is not int or not 2 <= policy['samples'] <= 20 or
            type(policy['max_incidents']) is not int or not 1 <= policy['max_incidents'] <= 10):
        raise ValueError('bounded_recovery_policy_required')
    for key, low, high in (('observation_timeout', 1, 120), ('grace_seconds', 1, 60),
                          ('window_seconds', 60, 604800), ('max_temperature_c', 1, 100),
                          ('idle_utilization_pct', 0, 10)):
        v = policy[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('invalid_recovery_limit_' + key)
    for key in ('gpu_uuids', 'idle_memory_mb'):
        if not isinstance(policy[key], dict) or set(policy[key]) != set(admission.lock_paths):
            raise ValueError('all_physical_gpu_calibrations_required')
    uuids = {r: physical_gpu_uuid(u) for r, u in policy['gpu_uuids'].items()}
    if len(set(uuids.values())) != len(uuids):
        raise ValueError('distinct_recovery_physical_gpus_required')
    def ceilings(values):
        if not isinstance(values, dict) or set(values) != set(uuids) or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('invalid_service_idle_calibration')
    ceilings(policy['idle_memory_mb'])
    if on_demand(targets):
        ceilings(policy['stopped_idle_memory_mb'])
        if targets['wallpaper'].shutdown_timeout <= policy['grace_seconds']:
            raise ValueError('shutdown_budget_must_include_grace_and_cleanup')
    services = policy['services']
    if not isinstance(services, dict) or 'wallpaper' not in services:
        raise ValueError('dedicated_wallpaper_binding_required')
    for service, binding in services.items():
        if (not isinstance(binding, dict) or set(binding) != {
                'name', 'id', 'image', 'contract_sha256', 'idle_memory_mb', 'identity_mode'} or
                binding['identity_mode'] not in ('exact', 'contract') or
                not re.fullmatch(r'[a-f0-9]{64}', binding['id']) or
                not re.fullmatch(r'sha256:[a-f0-9]{64}', binding['image']) or
                not re.fullmatch(r'[a-f0-9]{64}', binding['contract_sha256']) or
                not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', binding['name'])):
            raise ValueError('immutable_service_binding_required')
        ceilings(binding['idle_memory_mb'])
    wall = targets.get('wallpaper')
    if (policy['wallpaper'] != services['wallpaper']['id'] or
            services['wallpaper']['identity_mode'] != 'exact' or
            services['wallpaper']['name'] != 'comfyui-wallpaper' or wall is None or wall.kind != 'comfy'):
        raise ValueError('only_dedicated_wallpaper_recovery_permitted')
    if len({s['id'] for s in services.values()}) != len(services):
        raise ValueError('distinct_service_identities_required')
    if wall.conditioning_idle_memory_mb is not None and wall.conditioning_gpu_uuids() != uuids:
        raise ValueError('wallpaper_recovery_uuid_contract_mismatch')
    if any(t.mode_switch is not None and t.mode_switch['gpu_uuids'] != uuids for t in targets.values()):
        raise ValueError('meeting_recovery_uuid_contract_mismatch')


class RecoveryObserver:
    """Bounded local probes and exact-container actions, injectable for offline tests."""
    def __init__(self, policy, targets, backends, run=subprocess.run, poll_interval=.1):
        self.p, self.targets, self.backends = policy, targets, backends
        self.run, self.poll_interval = run, poll_interval

    def command(self, argv, deadline):
        try:
            result = self.run(argv, text=True, capture_output=True, check=False,
                              timeout=remaining(deadline))
            if result.returncode != 0:
                raise Uncertain('recovery_control_or_observation_failed')
            return result.stdout
        except (OSError, subprocess.SubprocessError, Uncertain):
            raise Uncertain('recovery_control_or_observation_uncertain') from None

    @staticmethod
    def contract(meta):
        # Exclude runtime timestamps/PIDs/status; bind all executable configuration.
        if not isinstance(meta.get('Mounts'), list):
            raise RecoveryRequired('container_mounts_invalid')
        result = copy.deepcopy({k: meta[k] for k in ('Name', 'Image', 'Config', 'HostConfig', 'Mounts')})
        destinations = set()
        for mount in result['Mounts']:
            destination = mount.get('Destination') if isinstance(mount, dict) else None
            if not isinstance(destination, str) or not destination.startswith('/') or '\x00' in destination:
                raise RecoveryRequired('container_mount_destination_invalid')
            # Detect aliases without rewriting any original path/attribute.
            key = posixpath.normpath('/' + destination.lstrip('/'))
            if key in destinations:
                raise RecoveryRequired('container_mount_destinations_ambiguous')
            destinations.add(key)
        # Docker inspect may enumerate the same mounts in any order. Sort only
        # the deep copy; every record and every original attribute stays bound.
        result['Mounts'].sort(key=lambda mount: mount['Destination'])
        # Docker's default hostname changes when an otherwise identical service
        # is legitimately recreated (e.g. dual-meeting restoration). Explicit
        # hostnames remain bound. Wallpaper still requires the exact container ID.
        if result['Config'].get('Hostname') == meta['Id'][:12]:
            result['Config'].pop('Hostname')
        return digest(result)

    def container(self, service, deadline):
        b = self.p['services'][service]
        selector = b['id'] if b['identity_mode'] == 'exact' else b['name']
        rows = strict_json(self.command([self.p['docker'], 'inspect', selector], deadline))
        if not isinstance(rows, list) or len(rows) != 1:
            raise RecoveryRequired('container_inspect_invalid')
        m = rows[0]
        if ((b['identity_mode'] == 'exact' and m.get('Id') != b['id']) or
                not re.fullmatch(r'[a-f0-9]{64}', m.get('Id', '')) or
                m.get('Name') != '/' + b['name'] or m.get('Image') != b['image'] or
                self.contract(m) != b['contract_sha256'] or m['State'].get('Restarting') or
                m['State'].get('Paused') or m['State'].get('Dead') or
                (service == 'wallpaper' and m['HostConfig'].get('RestartPolicy', {}).get('Name') not in ('no', ''))):
            raise RecoveryRequired('container_identity_or_supervision_changed')
        # Independent name lookup prevents controlling a renamed/replaced container.
        named = strict_json(self.command([self.p['docker'], 'inspect', b['name']], deadline))
        if len(named) != 1 or named[0].get('Id') != m['Id']:
            raise RecoveryRequired('container_name_identity_changed')
        return {'id': m['Id'], 'started_at': m['State']['StartedAt'],
                'running': m['State']['Running'], 'pid': m['State']['Pid'],
                'contract_sha256': b['contract_sha256']}

    def signal(self, identity, signal, deadline):
        current = self.container('wallpaper', deadline)
        if current['started_at'] != identity['started_at'] or current['id'] != identity['id']:
            raise RecoveryRequired('container_generation_changed_before_signal')
        if not current['running'] and current['pid'] == 0:
            return
        self.command([self.p['docker'], 'kill', '--signal=' + signal, identity['id']], deadline)

    def start(self, identity, deadline):
        current = self.container('wallpaper', deadline)
        if (current['id'] != identity['id'] or current['running'] or current['pid'] != 0 or
                current['started_at'] != identity['started_at']):
            raise RecoveryRequired('container_changed_before_start')
        self.command([self.p['docker'], 'start', identity['id']], deadline)

    def placement(self, identity, resources, deadline):
        if self.container('wallpaper', deadline) != identity:
            raise RecoveryRequired('wallpaper_generation_changed_before_placement')
        raw = self.command([self.p['docker'], 'exec', identity['id'], 'nvidia-smi',
                            '--query-gpu=index,uuid', '--format=csv,noheader'], deadline)
        rows = list(csv.reader(io.StringIO(raw)))
        expected = [(str(i), self.p['gpu_uuids'][r]) for i, r in enumerate(resources)]
        if [(row[0].strip(), physical_gpu_uuid(row[1].strip())) for row in rows if len(row) == 2] != expected or len(rows) != len(expected):
            raise RecoveryRequired('wallpaper_physical_placement_unverified')
        if self.container('wallpaper', deadline) != identity:
            raise RecoveryRequired('wallpaper_generation_changed_during_placement')
        return {'physical_gpu_uuids': {r: self.p['gpu_uuids'][r] for r in resources}}

    def processes(self, identity, deadline, *, on_missing=None, pids_only=False):
        if not identity['running']:
            return {}
        raw = self.command([self.p['docker'], 'top', identity['id'], '-eo', 'pid'], deadline)
        lines = raw.strip().splitlines()
        if not lines or lines[0].strip() != 'PID':
            raise RecoveryRequired('container_process_identity_unverified')
        pids = {}
        listed = set()
        for line in lines[1:]:
            try:
                pid = str(int(line.strip()))
            except ValueError:
                raise RecoveryRequired('container_process_identity_unverified') from None
            if int(pid) <= 0 or pid in listed:
                raise RecoveryRequired('container_process_identity_unverified')
            listed.add(pid)
            if pids_only:
                pids[pid] = None
                continue
            try:
                pids[pid] = process_birth(pid)
            except ProcessDisappeared:
                # Strict by default for all existing recovery/lifecycle callers.
                # Only the cleanup guard may supply a bound child-exit verifier.
                if pid == str(identity['pid']) or on_missing is None or on_missing(pid) is not True:
                    raise
        if str(identity['pid']) not in pids:
            raise RecoveryRequired('container_main_process_unverified')
        return pids

    def gone(self, processes):
        for pid, started in processes.items():
            try:
                current = Path('/proc/' + pid + '/stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                continue
            if current == started:
                raise RecoveryRequired('original_container_process_survives')

    def queues(self, resources, deadline, prompt=None, teardown=False, skip_wallpaper=False):
        observations = {}
        for name, target in self.targets.items():
            if skip_wallpaper and name == 'wallpaper':
                continue
            if not set(target.resources).intersection(resources) or target.mode_switch is not None:
                continue
            backend = self.backends[name]
            try:
                if target.kind == 'comfy':
                    timeout = remaining(deadline)
                    if teardown and name == 'wallpaper':
                        # A wedged owned endpoint must not exhaust the Docker/
                        # GPU cleanup budget. Other queues still fail closed.
                        timeout = min(2, timeout / 8)
                    q = backend.transport.json('GET', '/queue', timeout=timeout)
                    if any(not isinstance(q.get(k), list) for k in ('queue_running', 'queue_pending')):
                        raise RecoveryRequired('invalid_expected_queue_' + name)
                    entries = q['queue_running'] + q['queue_pending']
                    if entries and not (teardown and name == 'wallpaper' and prompt and
                            all(isinstance(e, list) and len(e) >= 2 and e[1] == prompt for e in entries)):
                        raise RecoveryRequired('unexplained_backend_activity_' + name)
                    if not teardown or name != 'wallpaper':
                        stats = backend.transport.json('GET', '/system_stats', timeout=remaining(deadline))
                        if not isinstance(stats.get('system'), dict):
                            raise Uncertain('backend_readiness_unverified_' + name)
                    observations[name] = q
                else:
                    q = backend.transport.json('GET', '/api/ps', timeout=remaining(deadline))
                    if q.get('models') != []:
                        raise RecoveryRequired('unexplained_backend_activity_' + name)
                    version = backend.transport.json('GET', '/api/version', timeout=remaining(deadline))
                    if not isinstance(version.get('version'), str) or not version['version']:
                        raise RecoveryRequired('backend_readiness_unverified_' + name)
                    observations[name] = q
            except Uncertain:
                if not (teardown and name == 'wallpaper'):
                    raise
                # Deadline + exclusive admission + exact container permit termination.
                # A failed API observation says nothing about process liveness.
                observations[name] = {'api_unavailable': True}
        return observations

    def gpu_sample(self, resources, deadline, executing=False, stopped=False, *, music_context=None):
        # Opt-in only for the bound post-ack Music guard. Default callers retain
        # their original hard service-limit behavior. No exception is suppressed.
        if music_context is not None and (tuple(resources) != ('gpu0',) or executing or not stopped):
            raise RecoveryRequired('music_release_observation_scope_invalid')
        pending = []
        compute_births = {}
        seen_compute = set()
        driver_health = self.driver_health(deadline)
        known = {}
        service_identities = {}
        for service in self.p['services']:
            ident = self.container(service, deadline)
            service_identities[service] = ident
            if stopped and service == 'wallpaper' and (ident['running'] or ident['pid'] != 0):
                raise RecoveryRequired('wallpaper_not_stopped_for_idle_cleanup')
            if ident['running']:
                raw = self.command([self.p['docker'], 'top', ident['id'], '-eo', 'pid'], deadline)
                lines = raw.strip().splitlines()
                if not lines or lines[0].strip() != 'PID':
                    raise RecoveryRequired('service_process_identity_unverified')
                for line in lines[1:]:
                    pid = int(line.strip())
                    if pid in known:
                        raise RecoveryRequired('ambiguous_service_process_identity')
                    known[pid] = service
        usage = {s: {r: 0 for r in resources} for s in self.p['services']}
        apps = self.command([self.p['nvidia_smi'], '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], deadline)
        uuid_resources = {self.p['gpu_uuids'][r]: r for r in resources}
        for row in csv.reader(io.StringIO(apps)):
            if not row:
                continue
            if len(row) != 3:
                raise RecoveryRequired('gpu_process_observation_invalid')
            resource = uuid_resources.get(physical_gpu_uuid(row[0].strip()))
            if resource is None:
                continue
            pid, memory = int(row[1]), float(row[2])
            if pid not in known or not math.isfinite(memory) or memory < 0:
                raise RecoveryRequired('unexplained_gpu_process_or_memory')
            if music_context is not None:
                service = known[pid]
                birth = process_birth(pid)
                if ((resource, pid) in seen_compute or
                        music_context['processes'].get(service, {}).get(str(pid)) != birth):
                    raise RecoveryRequired('music_release_unknown_or_reused_compute_process')
                seen_compute.add((resource, pid))
                compute_births[str(pid)] = birth
            usage[known[pid]][resource] += memory
        for service, values in usage.items():
            if executing and service == 'wallpaper':
                continue
            if any(v > self.p['services'][service]['idle_memory_mb'][r] for r, v in values.items()):
                if music_context is not None and service == 'music':
                    pending.append('music_memory_settling')
                else:
                    raise RecoveryRequired('service_exceeds_calibrated_idle_baseline_' + service)
        samples = {}
        for r in resources:
            rows = list(csv.reader(io.StringIO(self.command([
                self.p['nvidia_smi'], '--id=' + self.p['gpu_uuids'][r],
                '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total,retired_pages.pending',
                '--format=csv,noheader,nounits'], deadline))))
            if len(rows) != 1 or len(rows[0]) != 6 or physical_gpu_uuid(rows[0][0].strip()) != self.p['gpu_uuids'][r]:
                raise RecoveryRequired('physical_gpu_identity_unverified_' + r)
            _, memory, utilization, temperature, ecc, retired = [v.strip() for v in rows[0]]
            values = [float(v) for v in (memory, utilization, temperature)]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise RecoveryRequired('invalid_gpu_health_' + r)
            if (values[2] > self.p['max_temperature_c'] or ecc not in ('0', '[N/A]', 'N/A') or
                    retired not in ('No', 'no', '0', '[N/A]', 'N/A')):
                raise RecoveryRequired('gpu_health_requires_operator_' + r)
            ceiling = self.p['stopped_idle_memory_mb'] if stopped else self.p['idle_memory_mb']
            if not executing and (values[0] > ceiling[r] or
                                   values[1] > self.p['idle_utilization_pct']):
                if music_context is not None:
                    pending.append('gpu0_memory_or_utilization_settling')
                else:
                    raise CleanupPending('gpu_not_at_calibrated_idle_' + r)
            samples[r] = {'uuid': self.p['gpu_uuids'][r], 'memory_mb': values[0],
                          'utilization_pct': values[1], 'temperature_c': values[2],
                          'uncorrected_ecc': ecc, 'retired_pending': retired}
        if music_context is not None:
            for service in ('music', 'wallpaper'):
                if (service_identities[service] != music_context['identities'][service] or
                        self.container(service, deadline) != service_identities[service]):
                    raise RecoveryRequired('music_release_generation_changed')
            if self.processes(service_identities['music'], deadline) != music_context['processes']['music']:
                raise RecoveryRequired('music_release_process_birth_changed')
            for pid, birth in compute_births.items():
                if process_birth(pid) != birth:
                    raise RecoveryRequired('music_release_process_birth_changed')
            remaining(deadline, CleanupPending)
        return {'gpus': samples, 'service_memory_mb': usage, 'driver_health': driver_health,
                'service_identities': service_identities,
                **({'music_settling': pending, 'compute_births': compute_births}
                   if music_context is not None else {})}

    def driver_health(self, deadline):
        # Require readable, current-boot kernel diagnostics; an empty/inaccessible
        # journal is not proof that the driver has no reported faults.
        raw = self.command([self.p['journalctl'], '--dmesg', '--boot=0', '--no-pager', '--output=json'], deadline)
        if len(raw) > 8 * 1024 * 1024:
            raise RecoveryRequired('kernel_health_log_requires_operator')
        boot = guest_boot_id()
        if not valid_boot(boot):
            raise RecoveryRequired('kernel_health_boot_unverified')
        count = 0
        for line in raw.splitlines():
            item = strict_json(line)
            if item.get('_BOOT_ID') != boot.replace('-', '') or item.get('_TRANSPORT') != 'kernel':
                raise RecoveryRequired('kernel_health_log_boot_unverified')
            message = item.get('MESSAGE')
            if not isinstance(message, str):
                raise RecoveryRequired('kernel_health_log_invalid')
            count += 1
            if re.search(r'(?:NVRM|nvidia).*(?:Xid|fallen off|RmInitAdapter|fatal|GPU.*lost)', message, re.I):
                raise RecoveryRequired('kernel_gpu_driver_fault_requires_operator')
        if count == 0:
            raise RecoveryRequired('kernel_health_log_unavailable')
        return {'boot_id': boot, 'kernel_records_checked': count, 'reported_gpu_faults': False}

    def ready(self, resources, deadline, skip_wallpaper=False, stopped=False):
        stable = 0
        while True:
            remaining(deadline)
            try:
                queues = self.queues(resources, deadline, skip_wallpaper=skip_wallpaper)
                sample = (self.gpu_sample(resources, deadline, stopped=True) if stopped else
                          self.gpu_sample(resources, deadline))
            except (CleanupPending, Uncertain):
                stable = 0
            else:
                remaining(deadline)
                stable += 1
                if stable >= self.p['samples']:
                    return {'queues': queues, 'observations': sample, 'stable_samples': stable,
                            'observed_at': timestamp(), 'cleanup_verified': True,
                            **({'idle_ready': True, 'backend_stopped': True} if stopped else {'backend_ready': True})}
            time.sleep(min(self.poll_interval, remaining(deadline)))

    def stopped_ready(self, resources, deadline, processes=None):
        original = self.container('wallpaper', deadline)
        if original['running'] or original['pid'] != 0:
            raise RecoveryRequired('wallpaper_stopped_idle_required')
        self.gone(processes or {})
        proof = self.ready(resources, deadline, skip_wallpaper=True, stopped=True)
        self.gone(processes or {})
        if self.container('wallpaper', deadline) != original:
            raise RecoveryRequired('wallpaper_changed_during_stopped_cleanup')
        remaining(deadline)
        return dict(proof, stopped_container=original, original_execution_gone=True)


class RecoveryManager:
    def __init__(self, admission, targets, backends, observer=None):
        self.a, self.store, self.targets, self.backends = admission, admission.store, targets, backends
        installed = admission.snapshot()['recovery_policy']
        self.p = installed['policy']
        validate_policy(self.p, admission, targets)
        self.policy_id = installed['policy_id']
        self.observer = observer or RecoveryObserver(self.p, targets, backends)
        self.mutex = NativeLock(self.store.path('recovery.lock'))
        self.jobs = NativeLock(self.store.path('jobs.lock'))
        self.last_tick = 0
        self.thread = None
        self.wake, self.stopping = threading.Event(), threading.Event()
        self.lifecycle = None
        if on_demand(targets):
            from .wallpaper_lifecycle import WallpaperLifecycle
            self.lifecycle = WallpaperLifecycle(self)

    def start(self):
        """One lifetime-managed worker; neither scheduler nor inference pool runs recovery."""
        if self.thread is not None:
            raise RuntimeError('recovery_worker_already_started')
        self.thread = threading.Thread(target=self._loop, name='gpu-recovery', daemon=True)
        self.thread.start()

    def notify(self):
        self.wake.set()

    def _loop(self):
        try:
            while not self.stopping.is_set():
                self.wake.clear()
                self.tick(force=True)
                self.wake.wait(.5)
        except BaseException as exc:
            # No automatic worker replacement can repeat an ambiguous action.
            # The Runner retains its supervisor lock until close joins us.
            self.stopping.set()
            self.change(lambda s: s.update(recovery_worker_status='operator_required',
                                          recovery_worker_error=type(exc).__name__))

    def close(self):
        self.stopping.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join()

    def change(self, fn):
        with self.a.mutex.held():
            state = self.a.snapshot()
            before = encode(state)
            result = fn(state)
            try:
                if encode(state) != before:
                    self.store.write('ownership.json', state)
            except OSError as exc:
                raise PersistenceUncertain() from exc
            return result

    def update_job(self, job_id, **fields):
        with self.jobs.held():
            job = self.store.job(job_id)
            if all(job.get(key) == value and key in job for key, value in fields.items()):
                return job
            job.update(fields, updated_at=timestamp())
            try:
                self.store.put_job(job)
            except OSError as exc:
                raise PersistenceUncertain() from exc
            return job

    @contextmanager
    def locks(self, resources, owner=None, boot_owners=None):
        locks = []
        try:
            with self.a.mutex.held():
                state = self.a.snapshot()
                self.a._check_physical_identities(state)
                for r in resources:
                    existing = state['owners'].get(r)
                    if boot_owners is not None:
                        previous = boot_owners.get(r)
                        if (existing is None) != (previous is None) or (
                                existing is not None and binding(existing) != binding(previous)):
                            raise RecoveryRequired('boot_startup_owner_identity_changed')
                    elif (owner is None and existing) or (owner is not None and (
                            existing is None or binding(existing) != binding(owner))):
                        raise RecoveryRequired('recovery_owner_identity_changed')
                    lock = NativeLock(self.a.lock_paths[r], create=False)
                    if not lock.acquire(0):
                        raise RecoveryRequired('recovery_execution_still_supervised')
                    locks.append(lock)
                    if lock_identity(os.fstat(lock.fd)) != state['lock_identity'][r]:
                        raise RecoveryRequired('physical_lock_identity_changed_during_acquisition')
            yield
        finally:
            for lock in reversed(locks):
                lock.close()

    def prepare(self, job_id, lease):
        job = self.store.job(job_id)
        fields = {'boot_id': lease.owner.get('boot_id'),
                  'execution_deadline': now() + self.targets[job['target']].execution_timeout}
        if job['target'] == 'music' and 'stopped_idle_memory_mb' in self.p:
            from .music_cleanup import capture_execution
            fields['music_execution'] = capture_execution(self.observer, time.monotonic() + self.p['observation_timeout'])
        if job['target'] == 'wallpaper':
            ident = self.observer.container('wallpaper', time.monotonic() + self.p['observation_timeout'])
            if not ident['running'] or ident['pid'] <= 0:
                raise RecoveryRequired('wallpaper_not_running_before_submission')
            fields['execution_container'] = ident
            fields['execution_processes'] = self.observer.processes(ident, time.monotonic() + self.p['observation_timeout'])
        self.update_job(job_id, **fields)

    def startup(self):
        if self.lifecycle:
            self.lifecycle.finalize_released()
        boot = self.a.boot_reader()
        owners = self.a.snapshot()['owners']
        # Never replay never-admitted work; same-boot independently supervised stages survive.
        with self.jobs.held():
            for job in self.store.jobs():
                if job['state'] in FINAL or any(o['job_id'] == job['request_id'] for o in owners.values()):
                    continue
                if job.get('host_stage') and (not valid_boot(job.get('boot_id')) or job['boot_id'] == boot):
                    continue
                job.update(state='failed', error_code='restart_before_admission_no_replay', updated_at=timestamp())
                if job.get('host_stage'):
                    job['host_exit_code'] = 70
                self.store.put_job(job)
        self.tick(force=True)

    def boot_record(self, boot, **fields):
        def update(state):
            records = state.setdefault('boot_startups', {})
            record = dict(records.get(boot, {}), **fields)
            records[boot] = record
            return record
        return self.change(update)

    def running(self, original, deadline, require_new=False):
        """Observe Docker's asynchronous start; never retry a start command."""
        while True:
            current = self.observer.container('wallpaper', deadline)
            if (current['id'] != original['id'] or
                    current['contract_sha256'] != original['contract_sha256']):
                raise RecoveryRequired('wallpaper_start_identity_changed')
            if current['running'] and current['pid'] > 0 and (
                    not require_new or current['started_at'] != original['started_at']):
                return current
            time.sleep(min(.1, remaining(deadline)))

    def boot_startup(self, boot):
        """Exactly one approved-container start per verified new kernel boot.

        The shared boot fence stays closed, including across interrupted writes.
        This path is never entered when verified_boot equals the current boot.
        """
        state = self.a.snapshot()
        record = state.get('boot_startups', {}).get(boot, {})
        if record.get('phase') == 'operator_required':
            return False
        if (state.get('boot_recovery_status') == 'operator_required' and
                state.get('boot_recovery_attempt_boot') == boot):
            return False
        resources = tuple(sorted(self.a.lock_paths))
        try:
            if not valid_boot(boot) or not valid_boot(state.get('verified_boot')):
                raise RecoveryRequired('verified_previous_and_current_boot_required_for_startup')
            if boot == state['verified_boot']:
                return True
            with self.locks(resources, boot_owners=state['owners']):
                for owner in state['owners'].values():
                    if not valid_boot(owner.get('boot_id')) or owner['boot_id'] == boot:
                        self.update_job(owner['job_id'], recovery_status='operator_required',
                                        recovery_error='lease_boot_metadata_unverified')
                        raise RecoveryRequired('boot_startup_owner_boot_unverified')
                # Genuine old execution is interrupted; current readiness is a
                # separate condition and owners remain until all proof passes.
                for owner in {o['lease_id']: o for o in state['owners'].values()}.values():
                    job = self.store.job(owner['job_id'])
                    held = sorted(r for r, o in state['owners'].items() if o['lease_id'] == owner['lease_id'])
                    if (job.get('request_hash') != owner['request_hash'] or job.get('target') != owner['target'] or
                            sorted(job.get('resources', [])) != held):
                        raise RecoveryRequired('boot_startup_job_owner_binding_changed')
                    if job['state'] not in FINAL or (job.get('music_cleanup_context') and
                            job['state'] == 'success' and owner['lease_id'] not in
                            self.a.snapshot().get('music_release_receipts', {})):
                        self.update_job(owner['job_id'], state='failed', error_code='guest_reboot_interrupted',
                            **({'host_exit_code': 70} if job.get('host_stage') else {}))
                deadline = time.monotonic() + self.p['observation_timeout']
                current = self.observer.container('wallpaper', deadline)
                expected = {'id': self.p['wallpaper'],
                            'contract_sha256': self.p['services']['wallpaper']['contract_sha256']}
                if any(current.get(k) != v for k, v in expected.items()):
                    raise RecoveryRequired('boot_wallpaper_identity_changed')
                identity = {'boot_id': boot, 'previous_verified_boot': state['verified_boot'],
                    'policy_id': self.policy_id, 'container': expected, 'physical_gpu_uuids': self.p['gpu_uuids']}
                if record and record.get('identity') != identity:
                    raise RecoveryRequired('boot_startup_identity_changed')
                if not record:
                    record = self.boot_record(boot, identity=identity, phase='observing', created_at=timestamp())
                if self.lifecycle:
                    # Intentionally stopped is healthy. A reboot grants no start.
                    proof = self.observer.stopped_ready(resources, deadline)
                    self.boot_record(boot, phase='ready', running_container=current, proof=proof)
                    return True
                if record.get('phase') == 'ready':
                    if current != record['running_container']:
                        raise RecoveryRequired('boot_wallpaper_generation_changed_after_readiness')
                    return True
                if record.get('start_intent'):
                    # Crash after intent: observe only. No second start even if
                    # the first command did not take effect.
                    current = self.running(record['original_container'], deadline, require_new=True)
                elif not current['running']:
                    if current['pid'] != 0:
                        raise RecoveryRequired('boot_wallpaper_stopped_state_unverified')
                    # Verify other queues/physical cleanup while wallpaper is
                    # deliberately absent. No elapsed-time cleanup assumptions.
                    prestart = self.observer.ready(resources, deadline, skip_wallpaper=True)
                    record = self.boot_record(boot, phase='starting', original_container=current,
                                              prestart_proof=prestart, start_intent=timestamp())
                    try:
                        self.observer.start(current, deadline)
                    except Uncertain:
                        record = self.boot_record(boot, start_response_uncertain=True)
                    current = self.running(current, deadline, require_new=True)
                else:
                    current = self.running(current, deadline)
                # Preserve the first observed generation across a readiness race
                # and any interrupted persistence; external restarts fail closed.
                if record.get('running_container') and record['running_container'] != current:
                    raise RecoveryRequired('boot_wallpaper_generation_changed_during_startup')
                self.boot_record(boot, phase='running', running_container=current)
                proof = self.observer.ready(resources, deadline)
                if self.observer.container('wallpaper', deadline) != current:
                    raise RecoveryRequired('boot_wallpaper_generation_changed_during_readiness')
                remaining(deadline)
                self.boot_record(boot, phase='ready', proof=proof)
                return True
        except PersistenceUncertain:
            raise
        except Exception as exc:
            if isinstance(exc, RecoveryRequired) and str(exc) == 'recovery_execution_still_supervised':
                return False
            if record:
                self.boot_record(boot, phase='operator_required', error=error(exc))
            self.change(lambda s: s.update(boot_recovery_status='operator_required',
                boot_recovery_error=error(exc), boot_recovery_attempt_boot=boot))
            return False

    def boot_readiness(self, resources, deadline, *, startup_record=None):
        proof = (self.observer.stopped_ready(resources, deadline) if self.lifecycle else
                 self.observer.ready(resources, deadline))
        current = self.boot_generation(deadline, startup_record=startup_record)
        return dict(proof, boot_wallpaper_container=current, verified_boot=self.a.boot_reader())

    def boot_generation(self, deadline, *, startup_record=None):
        record = (self.a.snapshot().get('boot_startups', {}).get(self.a.boot_reader(), {})
                  if startup_record is None else startup_record)
        current = self.observer.container('wallpaper', deadline)
        if record.get('phase') != 'ready' or current != record['running_container']:
            raise RecoveryRequired('boot_wallpaper_generation_changed_before_handoff')
        remaining(deadline)
        return current

    def tick(self, force=False):
        if not force and time.monotonic() - self.last_tick < .5:
            return
        if not self.mutex.acquire(0):
            return
        self.last_tick = time.monotonic()
        try:
            if self.lifecycle:
                self.lifecycle.finalize_released()
            state = self.a.snapshot()
            boot = self.a.boot_reader()
            if state.get('verified_boot') != boot and not self.boot_startup(boot):
                return
            unique = {o['lease_id']: o for o in state['owners'].values()}
            for owner in unique.values():
                resources = tuple(sorted(r for r, o in state['owners'].items() if o['lease_id'] == owner['lease_id']))
                self.recover(owner, resources)
            state = self.a.snapshot()
            boot = self.a.boot_reader()
            if state.get('verified_boot') != boot and not state['owners']:
                if (state.get('boot_recovery_status') == 'operator_required' and
                        state.get('boot_recovery_attempt_boot') == boot):
                    return
                try:
                    if not valid_boot(boot):
                        raise RecoveryRequired('current_kernel_boot_id_unverified')
                    with self.locks(tuple(sorted(self.a.lock_paths))):
                        proof = self.boot_readiness(tuple(sorted(self.a.lock_paths)),
                                time.monotonic() + self.p['observation_timeout'])
                        def verified(s):
                            self.a._check_physical_identities(s)
                            if s['owners'] or self.a.boot_reader() != boot:
                                raise RecoveryRequired('boot_readiness_changed_before_admission')
                            s.update(verified_boot=boot, boot_recovery_status='ready', boot_readiness=proof)
                        self.change(verified)
                except PersistenceUncertain:
                    raise
                except Exception as exc:
                    self.change(lambda s: s.update(boot_recovery_status='operator_required',
                                                  boot_recovery_error=error(exc), boot_recovery_attempt_boot=boot))
        except Exception as exc:
            # Storage/control ambiguity never escapes into scheduler admission.
            # Durable owners and the boot gate remain authoritative.
            try:
                self.change(lambda s: s.update(automatic_recovery_error=error(exc)))
            except Exception:
                pass
        finally:
            self.mutex.close()

    def incident(self, owner, resources, kind):
        identity = dict(binding(owner), resources=list(resources), physical_gpu_uuids={
            r: self.p['gpu_uuids'][r] for r in resources}, policy_id=self.policy_id)
        incident_id = digest(identity)
        def create(state):
            incidents = state.setdefault('automatic_recovery', {})
            if incident_id not in incidents:
                incidents[incident_id] = {'id': incident_id, 'identity': identity, 'kind': kind,
                                          'phase': 'observing', 'created_at': timestamp(),
                                          'observed_boot': self.a.boot_reader()}
            return incidents[incident_id]
        return self.change(create)

    def save(self, incident, **fields):
        def update(state):
            current = dict(state['automatic_recovery'][incident['id']], **fields)
            state['automatic_recovery'][incident['id']] = current
            return current
        return self.change(update)

    def recover(self, owner, resources):
        boot, old_boot = self.a.boot_reader(), owner.get('boot_id')
        reboot = valid_boot(boot) and valid_boot(old_boot) and boot != old_boot
        job = self.store.job(owner['job_id'])
        if valid_boot(boot) and valid_boot(old_boot) and not reboot and (
                job.get('host_stage') or owner['target'] != 'wallpaper'):
            return
        incident = None
        try:
            # Eligibility and native supervision precede ALL recovery writes.
            # A healthy worker can finish its ordinary cleanup undisturbed.
            with self.locks(resources, owner):
                job = self.store.job(owner['job_id'])
                if not valid_boot(boot) or not valid_boot(old_boot):
                    self.update_job(owner['job_id'], recovery_status='operator_required',
                                    recovery_error='lease_boot_metadata_unverified')
                    return
                if (job.get('request_hash') != owner['request_hash'] or job.get('target') != owner['target'] or
                        tuple(sorted(job.get('resources', []))) != resources):
                    self.update_job(owner['job_id'], recovery_status='operator_required', recovery_error='job_owner_binding_changed')
                    return
                if not reboot and self.lifecycle:
                    if self.lifecycle.resume_normal(owner, resources, job):
                        return
                incident = self.incident(owner, resources, 'previous_boot' if reboot else 'wallpaper')
                if incident['phase'] in ('operator_required', 'released'):
                    return
                deadline = time.monotonic() + self.p['observation_timeout']
                if reboot:
                    if job.get('execution_container'):
                        incident = self.save(incident, container=job['execution_container'])
                    if job['state'] not in FINAL or (job.get('music_cleanup_context') and
                            job['state'] == 'success' and owner['lease_id'] not in
                            self.a.snapshot().get('music_release_receipts', {})):
                        self.update_job(owner['job_id'], state='failed', error_code='guest_reboot_interrupted',
                                        **({'host_exit_code': 70} if job.get('host_stage') else {}))
                    # Reboot proves old execution interruption, not present readiness or placement.
                    proof = self.boot_readiness(resources, deadline)
                    if job.get('dual_gpu_session'):
                        from .dual_gpu import ComposeOllamaSwitch, switch_binding
                        target = self.targets[owner['target']]
                        if job.get('switch_binding') != switch_binding(target.mode_switch) or set(resources) != {'gpu0', 'gpu1'}:
                            raise RecoveryRequired('dual_restoration_binding_changed')
                        restored = ComposeOllamaSwitch(target).verify('gpu1')
                        self.update_job(owner['job_id'], dual_restoration_verified=dict(restored,
                            gpu1_restored=True, job_id=owner['job_id'], lease_id=owner['lease_id'], verified_boot=boot))
                        proof['dual_restoration_verified'] = restored
                    self.boot_generation(deadline)
                    self.finish(incident, owner, resources, proof)
                    return
                if self.lifecycle:
                    self.lifecycle.recover(incident, owner, resources, self.store.job(owner['job_id']), deadline)
                else:
                    self.wallpaper(incident, owner, resources, job, deadline)
        except (RecoveryRequired, CleanupPending, Uncertain, ValueError, KeyError, OSError) as exc:
            if isinstance(exc, RecoveryRequired) and str(exc) == 'recovery_execution_still_supervised':
                return
            if isinstance(exc, RecoveryRequired) and str(exc) == 'recovery_owner_identity_changed':
                # A worker may have completed and atomically released after our
                # owner snapshot. Its final record must remain undisturbed.
                if not any(o['lease_id'] == owner['lease_id'] for o in self.a.snapshot()['owners'].values()):
                    return
            if incident is not None:
                self.save(incident, phase='operator_required', error=error(exc))
            self.update_job(owner['job_id'], recovery_status='operator_required', recovery_error=error(exc))

    def wallpaper(self, incident, owner, resources, job, deadline):
        expiry = job.get('execution_deadline')
        try:
            valid_expiry = type(expiry) in (int, float) and math.isfinite(expiry) and expiry > 0
        except OverflowError:
            valid_expiry = False
        if not valid_expiry:
            raise RecoveryRequired('execution_deadline_missing_or_invalid')
        original = job.get('execution_container')
        if not isinstance(original, dict):
            raise RecoveryRequired('execution_container_metadata_missing')
        if incident.get('container') and incident['container'] != original:
            raise RecoveryRequired('incident_container_identity_changed')
        incident = self.save(incident, container=original)
        current = self.observer.container('wallpaper', deadline)
        if (current['id'] != original['id'] or current['contract_sha256'] != original['contract_sha256'] or
                (not incident.get('start_intent') and current['started_at'] != original['started_at'])):
            raise RecoveryRequired('original_container_generation_changed')
        if incident.get('start_intent'):
            # Start may have succeeded before a runner crash. Never issue it twice.
            current = self.running(original, deadline, require_new=True)
            self.observer.gone(incident['original_pids'])
            proof = self.observer.ready(resources, deadline)
            if self.observer.container('wallpaper', deadline) != current:
                raise RecoveryRequired('wallpaper_generation_changed_during_readiness')
            proof.update(original_execution_gone=True, stopped=incident['stopped_verified'], restarted=current)
            self.final_job(owner['job_id'], incident.get('execution_error') or 'wallpaper_interrupted_by_recovery')
            self.finish(incident, owner, resources, proof)
            return
        if incident['phase'] == 'verified':
            proof = self.observer.ready(resources, deadline)
            self.finish(incident, owner, resources, dict(incident['proof'], fresh_readiness=proof))
            return
        prompt = job.get('backend_detail', {}).get('prompt_id')
        backend = self.backends['wallpaper']
        # A single bounded history check may establish completion without any stop.
        if not incident.get('term_intent') and incident.get('cleanup_attempts', 0) < 2 and (
                incident.get('reconciled') or incident.get('history_attempts', 0) < 2):
            incident = self.save(incident, history_attempts=incident.get('history_attempts', 0) + 1)
            execution = None
            if incident.get('reconciled'):
                from .backends import Execution
                execution = Execution({}, {'prompt_id': prompt}, error_code=incident.get('execution_error'))
            elif prompt:
                try:
                    execution = backend.reconcile(prompt, min(deadline,
                        time.monotonic() + min(5, self.p['observation_timeout'] / 4)))
                except Uncertain:
                    pass
            if execution is not None:
                try:
                    self.publish(owner['job_id'], execution)
                except OSError as exc:
                    raise PersistenceUncertain() from exc
                incident = self.save(incident, reconciled=True, execution_error=execution.error_code)
                incident = self.save(incident, cleanup_attempts=incident.get('cleanup_attempts', 0) + 1)
                try:
                    proof = backend.cleanup(execution, min(deadline,
                        time.monotonic() + min(10, self.p['observation_timeout'] / 4)))
                    proof['readiness'] = self.observer.ready(resources, deadline)
                except (CleanupPending, Uncertain):
                    pass
                else:
                    self.final_job(owner['job_id'], execution.error_code)
                    self.finish(incident, owner, resources, proof)
                    return
        if not job.get('owned_execution_failed') and not incident.get('execution_error') and now() < expiry:
            self.save(incident, phase='waiting_execution_deadline')
            return
        if not incident.get('term_intent'):
            self.observer.queues(resources, deadline, prompt=prompt, teardown=True)
            self.observer.gpu_sample(resources, deadline, executing=True)
            incident = self.reserve_attempt(incident)
            # Intent is durable before any signal; never send this signal twice.
            incident = self.save(incident, phase='stopping', term_intent=timestamp(),
                                 original_pids=dict(job.get('execution_processes', {}),
                                                   **self.observer.processes(current, deadline)))
            if current['running']:
                try:
                    self.observer.signal(original, 'TERM', deadline)
                except Uncertain:
                    incident = self.save(incident, term_response_uncertain=True)
        # Resume by observation. Time limits bound waiting; only inspect proves exit.
        grace_end = seconds(incident['term_intent']) + self.p['grace_seconds']
        while True:
            current = self.observer.container('wallpaper', deadline)
            if not current['running'] and current['pid'] == 0:
                break
            if current['started_at'] != original['started_at']:
                raise RecoveryRequired('container_restarted_outside_recovery')
            if now() >= grace_end:
                if incident.get('kill_intent') or not self.p['allow_force']:
                    raise RecoveryRequired('wallpaper_exit_unverified_no_more_termination_attempts')
                incident = self.save(incident, kill_intent=timestamp())
                try:
                    self.observer.signal(original, 'KILL', deadline)
                except Uncertain:
                    incident = self.save(incident, kill_response_uncertain=True)
            time.sleep(min(.1, remaining(deadline)))
        # PID and all original CUDA contexts must be absent; calibrated unrelated idle contexts remain.
        incident = self.save(incident, phase='stopped', stopped_verified=current)
        self.observer.gone(incident['original_pids'])
        self.observer.gpu_sample(resources, deadline)
        if not incident.get('start_intent'):
            incident = self.save(incident, phase='starting', start_intent=timestamp())
            try:
                self.observer.start(original, deadline)
            except Uncertain:
                incident = self.save(incident, start_response_uncertain=True)
        current = self.running(original, deadline, require_new=True)
        proof = self.observer.ready(resources, deadline)
        if self.observer.container('wallpaper', deadline) != current:
            raise RecoveryRequired('wallpaper_generation_changed_during_readiness')
        proof.update(original_execution_gone=True, stopped=incident['stopped_verified'], restarted=current)
        self.final_job(owner['job_id'], incident.get('execution_error') or 'wallpaper_interrupted_by_recovery')
        self.finish(incident, owner, resources, proof)

    def reserve_attempt(self, incident):
        if incident.get('attempt_reserved'):
            return incident
        def reserve(state):
            budget = state.setdefault('recovery_budget', [])
            # Clock rollback cannot expire entries early. No reset on restart or policy reinstall.
            recent = [e for e in budget if now() - seconds(e['at']) < self.p['window_seconds']]
            if len(recent) >= self.p['max_incidents']:
                raise RecoveryRequired('persistent_recovery_rate_limit')
            budget.append({'incident_id': incident['id'], 'at': timestamp()})
            updated = dict(incident, attempt_reserved=True)
            state['automatic_recovery'][incident['id']] = updated
            return updated
        return self.change(reserve)

    def publish(self, job_id, execution):
        if self.store.read('results/' + job_id + '.json') is not None:
            return
        if execution.artifact is not None:
            self.store.write_bytes('results/' + job_id + '.bin', execution.artifact)
        self.store.write('results/' + job_id + '.json', {'value': execution.value,
            'media_type': execution.media_type, 'artifact_available': execution.artifact is not None})

    def final_job(self, job_id, execution_error):
        job = self.store.job(job_id)
        if job['state'] not in FINAL:
            self.update_job(job_id, state='failed' if execution_error else 'success', error_code=execution_error)

    def finish(self, incident, owner, resources, proof, *, operator_transition=None):
        # Proof -> terminal job -> atomic audit+release. Crash at any edge is resumable.
        readiness = proof.get('readiness', proof)
        ready_key = 'idle_ready' if self.lifecycle else 'backend_ready'
        if (proof.get('cleanup_verified') is not True or readiness.get(ready_key) is not True or
                readiness.get('cleanup_verified') is not True):
            raise RecoveryRequired('recovery_readiness_or_cleanup_unverified')
        incident = self.save(incident, phase='verified', proof=proof)
        self.update_job(owner['job_id'], recovery_status='verified', recovery_incident=incident['id'],
                        recovery_proof=proof)
        def release(state):
            self.a._check_physical_identities(state)
            if any(binding(state['owners'].get(r, {})) != binding(owner) for r in resources):
                raise RecoveryRequired('recovery_owner_changed_before_release')
            if self.store.job(owner['job_id'])['state'] not in FINAL:
                raise RecoveryRequired('recovery_job_not_terminal')
            if operator_transition is not None:
                from .reboot_contract import commit_transition
                commit_transition(self, state, operator_transition, proof)
            state['automatic_recovery'][incident['id']] = dict(incident, phase='released', released_at=timestamp())
            if not any(e.get('incident_id') == incident['id'] for e in state.setdefault('recoveries', [])):
                audit = {'incident_id': incident['id'], 'identity': incident['identity'],
                    'container': incident.get('container'), 'observed_boot': self.a.boot_reader(),
                    'proof': proof, 'recovered_at': timestamp(), 'provenance': 'runner_observations'}
                if operator_transition is not None:
                    audit['operator_contract_transition'] = operator_transition['audit']
                state['recoveries'].append(audit)
            for r in resources:
                del state['owners'][r]
        if operator_transition is not None:
            with self.jobs.held():
                self.change(release)
        else:
            self.change(release)


def binding(owner):
    return {k: owner.get(k) for k in ('job_id', 'request_hash', 'lease_id', 'target', 'boot_id')}


def error(exc):
    return getattr(exc, 'code', None) or (str(exc) if isinstance(exc, RecoveryRequired) else type(exc).__name__)


def authorize_policy(admission, targets, backends, policy, approval_ref, observer=None):
    """Deployment-only, owner-free upgrade. No invented boot fields for legacy records."""
    validate_policy(policy, admission, targets)
    if not isinstance(approval_ref, str) or not approval_ref.strip() or not valid_boot(admission.boot_reader()):
        raise ValueError('approval_and_verified_kernel_boot_required')
    policy_id = digest(policy)
    locks = []
    with NativeLock(admission.store.path('maintenance.lock')).held(0), \
            NativeLock(admission.store.path('supervisor.lock')).held(0), \
            NativeLock(admission.store.path('jobs.lock')).held(0), admission.mutex.held():
        state = admission.snapshot()
        if not state['ready'] or state['owners'] or any(j['state'] not in FINAL for j in admission.store.jobs()):
            raise RecoveryRequired('policy_upgrade_requires_clean_owner_free_installation')
        previous = state.get('recovery_policy')
        if previous:
            if previous['policy_id'] != policy_id:
                raise RecoveryRequired('recovery_policy_change_requires_separate_review')
            return previous
        try:
            admission._check_physical_identities(state)
            for r, path in sorted(admission.lock_paths.items()):
                lock = NativeLock(path, create=False)
                if not lock.acquire(0):
                    raise RecoveryRequired('policy_upgrade_physical_resource_busy')
                locks.append(lock)
                if lock_identity(os.fstat(lock.fd)) != state['lock_identity'][r]:
                    raise RecoveryRequired('physical_lock_identity_changed_during_acquisition')
            observer = observer or RecoveryObserver(policy, targets, backends)
            deadline = time.monotonic() + policy['observation_timeout']
            proof = (observer.stopped_ready(tuple(sorted(admission.lock_paths)), deadline) if on_demand(targets) else
                     observer.ready(tuple(sorted(admission.lock_paths)), deadline))
            for target in targets.values():
                if target.mode_switch is not None:
                    from .dual_gpu import ComposeOllamaSwitch
                    ComposeOllamaSwitch(target).verify('gpu1')
            state['recovery_policy'] = {'policy': policy, 'policy_id': policy_id,
                'approval_ref': approval_ref, 'authorized_at': timestamp()}
            state.update(verified_boot=admission.boot_reader(), boot_readiness=proof, boot_recovery_status='ready')
            admission.store.write('ownership.json', state)
            return state['recovery_policy']
        finally:
            for lock in reversed(locks):
                lock.close()
