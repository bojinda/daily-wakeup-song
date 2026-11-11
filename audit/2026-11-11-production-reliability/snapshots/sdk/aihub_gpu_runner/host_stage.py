"""Local host stages sharing Admission; no HTTP command-execution interface."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .backends import Execution, HTTPTransport, NvidiaMemoryProbe, OllamaBackend, CleanupPending
from .config import admission_from_config
from .core import Conflict, FINAL, NativeLock, RecoveryRequired, check_id, encode, timestamp, strict_json
from .dual_gpu import ComposeOllamaSwitch, switch_binding


class Interrupted(Exception):
    def __init__(self, number):
        self.number = number


def capture_input_files(environment):
    """Trusted local declarations; missing optional inputs stay missing."""
    specs = strict_json(environment.get('AIHUB_GPU_STAGE_INPUT_FILES', '{}'))
    if not isinstance(specs, dict) or len(specs) > 64:
        raise ValueError('invalid_stage_input_declaration')
    inputs = {}
    for role, spec in sorted(specs.items()):
        check_id(role)
        if (not isinstance(spec, dict) or set(spec) != {'path', 'required', 'snapshot'} or
                not isinstance(spec['path'], str) or type(spec['required']) is not bool or
                type(spec['snapshot']) is not bool):
            raise ValueError('invalid_stage_input_spec')
        path = Path(spec['path']).expanduser().resolve()
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            if spec['required']:
                raise ValueError('required_stage_input_missing') from None
            data = None
        inputs[role] = {'path': path, 'required': spec['required'], 'snapshot': spec['snapshot'],
                        'data': data, 'sha256': None if data is None else hashlib.sha256(data).hexdigest()}
    return inputs


def input_fingerprints(inputs):
    return {role: {'path_sha256': hashlib.sha256(str(item['path']).encode()).hexdigest(),
                   'present': item['data'] is not None, 'sha256': item['sha256'],
                   'required': item['required'], 'snapshot': item['snapshot']}
            for role, item in inputs.items()}


def input_sources_match(inputs):
    for item in inputs.values():
        try:
            current = item['path'].read_bytes()
        except FileNotFoundError:
            current = None
        if (None if current is None else hashlib.sha256(current).hexdigest()) != item['sha256']:
            return False
    return True


def command_identity(argv, environment, inputs=None):
    """Store only hashes, never command arguments/tokens or input contents."""
    source = environment.get('AIHUB_GPU_STAGE_INPUT')
    digest = None
    if source:
        hasher = hashlib.sha256()
        with open(source, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        digest = hasher.hexdigest()
    settings = {key: value for key, value in environment.items() if key.startswith(
        ('WHISPERX_', 'OLLAMA_', 'MEETING_', 'LESSON_', 'SPEAKER_', 'TRANSCRIPT_'))}
    # The private .env identity includes configured aliases/recap/model choices.
    config_file = environment.get('AIHUB_GPU_STAGE_SETTINGS_FILE')
    settings_hash = (hashlib.sha256(Path(config_file).read_bytes()).hexdigest()
                     if config_file and Path(config_file).is_file() else None)
    value = {'argv': list(argv), 'input_sha256': digest, 'settings': settings, 'config_sha256': settings_hash}
    declared = capture_input_files(environment) if inputs is None else inputs
    if declared:
        value['declared_inputs'] = input_fingerprints(declared)
    return hashlib.sha256(encode(value)).hexdigest()


class HostStage:
    """One stage lease, same durable journal/physical locks as Runner."""
    def __init__(self, admission, resource, job_id, request_hash, *, ollama_target=None,
                 host_policy=None, transport_factory=HTTPTransport, probe=None, poll_interval=.05,
                 input_files=None, switch_controller=None):
        if resource not in ('gpu0', 'gpu1', 'gpu0+gpu1'):
            raise ValueError('invalid_host_resource')
        resources = ('gpu0', 'gpu1') if resource == 'gpu0+gpu1' else (resource,)
        if resource != 'gpu0' and (ollama_target is None or ollama_target.kind != 'ollama'
                                   or ollama_target.resources != resources or
                                   resource == 'gpu0+gpu1' and ollama_target.mode_switch is None):
            raise ValueError('routine_ollama_target_required')
        policy = dict(host_policy or {})
        if resource == 'gpu0':
            ceiling = policy.get('idle_memory_mb')
            if (type(ceiling) not in (int, float) or not math.isfinite(ceiling) or ceiling < 0
                    or type(policy.get('cleanup_samples', 3)) is not int
                    or not 2 <= policy.get('cleanup_samples', 3) <= 20
                    or type(policy.get('cleanup_timeout', 60)) not in (int, float)
                    or not math.isfinite(policy.get('cleanup_timeout', 60))
                    or policy.get('cleanup_timeout', 60) <= 0):
                raise ValueError('host_gpu0_cleanup_calibration_required')
        self.admission, self.store = admission, admission.store
        self.resource, self.job_id, self.request_hash = resource, check_id(job_id), request_hash
        self.resources = resources
        self.dual = resource == 'gpu0+gpu1'
        self.is_ollama = resource != 'gpu0'
        self.target = ollama_target
        self.target_name = ollama_target.name if self.is_ollama else 'host-whisperx'
        self.policy, self.transport_factory = policy, transport_factory
        self.probe = probe or NvidiaMemoryProbe(0)
        self.poll_interval = poll_interval
        self.mutex = NativeLock(self.store.path('jobs.lock'))
        self.instance = NativeLock(self.store.path('host-' + self.job_id + '.lock'))
        self.lease = None
        self.input_files = input_files or {}
        self.switch = (switch_controller or ComposeOllamaSwitch(ollama_target)) if self.dual else None

    def update(self, **fields):
        with self.mutex.held():
            value = self.store.job(self.job_id)
            value.update(fields, updated_at=timestamp())
            self.store.put_job(value)
            return value

    def _cleanup(self):
        job = self.store.job(self.job_id)
        requests = job.get('host_requests', [])
        if any(item['state'] != 'completed' for item in requests):
            raise RecoveryRequired('host_request_unresolved')
        if self.is_ollama:
            if requests and self.admission.snapshot()['owners']['gpu1']['phase'] not in (
                    'host_requests_complete', 'cleanup_pending'):
                raise RecoveryRequired('host_request_persistence_unconfirmed')
            backend = OllamaBackend(self.target, self.transport_factory(self.target.base_url))
            from .ollama_cleanup import cleanup_guard
            deadline = time.monotonic() + self.target.cleanup_timeout
            guard = cleanup_guard(self.admission, self.target, self.lease.owner if self.lease else {},
                                  backend, record=self.update)
            if guard:
                guard.prepare(deadline)
                guard.mark_phase('owned_unload')
            try:
                if requests:
                    backend.cleanup_models({item['model'] for item in requests}, deadline)
            except Exception as exc:
                if guard:
                    guard.report_failure(exc)
                raise
            proof = {'completion_verified': True, 'cleanup_verified': True,
                    'policy': 'all_stage_requests_completed_then_owned_model_cleanup',
                    'request_count': len(requests)}
            if guard:
                proof['physical_ollama_idle'] = guard.verify(deadline)
            if self.dual:
                # Only known-complete calls plus verified owned unload can reach restoration.
                self.lease.mark('restore_intent')
                restored = self.switch.restore()
                if (any(restored.get(k) is not True for k in
                        ('gpu1_restored', 'backend_ready', 'placement_verified')) or
                        restored.get('mode') != 'gpu1' or restored.get('physical_gpus') != ['gpu1'] or
                        restored.get('switch_binding') != switch_binding(self.target.mode_switch)):
                    raise CleanupPending('dual_restoration_unconfirmed')
                restored = dict(restored, job_id=self.job_id, lease_id=self.lease.owner['lease_id'])
                self.update(dual_restoration_verified=restored)
                proof.update(restored)
            return proof
        deadline = time.monotonic() + self.policy.get('cleanup_timeout', 60)
        stable = 0
        while time.monotonic() < deadline:
            used = self.probe()
            if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
                raise CleanupPending('invalid_host_memory_observation')
            stable = stable + 1 if used <= self.policy['idle_memory_mb'] else 0
            if stable >= self.policy.get('cleanup_samples', 3):
                return {'completion_verified': True, 'cleanup_verified': True,
                        'policy': 'foreground_group_completed_calibrated_gpu_memory',
                        'stable_samples': stable, 'memory_mb': used}
            time.sleep(self.poll_interval)
        raise CleanupPending('host_gpu0_cleanup_unconfirmed')

    def run(self, argv, environment, timeout=3600):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('AIHUB_GPU_LOCK_TIMEOUT must be nonnegative seconds')
        if self.dual and environment.get('AIHUB_GPU_DUAL_APPROVAL') != self.target.mode_switch['activation_approval_ref']:
            raise ValueError('explicit_dual_window_approval_required')
        if not input_sources_match(self.input_files):
            raise RecoveryRequired('stage_inputs_changed_before_admission')
        if not self.instance.acquire(0):
            # Duplicate clients may wait for the same independently supervised
            # stage. Their deadline cannot cancel work or clear its owner.
            deadline = time.monotonic() + timeout
            while True:
                with self.mutex.held():
                    old = self.store.read(self.store.job_name(self.job_id))
                    if old is not None and old['request_hash'] != self.request_hash:
                        raise Conflict('request_id_payload_conflict')
                    if old is not None and old['state'] in FINAL:
                        owners = self.admission.snapshot()['owners']
                        if not any(o['job_id'] == self.job_id for o in owners.values()):
                            if not input_sources_match(self.input_files):
                                raise RecoveryRequired('stage_inputs_changed_before_cached_result')
                            return old.get('host_exit_code', 70)
                if self.instance.acquire(0):
                    self.instance.close()
                    raise RecoveryRequired('existing_host_stage_requires_recovery_no_replay')
                if time.monotonic() >= deadline:
                    return 75
                time.sleep(self.poll_interval)
        child = None
        previous_signals = {}
        def interrupted(number, frame):
            raise Interrupted(number)
        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                previous_signals[number] = signal.signal(number, interrupted)
            with self.mutex.held():
                old = self.store.read(self.store.job_name(self.job_id))
                if old:
                    if old['request_hash'] != self.request_hash or old.get('host_stage') is not True:
                        raise Conflict('request_id_payload_conflict')
                    owners = self.admission.snapshot()['owners']
                    if old['state'] in FINAL and not any(o['job_id'] == self.job_id for o in owners.values()):
                        if not input_sources_match(self.input_files):
                            raise RecoveryRequired('stage_inputs_changed_before_cached_result')
                        return old.get('host_exit_code', 70)
                    old.update(state='reconciling', error_code='host_restart_no_automatic_replay', updated_at=timestamp())
                    self.store.put_job(old)
                    raise RecoveryRequired('existing_host_stage_requires_recovery_no_replay')
                self.store.put_job({'request_id': self.job_id, 'request_hash': self.request_hash,
                    'target': self.target_name, 'resources': list(self.resources), 'host_stage': True,
                    'dual_gpu_session': self.dual,
                    'switch_binding': switch_binding(self.target.mode_switch) if self.dual else None,
                    'host_requests': [], 'state': 'waiting', 'error_code': None,
                    'boot_id': self.admission.boot_reader(),
                    'created_at': timestamp(), 'updated_at': timestamp()})
            deadline = time.monotonic() + timeout
            waited = False
            while True:
                # This method atomically checks journal, acquires physical lock,
                # and durably claims ownership under the shared admission mutex.
                self.lease = self.admission.try_acquire(self.job_id, self.request_hash,
                                                       self.resources, self.target_name)
                if self.lease:
                    break
                if not waited:
                    print('[gpu-lock] Waiting for shared admission: ' + self.job_id, file=sys.stderr)
                    waited = True
                if time.monotonic() >= deadline:
                    self.update(state='failed', error_code='resource_acquisition_timeout', host_exit_code=75)
                    print('[gpu-lock] TIMEOUT: ' + self.job_id, file=sys.stderr)
                    return 75
                time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
            self.update(state='running', lease_id=self.lease.owner['lease_id'])
            self.lease.mark('submit_intent')
            if self.resource == 'gpu0':
                # Unknown residency prevents launching another foreground workload.
                used = self.probe()
                if (type(used) not in (int, float) or not math.isfinite(used) or used < 0
                        or used > self.policy['idle_memory_mb']):
                    raise RecoveryRequired('unexpected_gpu0_residency')
            if not input_sources_match(self.input_files):
                self.update(state='failed', error_code='stage_inputs_changed_before_execution', host_exit_code=70)
                self.lease.release({'completion_verified': True, 'cleanup_verified': True,
                                    'policy': 'never_submitted_inputs_changed'})
                return 70
            snapshots = {}
            for role, item in self.input_files.items():
                if item['snapshot']:
                    if item['data'] is None:
                        snapshots[role] = None
                    else:
                        name = 'inputs/' + self.job_id + '/' + role + '.json'
                        self.store.write_bytes(name, item['data'])
                        snapshots[role] = str(self.store.path(name))
            self.update(input_bindings=input_fingerprints(self.input_files))
            if self.dual:
                # Persist intent first; an interrupted/ambiguous switch never restores itself.
                self.lease.mark('switch_intent')
                self.update(mode_phase='switch_intent')
                activated = self.switch.activate()
                if (activated.get('backend_ready') is not True or activated.get('placement_verified') is not True or
                        activated.get('mode') != 'dual' or
                        activated.get('physical_gpus') != ['gpu0', 'gpu1'] or
                        activated.get('switch_binding') != switch_binding(self.target.mode_switch)):
                    raise RecoveryRequired('dual_activation_not_verified')
                self.update(mode_phase='dual_ready', activation_verified=activated)
            if snapshots:
                environment = dict(environment, AIHUB_GPU_STAGE_INPUT_SNAPSHOTS=encode(snapshots).decode())
            environment = dict(environment, AIHUB_GPU_HOST_JOB_ID=self.job_id,
                               AIHUB_GPU_HOST_LEASE_ID=self.lease.owner['lease_id'],
                               AIHUB_GPU_LOCK_HELD_FILE=str(self.admission.lock_paths['gpu1' if self.is_ollama else 'gpu0']),
                               AIHUB_GPU_OLLAMA_TARGET=self.target_name if self.is_ollama else environment.get('AIHUB_GPU_OLLAMA_TARGET', 'ollama'))
            print('[gpu-lock] Acquired shared stage: ' + self.job_id, file=sys.stderr)
            child = subprocess.Popen(argv, env=environment, start_new_session=True, close_fds=True)
            code = child.wait()
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RecoveryRequired('host_descendants_still_active')
            # A negative wait status is interruption, never verified completion.
            if code < 0:
                raise RecoveryRequired('host_child_interrupted')
            if self.is_ollama:
                requests = self.store.job(self.job_id)['host_requests']
                phase = self.admission.snapshot()['owners']['gpu1']['phase']
                if (any(item['state'] != 'completed' for item in requests) or
                        requests and phase != 'host_requests_complete'):
                    raise RecoveryRequired('host_request_unresolved')
            self.lease.mark('cleanup_pending')
            self.update(state='cleanup_pending', host_exit_code=code)
            proof = self._cleanup()
            changed = not input_sources_match(self.input_files)
            if changed:
                code = 70
            self.update(state='success' if code == 0 else 'failed', host_exit_code=code,
                        error_code='stage_inputs_changed_during_execution' if changed else None if code == 0 else 'host_command_failed',
                        cleanup_verified=True, cleanup_proof=proof)
            self.lease.release(proof)
            print('[gpu-lock] Released verified stage: ' + self.job_id, file=sys.stderr)
            return code
        except Interrupted as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if self.lease:
                self.lease.mark('reconciling')
                self.update(state='reconciling', error_code='host_supervisor_interrupted')
            elif self.store.read(self.store.job_name(self.job_id)):
                self.update(state='cancelled', error_code='interrupted_before_admission',
                            host_exit_code=128 + exc.number)
            return 128 + exc.number
        except CleanupPending as exc:
            if self.lease:
                self.lease.mark('cleanup_pending')
                self.update(state='cleanup_pending', error_code=exc.code)
            raise
        except BaseException:
            if self.lease:
                # Never clear an owner after an ambiguous call or storage error.
                try:
                    self.lease.mark('reconciling')
                    self.update(state='reconciling', error_code='host_work_or_cleanup_unresolved')
                except Exception:
                    pass
            raise
        finally:
            if self.lease:
                self.lease.close()
            for number, handler in previous_signals.items():
                signal.signal(number, handler)
            self.instance.close()


class HostOllamaClient:
    """Borrow the already-owned stage; never acquire a nested GPU lock."""
    def __init__(self, admission, target, job_id, lease_id, transport=None):
        self.admission, self.store, self.target = admission, admission.store, target
        self.job_id, self.lease_id = check_id(job_id), lease_id
        self.backend = OllamaBackend(target, transport or HTTPTransport(target.base_url))
        self.mutex = NativeLock(self.store.path('jobs.lock'))

    def owner(self):
        owners = self.admission.snapshot()['owners']
        owner = owners.get('gpu1', {})
        if (owner.get('job_id') != self.job_id or owner.get('lease_id') != self.lease_id
                or owner.get('target') != self.target.name or
                any(owners.get(r) != owner for r in self.target.resources)):
            raise RecoveryRequired('host_stage_context_not_owned')
        return owner

    def generate(self, url, payload, timeout=3600):
        if url.rstrip('/') != self.target.base_url.rstrip('/'):
            raise ValueError('host_ollama_target_mismatch')
        self.backend.validate('generate', payload)
        with self.mutex.held():
            owner = self.owner()
            job = self.store.job(self.job_id)
            if job.get('host_stage') is not True or job['state'] != 'running':
                raise RecoveryRequired('host_stage_not_running')
            if self.target.mode_switch is not None and (job.get('dual_gpu_session') is not True or
                                                       job.get('mode_phase') != 'dual_ready'):
                raise RecoveryRequired('dual_session_activation_not_ready')
            identities = self.admission.snapshot()['lock_identity']
            for resource in self.target.resources:
                physical = NativeLock(self.admission.lock_paths[resource], create=False)
                if not physical.is_held(identities[resource]):
                    raise RecoveryRequired('host_supervisor_lock_not_held')
            requests = job['host_requests']
            if (any(item['state'] != 'completed' for item in requests) or
                    requests and owner['phase'] != 'host_requests_complete'):
                raise RecoveryRequired('previous_host_request_unresolved_no_replay')
            request = {'sequence': len(requests), 'request_hash': hashlib.sha256(encode(payload)).hexdigest(),
                       'model': payload['model'], 'state': 'submit_intent'}
            requests.append(request)
            self.store.put_job(job)
        self.admission._modify(owner, self.target.resources, phase='host_request_submit_intent',
                               detail={'sequence': request['sequence'], 'model': payload['model']})
        # No retry here. A lost reply leaves submit_intent and the stage reserved.
        execution = self.backend.execute('generate', payload,
            time.monotonic() + min(timeout, self.target.execution_timeout), lambda *args: None)
        with self.mutex.held():
            self.owner()
            job = self.store.job(self.job_id)
            current = job['host_requests'][request['sequence']]
            if current != request:
                raise RecoveryRequired('host_request_record_changed')
            current['state'] = 'completed'
            self.store.put_job(job)
        self.admission._modify(owner, self.target.resources, phase='host_requests_complete',
                               detail={'sequence': request['sequence'], 'model': payload['model']})
        return execution.value


def generate_in_host_stage(url, payload, timeout=3600):
    config = os.environ['AIHUB_GPU_RUNNER_CONFIG']
    _, targets, admission = admission_from_config(config)
    target = targets[os.environ.get('AIHUB_GPU_OLLAMA_TARGET', 'ollama')]
    return HostOllamaClient(admission, target, os.environ['AIHUB_GPU_HOST_JOB_ID'],
                            os.environ['AIHUB_GPU_HOST_LEASE_ID']).generate(url, payload, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resource')
    parser.add_argument('label')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if not args.command:
            raise ValueError('command_required')
        if Path(args.command[0]).name == 'whisperx':
            if ('--device_index' not in args.command or
                    args.command[args.command.index('--device_index') + 1] != '0'):
                raise ValueError('whisperx_gpu0_required')
        value, targets, admission = admission_from_config(os.environ['AIHUB_GPU_RUNNER_CONFIG'])
        resource = args.resource
        if resource not in ('gpu0', 'gpu1', 'gpu0+gpu1'):
            matches = [name for name, path in admission.lock_paths.items()
                       if path == Path(resource).resolve()]
            if len(matches) != 1:
                raise ValueError('unconfigured_physical_lock')
            resource = matches[0]
        for name, path in admission.lock_paths.items():
            override = os.environ.get('AIHUB_' + name.upper() + '_LOCK_FILE')
            if override and Path(override).resolve() != path:
                raise ValueError('physical_lock_override_mismatch')
        inputs = capture_input_files(os.environ)
        identity = command_identity(args.command, os.environ, inputs)
        target = targets.get(os.environ.get('AIHUB_GPU_OLLAMA_TARGET', 'ollama'))
        identity_value = {'resource': resource, 'command_identity': identity, 'scope': admission.scope}
        if resource == 'gpu0+gpu1':
            identity_value['ollama_target'] = None if target is None else target.name
        request_hash = hashlib.sha256(encode(identity_value)).hexdigest()
        job_id = os.environ.get('AIHUB_GPU_STAGE_ID') or 'host-' + request_hash[:48]
        stage = HostStage(admission, resource, job_id, request_hash, ollama_target=target,
                         host_policy=value.get('host_stages', {}).get('whisperx'), input_files=inputs)
        return stage.run(args.command, os.environ, float(os.environ.get('AIHUB_GPU_LOCK_TIMEOUT', '3600')))
    except (ValueError, KeyError) as exc:
        print('[gpu-lock] ERROR: host stage configuration/identity required; AIHUB_GPU_LOCK_TIMEOUT must be nonnegative seconds', file=sys.stderr)
        return 64
    except Exception:
        print('[gpu-lock] ERROR: shared admission/host work requires recovery; no direct fallback', file=sys.stderr)
        return 70

if __name__ == '__main__':
    raise SystemExit(main())
