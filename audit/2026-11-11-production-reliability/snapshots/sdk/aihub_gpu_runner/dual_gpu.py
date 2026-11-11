"""Trusted local CDI mode changes under an enclosing two-resource HostStage lease."""
import copy
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import re
import subprocess
import time

from .backends import CleanupPending, HTTPTransport, NvidiaMemoryProbe, Uncertain
from .core import NativeLock, Lease, RecoveryRequired, check_id, encode, strict_json


def switch_binding(policy):
    return hashlib.sha256(encode(policy)).hexdigest()


def validate_policy(policy):
    required = {'docker', 'nvidia_smi', 'project', 'project_directory', 'service', 'container',
                'gpu1_compose', 'dual_compose', 'gpu1_sha256', 'dual_sha256',
                'preserved_config_sha256', 'runtime_contract_sha256', 'image_id', 'compose_version', 'docker_version',
                'ollama_version', 'gpu_uuids', 'idle_memory_mb', 'cleanup_samples',
                'switch_timeout', 'activation_approval_ref'}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError('invalid_dual_switch_policy')
    for key in ('docker', 'nvidia_smi', 'project_directory', 'gpu1_compose', 'dual_compose'):
        if not isinstance(policy[key], str) or not Path(policy[key]).is_absolute():
            raise ValueError('absolute_dual_switch_paths_required')
    if policy['gpu1_compose'] == policy['dual_compose']:
        raise ValueError('distinct_dual_compose_required')
    for key in ('project', 'service', 'container'):
        check_id(policy[key])
    for key in ('gpu1_sha256', 'dual_sha256', 'preserved_config_sha256', 'runtime_contract_sha256'):
        if not isinstance(policy[key], str) or not re.fullmatch(r'[a-f0-9]{64}', policy[key]):
            raise ValueError('dual_hash_binding_required')
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', policy.get('image_id', '')):
        raise ValueError('dual_image_binding_required')
    for key in ('compose_version', 'docker_version', 'ollama_version', 'activation_approval_ref'):
        if not isinstance(policy[key], str) or not policy[key]:
            raise ValueError('dual_version_or_approval_required')
    uuids = policy['gpu_uuids']
    if (not isinstance(uuids, dict) or set(uuids) != {'gpu0', 'gpu1'} or
            any(not isinstance(value, str) or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', value)
                for value in uuids.values()) or len(set(uuids.values())) != 2):
        raise ValueError('physical_gpu_uuid_bindings_required')
    ceilings = policy['idle_memory_mb']
    if (not isinstance(ceilings, dict) or set(ceilings) != {'gpu0', 'gpu1'} or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in ceilings.values()) or type(policy['cleanup_samples']) is not int or
            not 2 <= policy['cleanup_samples'] <= 20 or
            type(policy['switch_timeout']) not in (int, float) or
            not math.isfinite(policy['switch_timeout']) or not 0 < policy['switch_timeout'] <= 600):
        raise ValueError('dual_cleanup_calibration_required')


def preserved_config(config, service):
    value = copy.deepcopy(config)
    value['services'][service].pop('devices', None)
    return hashlib.sha256(encode(value)).hexdigest()


class SessionOnlyBackend:
    """Dual targets cannot become per-request HTTP API switches."""
    def validate(self, *args):
        raise ValueError('dual_target_requires_local_meeting_session')


class ComposeOllamaSwitch:
    def __init__(self, target, *, run=subprocess.run, transport=None, probes=None,
                 poll_interval=.1):
        validate_policy(target.mode_switch)
        self.target, self.policy, self.run = target, target.mode_switch, run
        self.transport = transport or HTTPTransport(target.base_url)
        self.probes = probes or {r: NvidiaMemoryProbe(int(r[-1])) for r in ('gpu0', 'gpu1')}
        self.poll_interval = poll_interval

    def command(self, argv, deadline, data=None):
        try:
            outcome = self.run(argv, input=data, text=True, capture_output=True,
                               timeout=max(.001, deadline - time.monotonic()), check=False)
            if outcome.returncode != 0:
                raise RecoveryRequired('dual_control_command_failed')
            return outcome.stdout.strip()
        except (OSError, subprocess.SubprocessError, RecoveryRequired):
            # Child output can contain resolved private config; never echo it.
            raise RecoveryRequired('dual_control_command_or_response_uncertain') from None

    def compose_argv(self, file):
        p = self.policy
        return [p['docker'], 'compose', '-p', p['project'], '--project-directory',
                p['project_directory'], '-f', file]

    def render(self, mode, deadline):
        p = self.policy
        raw = Path(p[mode + '_compose']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != p[mode + '_sha256']:
            raise RecoveryRequired('dual_compose_binding_changed')
        config = strict_json(self.command(self.compose_argv('-') + ['config', '--format', 'json'],
                                         deadline, raw.decode('utf-8')))
        service = config['services'][p['service']]
        indices = ('1',) if mode == 'gpu1' else ('0', '1')
        devices = [{'source': 'nvidia.com/gpu=' + i, 'target': 'nvidia.com/gpu=' + i,
                    'permissions': 'rwm'} for i in indices]
        if (service.get('devices') != devices or service.get('build') or
                preserved_config(config, p['service']) != p['preserved_config_sha256'] or
                service.get('container_name') != p['container'] or
                service.get('privileged', False) or service.get('pre_start') or service.get('post_start')):
            raise RecoveryRequired('dual_unrelated_configuration_changed')
        env = service.get('environment', {})
        if any(key in env for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                     'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')):
            raise RecoveryRequired('dual_restrictive_device_environment_requires_review')
        if (str(env.get('OLLAMA_FLASH_ATTENTION', '')).lower() not in ('1', 'true', 'yes') or
                str(env.get('OLLAMA_KV_CACHE_TYPE', '')).lower() != 'q8_0'):
            raise RecoveryRequired('dual_q8_flash_attention_changed')
        # Digest/ID pin and locally available image ID prevent implicit upgrades.
        if not re.fullmatch(r'(?:.+@)?sha256:[a-f0-9]{64}', service.get('image', '')):
            raise RecoveryRequired('dual_digest_image_required')
        image = self.command([p['docker'], 'image', 'inspect', '--format', '{{.Id}}',
                              service['image']], deadline)
        if image != p['image_id']:
            raise RecoveryRequired('dual_local_image_binding_changed')
        return config

    def bindings(self, deadline):
        p = self.policy
        if (self.command([p['docker'], 'compose', 'version', '--short'], deadline) != p['compose_version'] or
                self.command([p['docker'], 'version', '--format', '{{.Server.Version}}'], deadline) != p['docker_version']):
            raise RecoveryRequired('dual_installed_version_changed')
        self.render('gpu1', deadline)
        self.render('dual', deadline)

    def observe(self, mode, deadline):
        p = self.policy
        template = ('{"running":{{json .State.Running}},"image":{{json .Image}},'
                    '"runtime":{{json .HostConfig.Runtime}},"privileged":{{json .HostConfig.Privileged}},'
                    '"requests":{{json .HostConfig.DeviceRequests}},"devices":{{json .HostConfig.Devices}},'
                    '"rules":{{json .HostConfig.DeviceCgroupRules}},'
                    '"preserved":{"mounts":{{json .Mounts}},"ports":{{json .NetworkSettings.Ports}},'
                    '"restart":{{json .HostConfig.RestartPolicy}},'
                    '"flash_attention":"{{range .Config.Env}}{{if eq (index (split . "=") 0) "OLLAMA_FLASH_ATTENTION"}}{{index (split . "=") 1}}{{end}}{{end}}",'
                    '"kv_cache":"{{range .Config.Env}}{{if eq (index (split . "=") 0) "OLLAMA_KV_CACHE_TYPE"}}{{index (split . "=") 1}}{{end}}{{end}}"}}')
        meta = strict_json(self.command([p['docker'], 'inspect', '--format', template, p['container']], deadline))
        contract = meta.get('preserved', {})
        if (str(contract.get('flash_attention', '')).lower() not in ('1', 'true', 'yes') or
                str(contract.get('kv_cache', '')).lower() != 'q8_0'):
            raise RecoveryRequired('dual_running_q8_flash_attention_changed')
        indices = ('1',) if mode == 'gpu1' else ('0', '1')
        requests = meta.get('requests') or []
        if (meta.get('running') is not True or meta.get('image') != p['image_id'] or
                meta.get('runtime') != 'runc' or meta.get('privileged') is not False or
                meta.get('devices') or meta.get('rules') or len(requests) != 1 or
                requests[0].get('Count') != 0 or
                set(requests[0].get('DeviceIDs') or []) != {'nvidia.com/gpu=' + i for i in indices} or
                hashlib.sha256(encode(meta.get('preserved'))).hexdigest() != p['runtime_contract_sha256']):
            raise RecoveryRequired('dual_running_container_binding_changed')
        host = self.command([p['nvidia_smi'], '--query-gpu=index,uuid', '--format=csv,noheader'], deadline)
        mapping = {row[0].strip(): row[1].strip() for row in csv.reader(io.StringIO(host))}
        if any(mapping.get(str(i)) != p['gpu_uuids']['gpu' + str(i)] for i in (0, 1)):
            raise RecoveryRequired('dual_physical_gpu_mapping_changed')
        visible = self.command([p['docker'], 'exec', p['container'], 'nvidia-smi',
                                '--query-gpu=index,uuid', '--format=csv,noheader'], deadline)
        rows = list(csv.reader(io.StringIO(visible)))
        if (len(rows) != len(indices) or {row[1].strip() for row in rows} !=
                {p['gpu_uuids']['gpu' + i] for i in indices}):
            raise RecoveryRequired('dual_physical_exposure_changed')
        version = self.transport.json('GET', '/api/version', timeout=max(.001, deadline - time.monotonic()))
        residency = self.transport.json('GET', '/api/ps', timeout=max(.001, deadline - time.monotonic()))
        if version.get('version') != p['ollama_version']:
            raise RecoveryRequired('dual_ollama_version_changed')
        if residency.get('models') != []:
            raise CleanupPending('dual_backend_not_ready_or_resident')
        used = {r: self.probes[r]() for r in ('gpu0', 'gpu1')}
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or
               value > p['idle_memory_mb'][r] for r, value in used.items()):
            raise CleanupPending('dual_gpu_memory_not_quiescent')
        return {'mode': mode, 'image_id': p['image_id'], 'physical_gpus': ['gpu' + i for i in indices],
                'backend_ready': True, 'placement_verified': True, 'memory_mb': used,
                'switch_binding': switch_binding(p)}

    def verify(self, mode):
        deadline = time.monotonic() + self.policy['switch_timeout']
        self.bindings(deadline)
        stable = 0
        while time.monotonic() < deadline:
            try:
                proof = self.observe(mode, deadline)
                stable += 1
                if stable >= self.policy['cleanup_samples']:
                    return dict(proof, stable_samples=stable)
            except (CleanupPending, Uncertain, OSError):
                stable = 0
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
        raise CleanupPending('dual_readiness_or_cleanup_unconfirmed')

    def switch(self, mode):
        # One bounded recreation, never a retry or automatic rollback on error.
        deadline = time.monotonic() + self.policy['switch_timeout']
        self.bindings(deadline)
        config = self.render(mode, deadline)
        self.command(self.compose_argv('-') + ['up', '-d', '--no-deps', '--no-build', '--pull', 'never',
                     '--force-recreate', '--wait', '--wait-timeout', str(int(self.policy['switch_timeout'])),
                     self.policy['service']], deadline, encode(config).decode())
        return self.verify(mode)

    def activate(self):
        self.verify('gpu1')
        return self.switch('dual')

    def restore(self):
        self.verify('dual')
        return dict(self.switch('gpu1'), gpu1_restored=True)


def restore_held(admission, target, job_id, evidence, controller=None):
    """Explicit local operator action; restore verified quiescent work, retain owners."""
    check_id(job_id)
    job_mutex = NativeLock(admission.store.path('jobs.lock'))
    locks = []
    with admission.mutex.held():
        owners = admission.snapshot()['owners']
        job = admission.store.job(job_id)
        owner = owners.get('gpu1', {})
        if (job.get('dual_gpu_session') is not True or job.get('switch_binding') != switch_binding(target.mode_switch) or
                target.resources != ('gpu0', 'gpu1') or
                any(owners.get(r) != owner for r in ('gpu0', 'gpu1')) or
                not evidence.get('approval_ref') or evidence.get('operator_approved_dual_restoration') is not True or
                evidence.get('backend_work_resolved') is not True or
                evidence.get('switch_binding') != job['switch_binding'] or
                any(evidence.get(k) != owner.get(k) for k in ('job_id', 'request_hash', 'lease_id', 'target')) or
                owner.get('job_id') != job_id or target.name != owner.get('target') or
                evidence.get('observed_mode') not in ('gpu1', 'dual')):
            raise ValueError('invalid_dual_restoration_evidence')
        try:
            for r in ('gpu0', 'gpu1'):
                from .core import lock_identity
                path = admission.lock_paths[r]
                if lock_identity(path.stat()) != admission.snapshot()['lock_identity'][r]:
                    raise RecoveryRequired('physical_lock_identity_changed')
                lock = NativeLock(path, create=False)
                if not lock.acquire(0):
                    raise RecoveryRequired('dual_supervisor_still_active')
                locks.append(lock)
                if lock_identity(os.fstat(lock.fd)) != admission.snapshot()['lock_identity'][r]:
                    raise RecoveryRequired('physical_lock_identity_changed_during_acquisition')
        except BaseException:
            for lock in reversed(locks):
                lock.close()
            raise
    lease = Lease(admission, owner, ('gpu0', 'gpu1'), locks)
    try:
        controller = controller or ComposeOllamaSwitch(target)
        # Both observations and any restore run while original owners/native locks remain.
        lease.mark('operator_restore_intent')
        with job_mutex.held():
            job = admission.store.job(job_id)
            job.pop('dual_restoration_verified', None)
            admission.store.put_job(job)
        if evidence['observed_mode'] == 'dual':
            proof = controller.restore()
        else:
            proof = dict(controller.verify('gpu1'), gpu1_restored=True)
        with job_mutex.held():
            job = admission.store.job(job_id)
            proof = dict(proof, job_id=job_id, lease_id=owner['lease_id'])
            if (any(proof.get(k) is not True for k in ('gpu1_restored', 'backend_ready', 'placement_verified')) or
                    proof.get('mode') != 'gpu1' or proof.get('physical_gpus') != ['gpu1'] or
                    proof.get('switch_binding') != job['switch_binding']):
                raise RecoveryRequired('dual_manual_restoration_not_verified')
            job.update(dual_restoration_verified=proof, state='cleanup_pending',
                       restoration_approval_ref=evidence['approval_ref'])
            admission.store.put_job(job)
        lease.mark('cleanup_pending')
        return proof
    finally:
        # Manual recovery is separately reviewed; restoration alone never clears owners.
        lease.close()
