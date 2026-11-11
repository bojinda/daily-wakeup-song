"""Owned, exact-container wallpaper lifecycle. No inference retries or scheduler."""
import math
import time

from .backends import CleanupPending, Uncertain
from .core import FINAL, RecoveryRequired, encode, timestamp
from .recovery import PersistenceUncertain, binding, digest, now, remaining, seconds, valid_boot


class WallpaperLifecycle:
    def __init__(self, manager):
        self.m, self.a, self.store = manager, manager.a, manager.store
        self.target, self.observer = manager.targets['wallpaper'], manager.observer

    def identity(self, owner, resources):
        return dict(binding(owner), resources=list(resources), policy_id=self.m.policy_id,
                    container_id=self.m.p['wallpaper'], physical_gpu_uuids={
                        r: self.m.p['gpu_uuids'][r] for r in resources})

    def bound(self, owner, resources):
        state = self.a.snapshot()
        self.a._check_physical_identities(state)
        if (tuple(resources) != self.target.resources or not valid_boot(owner.get('boot_id')) or
                owner['boot_id'] != self.a.boot_reader() or
                any(binding(state['owners'].get(r, {})) != binding(owner) for r in resources) or
                {r for r, o in state['owners'].items() if o['lease_id'] == owner['lease_id']} != set(resources)):
            raise RecoveryRequired('wallpaper_lifecycle_complete_owner_binding_required')
        job = self.store.job(owner['job_id'])
        if job['request_hash'] != owner['request_hash'] or job['target'] != 'wallpaper':
            raise RecoveryRequired('wallpaper_lifecycle_job_binding_changed')
        return job

    def record(self, owner, resources):
        job = self.bound(owner, resources)
        record = job.get('wallpaper_lifecycle')
        if not isinstance(record, dict) or record.get('identity') != self.identity(owner, resources):
            raise RecoveryRequired('wallpaper_lifecycle_identity_missing_or_changed')
        if record.get('phase') not in ('prepared', 'starting', 'ready', 'submit_intent', 'terminal',
                                        'artifact_saved', 'stopping', 'stopped', 'verified'):
            raise RecoveryRequired('wallpaper_lifecycle_phase_unverified')
        if record.get('submission_intent'):
            value = job.get('execution_deadline')
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise RecoveryRequired('execution_deadline_missing_or_invalid')
        return record

    def save(self, job_id, **fields):
        job = self.store.job(job_id)
        record = dict(job['wallpaper_lifecycle'], **fields)
        self.m.update_job(job_id, wallpaper_lifecycle=record)
        return record

    @staticmethod
    def deadline(record, field, budget):
        value = record.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RecoveryRequired(field + '_missing_or_invalid')
        left = value - now()
        if left <= 0:
            raise Uncertain(field + '_expired')
        return time.monotonic() + min(left, budget)

    def start(self, owner, resources):
        job_id = owner['job_id']
        if self.bound(owner, resources).get('wallpaper_lifecycle'):
            raise RecoveryRequired('wallpaper_lifecycle_no_start_replay')
        deadline = time.monotonic() + self.target.startup_timeout
        expiry = now() + self.target.startup_timeout
        current = self.observer.container('wallpaper', deadline)
        # A running, unowned generation is never adopted or stopped implicitly.
        if current['running'] or current['pid'] != 0:
            raise RecoveryRequired('wallpaper_must_be_stopped_before_new_job')
        record = {'identity': self.identity(owner, resources), 'phase': 'prepared',
                  'original_container': current, 'created_at': timestamp(),
                  'startup_deadline': expiry}
        self.m.update_job(job_id, wallpaper_lifecycle=record, boot_id=owner['boot_id'])
        self.observer.stopped_ready(resources, deadline)
        self.save(job_id, phase='starting', start_intent=timestamp())
        try:
            self.observer.start(current, deadline)
        except Uncertain:
            self.save(job_id, start_response_uncertain=True)
        current = self.m.running(current, deadline, require_new=True)
        self.save(job_id, running_container=current,
                  original_pids=self.observer.processes(current, deadline))
        placement = self.observer.placement(current, resources, deadline)
        proof = self.observer.ready(resources, deadline)
        if self.observer.container('wallpaper', deadline) != current:
            raise RecoveryRequired('wallpaper_changed_during_startup_readiness')
        remaining(deadline)
        self.save(job_id, phase='ready', startup_proof=proof, placement=placement)
        # Execution gets its full separate budget only after startup is verified.
        self.m.update_job(job_id, execution_deadline=now() + self.target.execution_timeout,
            execution_container=current, execution_processes=self.store.job(job_id)['wallpaper_lifecycle']['original_pids'])

    def submit_intent(self, owner, resources):
        record = self.record(owner, resources)
        if record['phase'] != 'ready':
            raise RecoveryRequired('wallpaper_submission_not_new_and_ready')
        self.save(owner['job_id'], phase='submit_intent', submission_intent=timestamp())

    def run(self, runner, job, lease):
        owner, resources, job_id = lease.owner, lease.resources, job['request_id']
        self.start(owner, resources)
        self.submit_intent(owner, resources)
        lease.mark('submit_intent')
        def progress(phase, detail=None):
            lease.mark(phase, detail)
            runner._update(job_id, backend_detail=detail or {}, phase=phase)
        deadline = self.deadline(self.store.job(job_id), 'execution_deadline', self.target.execution_timeout)
        execution = self.m.backends['wallpaper'].execute(job['operation'], job['payload'], deadline, progress)
        runner._update(job_id, state='cleanup_pending', completion_verified=True,
                       backend_detail=execution.handle, owned_execution_failed=bool(execution.error_code))
        lease.mark('cleanup_pending', execution.handle)
        self.persist(job_id, execution)
        if execution.error_code:
            raise Uncertain('wallpaper_owned_execution_failed')
        self.complete(owner, resources)

    def persist(self, job_id, execution):
        if execution.error_code == 'artifact_retrieval_or_validation_failed':
            raise Uncertain('wallpaper_artifact_unavailable_keep_backend')
        result = {'value': execution.value, 'media_type': execution.media_type,
                  'artifact_available': execution.artifact is not None}
        if not execution.error_code and execution.artifact is None:
            raise RecoveryRequired('wallpaper_success_artifact_required')
        self.save(job_id, phase='terminal', result=result, result_sha256=digest(result),
            artifact_sha256=None if execution.artifact is None else digest_bytes(execution.artifact),
            execution_error=execution.error_code, prompt=execution.handle.get('prompt_id'))
        try:
            if execution.artifact is not None:
                self.store.write_bytes('results/' + job_id + '.bin', execution.artifact)
            self.store.write('results/' + job_id + '.json', result)
        except OSError as exc:
            raise PersistenceUncertain() from exc
        self.verify_result(job_id, self.store.job(job_id)['wallpaper_lifecycle'])
        self.save(job_id, phase='artifact_saved')

    def verify_result(self, job_id, record):
        result = self.store.read('results/' + job_id + '.json')
        if result is None or digest(result) != record.get('result_sha256') or result != record.get('result'):
            raise RecoveryRequired('wallpaper_durable_result_unverified')
        expected = record.get('artifact_sha256')
        if result.get('artifact_available'):
            try:
                actual = digest_bytes(self.store.path('results/' + job_id + '.bin').read_bytes())
            except FileNotFoundError:
                actual = None
            if not expected or actual != expected:
                raise RecoveryRequired('wallpaper_durable_artifact_unverified')
        elif expected is not None or not record.get('execution_error'):
            raise RecoveryRequired('wallpaper_missing_success_artifact')
        return result

    def shutdown(self, owner, resources, deadline, incident=None):
        job_id = owner['job_id']
        record = self.record(owner, resources)
        self.verify_result(job_id, record)  # Never destroy the only available output.
        original = record.get('running_container') or record['original_container']
        current = self.observer.container('wallpaper', deadline)
        if any(current[k] != original[k] for k in ('id', 'started_at', 'contract_sha256')):
            raise RecoveryRequired('wallpaper_generation_changed_before_shutdown')
        if current['running'] and current != original:
            raise RecoveryRequired('wallpaper_process_identity_changed_before_shutdown')
        if current['running'] and not record.get('term_intent'):
            prompt = record.get('prompt')
            # Terminal history/output are already durable. Comfy may briefly
            # retain that same prompt in its queue while its worker unwinds;
            # only that exact prompt is allowed, never unrelated queued work.
            self.observer.queues(resources, deadline, prompt=prompt, teardown=True)
            self.observer.gpu_sample(resources, deadline, executing=True)
            if incident is not None:
                incident = self.m.reserve_attempt(incident)
            pids = dict(record.get('original_pids', {}), **self.observer.processes(current, deadline))
            record = self.save(job_id, phase='stopping', term_intent=timestamp(), original_pids=pids)
            try:
                self.observer.signal(original, 'TERM', deadline)
            except Uncertain:
                record = self.save(job_id, term_response_uncertain=True)
        while True:
            current = self.observer.container('wallpaper', deadline)
            if any(current[k] != original[k] for k in ('id', 'started_at', 'contract_sha256')):
                raise RecoveryRequired('wallpaper_generation_changed_during_shutdown')
            if not current['running'] and current['pid'] == 0:
                break
            if not record.get('term_intent'):
                raise RecoveryRequired('wallpaper_shutdown_intent_missing')
            if now() >= seconds(record['term_intent']) + self.m.p['grace_seconds']:
                if incident is None:
                    raise CleanupPending('wallpaper_shutdown_grace_expired')
                if record.get('kill_intent'):
                    time.sleep(min(.1, remaining(deadline)))
                    continue  # Observe one ambiguous KILL; never issue it again.
                if not self.m.p['allow_force']:
                    raise RecoveryRequired('wallpaper_exit_unverified_no_more_termination_attempts')
                self.m.reserve_attempt(incident)
                record = self.save(job_id, kill_intent=timestamp())
                try:
                    self.observer.signal(original, 'KILL', deadline)
                except Uncertain:
                    self.save(job_id, kill_response_uncertain=True)
            time.sleep(min(.1, remaining(deadline)))
        self.save(job_id, phase='stopped', stopped_container=current)
        proof = self.observer.stopped_ready(resources, deadline, record.get('original_pids', {}))
        self.save(job_id, phase='verified', cleanup_proof=proof)
        return proof

    def complete(self, owner, resources, incident=None, deadline=None):
        job_id = owner['job_id']
        record = self.record(owner, resources)
        if deadline is None:
            if 'shutdown_deadline' not in record:
                record = self.save(job_id, shutdown_deadline=now() + self.target.shutdown_timeout)
            deadline = self.deadline(record, 'shutdown_deadline', self.target.shutdown_timeout)
        proof = self.shutdown(owner, resources, deadline, incident)
        record = self.record(owner, resources)
        self.verify_result(job_id, record)
        receipt = {'identity': record['identity'], 'result_sha256': record['result_sha256'],
                   'artifact_sha256': record['artifact_sha256'], 'execution_error': record.get('execution_error'),
                   'proof': proof, 'released_at': timestamp()}
        if proof.get('idle_ready') is not True or proof.get('cleanup_verified') is not True:
            raise RecoveryRequired('wallpaper_stopped_cleanup_proof_required')
        if self.observer.container('wallpaper', deadline) != proof['stopped_container']:
            raise RecoveryRequired('wallpaper_changed_before_lifecycle_handoff')
        remaining(deadline)
        def release(state):
            self.a._check_physical_identities(state)
            if any(binding(state['owners'].get(r, {})) != binding(owner) for r in resources):
                raise RecoveryRequired('wallpaper_owner_changed_before_release')
            receipts = state.setdefault('wallpaper_lifecycle_receipts', {})
            if owner['lease_id'] in receipts:
                raise RecoveryRequired('wallpaper_lifecycle_receipt_already_exists_with_owner')
            receipts[owner['lease_id']] = receipt
            if incident is not None:
                current = state['automatic_recovery'][incident['id']]
                state['automatic_recovery'][incident['id']] = dict(current, phase='released', proof=proof,
                                                                  released_at=receipt['released_at'])
                if not any(r.get('incident_id') == incident['id'] for r in state.setdefault('recoveries', [])):
                    state['recoveries'].append({'incident_id': incident['id'], 'identity': incident['identity'],
                        'container': record.get('running_container'), 'observed_boot': self.a.boot_reader(),
                        'proof': proof, 'recovered_at': receipt['released_at'], 'provenance': 'runner_observations'})
            for r in resources:
                del state['owners'][r]
        self.m.change(release)  # Receipt + paired handoff + optional recovery audit are atomic.
        self.finalize_released()

    def finalize_released(self):
        state = self.a.snapshot()
        for lease_id, receipt in state.get('wallpaper_lifecycle_receipts', {}).items():
            identity = receipt['identity']
            job_id = identity['job_id']
            job = self.store.job(job_id)
            record = job.get('wallpaper_lifecycle', {})
            if record.get('phase') == 'released' and job['state'] in FINAL:
                continue
            if (record.get('identity') != identity or identity['lease_id'] != lease_id or
                    job['request_hash'] != identity['request_hash'] or
                    any(o['job_id'] == job_id for o in state['owners'].values()) or
                    record.get('result_sha256') != receipt['result_sha256'] or
                    record.get('artifact_sha256') != receipt['artifact_sha256']):
                raise RecoveryRequired('wallpaper_lifecycle_release_receipt_changed')
            self.verify_result(job_id, record)
            record = dict(record, phase='released')
            self.m.update_job(job_id, wallpaper_lifecycle=record,
                state='failed' if receipt['execution_error'] else 'success', error_code=receipt['execution_error'],
                cleanup_verified=True, completion_verified=True, cleanup_proof=receipt['proof'],
                **({'recovery_status': 'verified'} if job.get('recovery_status') else {}))

    def resume_normal(self, owner, resources, job):
        record = job.get('wallpaper_lifecycle', {})
        if (record.get('phase') not in ('artifact_saved', 'stopping', 'stopped', 'verified') or
                record.get('execution_error') or job.get('lifecycle_failure') or job.get('lifecycle_operator_error')):
            return False
        try:
            # Resume only durable completed output. Never resume inference/start.
            self.complete(owner, resources)
            return True
        except (CleanupPending, Uncertain, RecoveryRequired) as exc:
            self.m.update_job(owner['job_id'], lifecycle_failure=str(exc))
            return False

    def recover(self, incident, owner, resources, job, deadline):
        if job.get('lifecycle_operator_error'):
            raise RecoveryRequired(job['lifecycle_operator_error'])
        record = self.record(owner, resources)
        job_id = owner['job_id']
        if record.get('start_intent') and not record.get('running_container'):
            # Ambiguous start is observed, never repeated, even before submission.
            current = self.m.running(record['original_container'], deadline, require_new=True)
            record = self.save(job_id, running_container=current,
                               original_pids=self.observer.processes(current, deadline))
        prompt = record.get('prompt') or job.get('backend_detail', {}).get('prompt_id')
        def publication_attempt():
            nonlocal incident
            if incident.get('publication_attempts', 0) >= 2:
                raise RecoveryRequired('wallpaper_artifact_persistence_unresolved_keep_backend')
            incident = self.m.save(incident, publication_attempts=incident.get('publication_attempts', 0) + 1)
        if record.get('phase') == 'terminal':
            # Finish an interrupted artifact/json commit when the bytes survived.
            artifact = record.get('artifact_sha256')
            path = self.store.path('results/' + job_id + '.bin')
            if artifact is None or path.is_file() and digest_bytes(path.read_bytes()) == artifact:
                publication_attempt()
                try:
                    self.store.write('results/' + job_id + '.json', record['result'])
                except OSError as exc:
                    raise PersistenceUncertain() from exc
                self.verify_result(job_id, record)
                record = self.save(job_id, phase='artifact_saved')
        saved = record.get('phase') in ('artifact_saved', 'stopping', 'stopped', 'verified')
        if not saved and prompt and incident.get('history_attempts', 0) < 2:
            incident = self.m.save(incident, history_attempts=incident.get('history_attempts', 0) + 1)
            try:
                # History/artifact HTTP may use at most a quarter of the
                # remaining attempt (and 5s). Keep the original deadline for
                # exact-container termination and stopped-GPU verification.
                history_deadline = time.monotonic() + min(5, remaining(deadline) / 4)
                execution = self.m.backends['wallpaper'].reconcile(prompt, history_deadline)
            except Uncertain:
                execution = None
            if execution is not None:
                publication_attempt()
                self.persist(job_id, execution)
                record = self.record(owner, resources)
                saved = True
        if not saved:
            if record.get('phase') == 'terminal':
                raise RecoveryRequired('wallpaper_artifact_persistence_unresolved_keep_backend')
            # No submission means no inference to replay. A started execution is
            # fenced by its own deadline; API failures alone authorize no stop.
            field = 'execution_deadline' if record.get('submission_intent') else 'startup_deadline'
            source = job if field == 'execution_deadline' else record
            try:
                self.deadline(source, field, self.target.execution_timeout if field == 'execution_deadline'
                              else self.target.startup_timeout)
            except Uncertain:
                pass  # Verified expiry, not an API timeout, permits failure cleanup.
            else:
                if not job.get('owned_execution_failed'):
                    self.m.save(incident, phase='waiting_execution_deadline')
                    return
            from .backends import Execution
            self.persist(job_id, Execution({}, {'prompt_id': prompt},
                error_code='wallpaper_interrupted_by_recovery'))
        self.complete(owner, resources, incident, deadline)


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()
