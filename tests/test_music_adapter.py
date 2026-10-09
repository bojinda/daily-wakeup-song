"""Music adapter tests: native temp state/locks, mocked backends, Flask in process."""
import concurrent.futures
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from aihub_gpu_runner.api import Application
from aihub_gpu_runner.backends import ComfyBackend, OllamaBackend
from aihub_gpu_runner.client import JobClient, PendingJob, CURRENT_OPERATION, publish_bytes
from aihub_gpu_runner.config import admission_from_config
from aihub_gpu_runner.core import NativeLock, Runner, encode
from test_runner_adapter import InProcessAPI, OllamaTransport, ComfyTransport, TOKEN

ROOT = Path(__file__).resolve().parents[1]
SONG = b'ID3synthetic-new-mp3-audio'
OLD = b'ID3synthetic-current-playable-song'
READY = b'ID3synthetic-ready-for-promotion'


class ChatTransport(OllamaTransport):
    def __init__(self, event=None, malformed=False, fail=False):
        super().__init__(fail=fail)
        self.event, self.malformed = event, malformed
    def json(self, method, route, payload=None, timeout=5):
        if route == '/api/chat':
            self.generations += 1
            self.payloads.append(copy.deepcopy(payload))
            if self.event is not None and not self.event.wait(3):
                raise OSError('synthetic chat blocked')
            if self.fail:
                raise OSError('synthetic lost chat reply')
            content = 'malformed reply' if self.malformed else json.dumps({'tags': 'synthetic production tags', 'lyrics': '[verse]\nGood morning synthetic lyrics'})
            return {'done': True, 'message': {'content': content}}
        return super().json(method, route, payload, timeout)


class AudioTransport(ComfyTransport):
    def __init__(self, cleanup_event=None, variant='valid'):
        super().__init__(cleanup_event=cleanup_event)
        self.variant = variant
    def json(self, method, route, payload=None, timeout=5):
        if route == '/history/owned-prompt':
            self.history_reads += 1
            descriptor = {'filename': 'synthetic.mp3', 'subfolder': 'audio', 'type': 'output'}
            outputs = {'104': {'audio': [descriptor]}, 'wrong-node': {'audio': [{'filename': 'wrong.mp3'}]}}
            if self.variant == 'missing':
                outputs = {'wrong-node': outputs['wrong-node']}
            if self.variant == 'wav':
                descriptor['filename'] = 'synthetic.wav'
            if self.variant == 'input':
                descriptor['type'] = 'input'
            if self.variant == 'error':
                return {'owned-prompt': {'status': {'completed': False, 'status_str': 'error'}, 'outputs': outputs}}
            if self.variant == 'partial' and self.history_reads == 1:
                return {'owned-prompt': {'status': {'completed': False, 'status_str': 'running'}, 'outputs': outputs}}
            return {'owned-prompt': {'status': {'completed': True, 'status_str': 'success'}, 'outputs': outputs}}
        return super().json(method, route, payload, timeout)
    def binary(self, route, params, timeout=5):
        self.views.append(params)
        return (b'not-an-mp3' if self.variant == 'bytes' else SONG,
                'audio/wav' if self.variant == 'mime' else 'audio/mpeg')


def load_app(name='music_test'):
    sys.modules.pop('gpu_runner', None)
    spec = importlib.util.spec_from_file_location(name, ROOT / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MusicAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='music-adapter-')
        self.root = Path(self.temp.name)
        self.latest, self.archive = self.root / 'latest', self.root / 'archive'
        self.latest.mkdir()
        self.latest_file, self.new_file = self.latest / 'wakeup-latest.mp3', self.latest / 'wakeup-new.mp3'
        self.latest_file.write_bytes(OLD)
        self.new_file.write_bytes(READY)
        self.environment = patch.dict(os.environ, {'LATEST_DIR': str(self.latest), 'ARCHIVE_DIR': str(self.archive),
            'GENRE_PACKS_PATH': str(ROOT / 'genre_packs.json'), 'WORKFLOW_PATH': str(ROOT / 'workflow/wakeup_api.json'),
            'OLLAMA_MODEL': 'synthetic', 'TARGET_SECONDS': '135', 'GENRE_PICK_MODE': 'daily',
            'GPU_RUNNER_CLIENT_STATE': str(self.root / 'client-state')})
        self.environment.start()
        sys.path.insert(0, str(ROOT))
        self.app = load_app()
        self.workflow = copy.deepcopy(self.app.WORKFLOW_TEMPLATE)
        self.policy = json.loads(Path(os.environ['MUSIC_POLICY_EXAMPLE']).read_text())
        template = self.root / 'music.json'
        template.write_bytes(encode(self.workflow))
        wallpaper = self.root / 'wallpaper.json'
        self.wallpaper_workflow = {'7': {'class_type': 'Seed', 'inputs': {'seed': 1}},
                                  '36': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'synthetic'}}}
        wallpaper.write_bytes(encode(self.wallpaper_workflow))
        self.paths = {key: self.root / (key + '.lock') for key in ('gpu0', 'gpu1')}
        for path in self.paths.values():
            path.touch()
        config = self.root / 'runner.synthetic.json'
        music_target = copy.deepcopy(self.policy)
        music_target.update(base_url='http://127.0.0.1:2', workflow_template=str(template),
                            workflow_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                            idle_memory_mb=0, cleanup_timeout=.15, execution_timeout=2, acquisition_timeout=2)
        config.write_text(json.dumps({'state_dir': str(self.root / 'runner-state'),
            'lock_paths': {key: str(path) for key, path in self.paths.items()},
            'host_stages': {'whisperx': {'idle_memory_mb': 0, 'cleanup_samples': 2, 'cleanup_timeout': 1}},
            'targets': {'ollama': {'kind': 'ollama', 'resources': ['gpu1'], 'base_url': 'http://127.0.0.1:1',
                                   'models': ['synthetic'], 'acquisition_timeout': 2, 'cleanup_timeout': 1},
                        'music': music_target,
                        'wallpaper': {'kind': 'comfy', 'resources': ['gpu0'], 'base_url': 'http://127.0.0.1:3',
                            'workflow_template': str(wallpaper), 'workflow_sha256': hashlib.sha256(wallpaper.read_bytes()).hexdigest(),
                            'mutations': {'7.seed': {'type': 'integer'}}, 'output_node': '36',
                            'idle_memory_mb': 0, 'cleanup_samples': 2}}}))
        _, self.targets, self.admission = admission_from_config(config)
        self.config = config
        self.admission.bootstrap({'approval_ref': 'SYNTHETIC ONLY', 'bootstrap_id': self.admission.snapshot()['bootstrap_id'],
                                  'backend_quiescent': True, 'cleanup_verified': True})
        self.chat, self.audio, self.wallpaper = ChatTransport(), AudioTransport(), ComfyTransport()
        self.memory_high = False
        self.runner = Runner(self.admission, self.targets,
            {'ollama': OllamaBackend(self.targets['ollama'], self.chat, poll_interval=.005),
             'music': ComfyBackend(self.targets['music'], self.audio, probe=self.probe, poll_interval=.005),
             'wallpaper': ComfyBackend(self.targets['wallpaper'], self.wallpaper, probe=lambda: 0, poll_interval=.005)},
            poll_interval=.005)
        self.transport = InProcessAPI(Application(self.runner, TOKEN))
        self.client = JobClient(self.transport, wait_timeout=2, poll_interval=.005)
        self.factory = patch.object(JobClient, 'from_environment', return_value=self.client)
        self.factory.start()
        self.http = [patch('requests.post', side_effect=AssertionError('live generation forbidden')),
                     patch('requests.get', side_effect=AssertionError('live metadata/download forbidden'))]
        for block in self.http:
            block.start()
        self.processes, self.events, self.job_ids = [], [], []
    def probe(self):
        return 1000 if self.memory_high and self.audio.frees else 0
    def tearDown(self):
        for event in self.events:
            event.set()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=8)
        # Let synthetic background workers finish before removing their private state.
        for job_id in self.job_ids:
            self.wait_job(job_id, fail=False)
        self.runner.close()
        for block in self.http:
            block.stop()
        self.factory.stop()
        self.environment.stop()
        sys.path.pop(0)
        self.temp.cleanup()
    def post(self, job_id='wake-operation'):
        response = self.app.app.test_client().post('/swap_and_generate', headers={'X-Request-ID': job_id})
        if response.status_code == 200 and response.json.get('status') == 'started':
            self.job_ids.append(job_id)
        return response
    def wait_job(self, job_id, fail=True):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            job = self.app._get_job(job_id)
            record = self.app._output_slot().operation_status(job_id)
            if job.get('status') in ('done', 'error') and record and record['status'] in ('ok', 'error'):
                probe = NativeLock(self.app._output_slot().lock_path)
                if probe.acquire(0):
                    probe.close()
                    return job
            time.sleep(.005)
        if fail:
            self.fail('synthetic music worker did not finish')
        return self.app._get_job(job_id)
    def wait_state(self, job_id, state, step=None):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = self.app._get_job(job_id)
            if job.get('runner_state') == state and (step is None or
                    job.get('runner_job_id') == JobClient.stage_id(job_id, step)):
                return
            time.sleep(.005)
        self.fail('synthetic runner state not observed: ' + state)
    def test_promote_archive_and_respond_before_gpu1_wait(self):
        lease = self.admission.try_acquire('meeting-summary', 'hash', ('gpu1',), 'ollama')
        began = time.monotonic()
        response = self.post()
        self.assertLess(time.monotonic() - began, .5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json), {'job_id', 'archived', 'promoted', 'latest', 'new', 'status'})
        self.assertTrue(response.json['promoted'])
        self.assertEqual(self.latest_file.read_bytes(), READY)
        self.assertEqual(Path(response.json['archived']).read_bytes(), OLD)
        self.assertFalse(self.new_file.exists())
        self.wait_state('wake-operation', 'waiting')
        self.assertEqual((self.chat.generations, self.audio.submissions), (0, 0))
        lease.release({'completion_verified': True, 'cleanup_verified': True})
        lease.close()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertEqual(self.latest_file.read_bytes(), READY)
        self.assertEqual(self.new_file.read_bytes(), SONG)
        self.assertEqual(stat.S_IMODE(self.new_file.stat().st_mode), 0o644)
    def test_duplicate_trigger_does_not_promote_archive_or_generate_twice(self):
        event = threading.Event()
        self.events.append(event)
        self.chat.event = event
        first = self.post()
        second = self.post('different-trigger')
        self.assertEqual(second.json['status'], 'busy')
        self.assertEqual(second.json['job_id'], first.json['job_id'])
        self.assertEqual(len(list(self.archive.glob('*.mp3'))), 1)
        event.set()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        again = self.post()
        self.assertEqual(again.json, first.json)
        self.assertEqual((self.chat.generations, self.audio.submissions), (1, 1))
        self.assertEqual(len(list(self.archive.glob('*.mp3'))), 1)
    def test_simultaneous_triggers_schedule_one_replacement(self):
        event = threading.Event()
        self.events.append(event)
        self.chat.event = event
        barrier = threading.Barrier(8)
        def trigger(number):
            barrier.wait(2)
            return self.post('parallel-' + str(number)).json
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(trigger, range(8)))
        winners = [value['job_id'] for value in responses if value['status'] == 'started']
        self.assertEqual(len(winners), 1)
        self.assertEqual(sum(value['status'] == 'busy' for value in responses), 7)
        event.set()
        self.assertEqual(self.wait_job(winners[0])['status'], 'done')
        self.assertEqual((self.chat.generations, self.audio.submissions), (1, 1))
    def test_music_and_wallpaper_serialize_through_same_gpu0_admission(self):
        event = threading.Event()
        self.events.append(event)
        self.audio.cleanup_event = event
        self.post()
        self.wait_state('wake-operation', 'cleanup_pending', 'music')
        self.runner.submit('wallpaper-waits', 'wallpaper', 'workflow', self.wallpaper_workflow)
        time.sleep(.03)
        self.assertEqual(self.wallpaper.submissions, 0)
        self.assertEqual(self.admission.snapshot()['owners']['gpu0']['target'], 'music')
        self.assertFalse(self.new_file.exists())
        event.set()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertEqual(self.runner.wait('wallpaper-waits', 2)['state'], 'success')
        self.assertEqual(self.wallpaper.submissions, 1)
    def test_gpu1_is_physically_released_before_music_submission(self):
        original = self.audio.json
        def inspected(method, route, payload=None, timeout=5):
            if route == '/prompt':
                self.assertNotIn('gpu1', self.admission.snapshot()['owners'])
                physical = NativeLock(self.paths['gpu1'])
                self.assertTrue(physical.acquire(0))
                physical.close()
            return original(method, route, payload, timeout)
        self.audio.json = inspected
        self.post()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertEqual(self.chat.unloads, 1)
    def test_cleanup_failure_preserves_last_song_and_reserved_owner(self):
        self.memory_high = True
        self.post()
        job = self.wait_job('wake-operation')
        self.assertEqual(job['status'], 'error')
        self.assertEqual(job['runner_state'], 'cleanup_pending')
        self.assertEqual(self.latest_file.read_bytes(), READY)
        self.assertFalse(self.new_file.exists())
        self.assertIn('gpu0', self.admission.snapshot()['owners'])
        self.assertIsNone(self.admission.try_acquire('meeting-after-pending', 'hash', ('gpu0',), 'host-whisperx'))
    def test_missing_and_incorrect_audio_never_publish(self):
        for variant in ('missing', 'wav', 'input', 'bytes', 'mime'):
            with self.subTest(variant=variant):
                transport = AudioTransport(variant=variant)
                self.audio = transport
                self.runner.backends['music'] = ComfyBackend(self.targets['music'], transport, probe=lambda: 0, poll_interval=.005)
                job_id = 'bad-audio-' + variant
                self.post(job_id)
                self.assertEqual(self.wait_job(job_id)['status'], 'error')
                self.assertEqual(self.latest_file.read_bytes(), READY)
                self.assertFalse(self.new_file.exists())
                self.assertEqual(transport.submissions, 1)
                self.assertEqual(self.admission.snapshot()['owners'], {})
    def test_partial_audio_waits_for_terminal_history(self):
        self.audio.variant = 'partial'
        self.post()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertEqual(self.audio.history_reads, 2)
        self.assertEqual(self.audio.views[0]['filename'], 'synthetic.mp3')
    def test_comfy_terminal_error_cleans_then_preserves_last_song(self):
        self.audio.variant = 'error'
        self.post()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'error')
        self.assertEqual(self.latest_file.read_bytes(), READY)
        self.assertFalse(self.new_file.exists())
        self.assertEqual(self.audio.views, [])
        self.assertEqual(self.audio.frees, 1)
        self.assertEqual(self.admission.snapshot()['owners'], {})
    def test_lost_chat_reply_does_not_start_music_or_duplicate(self):
        self.chat.fail = True
        self.post()
        job = self.wait_job('wake-operation')
        self.assertEqual(job['status'], 'error')
        self.assertEqual(self.audio.submissions, 0)
        self.assertIn('gpu1', self.admission.snapshot()['owners'])
        self.assertEqual(self.post().status_code, 503)
        self.assertEqual(self.chat.generations, 1)
        self.assertEqual(self.latest_file.read_bytes(), READY)
    def test_timeout_reconnects_existing_music_job_without_duplicate_generation(self):
        event = threading.Event()
        self.events.append(event)
        self.audio.cleanup_event = event
        self.client.wait_timeout = .08
        self.post()
        job = self.wait_job('wake-operation')
        self.assertEqual(job['status'], 'error')
        self.assertIn('gpu0', self.admission.snapshot()['owners'])
        event.set()
        self.client.wait_timeout = 2
        operation = self.app._output_slot().operation_status('wake-operation')
        key = hashlib.sha256(encode('music')).hexdigest()
        payload = self.app._output_slot().store.read('plans/wake-operation-' + key + '.json')
        request_id = JobClient.stage_id('wake-operation', 'music')
        data, media = self.client.run(request_id, 'music', 'workflow', payload, artifact=True)
        self.assertEqual((data, media), (SONG, 'audio/mpeg'))
        self.assertEqual(self.audio.submissions, 1)
        self.assertFalse(self.new_file.exists())  # Failed app operation is not automatically republished.
        self.assertEqual(self.latest_file.read_bytes(), READY)
    def test_restart_retains_completed_response_and_job_status(self):
        response = self.post()
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        restarted = load_app('music_restarted')
        result = restarted.app.test_client().post('/swap_and_generate', headers={'X-Request-ID': 'wake-operation'})
        self.assertEqual(result.json, response.json)
        self.assertEqual(restarted.app.test_client().get('/job/wake-operation').json['status'], 'done')
        self.assertEqual((self.chat.generations, self.audio.submissions), (1, 1))
    def test_interrupted_client_operation_reports_error_after_restart_without_replay(self):
        operation = self.app._output_slot().begin('swap-and-generate', 'interrupted')
        self.app._set_job(operation, status='running')
        operation.lock.close()
        operation.lock = None
        restarted = load_app('music_interrupted_restart')
        self.assertEqual(restarted._get_job('interrupted')['status'], 'error')
        result = restarted.app.test_client().post('/swap_and_generate', headers={'X-Request-ID': 'interrupted'})
        self.assertEqual(result.status_code, 503)
        self.assertEqual((self.chat.generations, self.audio.submissions), (0, 0))
    def test_publication_failure_preserves_existing_song_and_exposes_error(self):
        with patch.object(self.app, 'publish_bytes', side_effect=OSError('synthetic disk failure')):
            self.post()
            self.assertEqual(self.wait_job('wake-operation')['status'], 'error')
        self.assertEqual(self.latest_file.read_bytes(), READY)
        self.assertFalse(self.new_file.exists())
    def test_genre_vocal_duration_prompt_and_workflow_mutations_unchanged(self):
        self.post()
        job = self.wait_job('wake-operation')
        self.assertEqual(job['status'], 'done')
        payload = self.chat.payloads[0]
        self.assertEqual(payload['options'], {'temperature': .8, 'num_predict': 900})
        self.assertEqual((payload['format'], payload['think'], payload['stream']), ('json', False, False))
        user = json.loads(payload['messages'][1]['content'])
        self.assertEqual(user['duration_seconds'], 135)
        self.assertEqual(user['genre_pack'], self.app.pick_genre_pack()['name'])
        self.assertEqual(user['base_tags'], job['base_tags_preview'] if len(user['base_tags']) <= 300 else user['base_tags'])
        self.assertIn(job['vocal_gender'], ('male', 'female'))
        self.assertEqual(self.audio.payload['225']['inputs']['value'], 'synthetic production tags')
        self.assertEqual(self.audio.payload['226']['inputs']['value'], '[verse]\nGood morning synthetic lyrics')
        self.assertEqual(self.audio.payload['117']['inputs']['value'], 135)
        self.assertTrue(1 <= self.audio.payload['409']['inputs']['seed'] <= 999_999_999_999_999)
        expected = copy.deepcopy(self.workflow)
        for node, key in (('225', 'value'), ('226', 'value'), ('117', 'value'), ('409', 'seed')):
            expected[node]['inputs'][key] = self.audio.payload[node]['inputs'][key]
        self.assertEqual(self.audio.payload, expected)
    def test_malformed_lyrics_reply_retains_existing_fallback(self):
        self.chat.malformed = True
        self.post()
        job = self.wait_job('wake-operation')
        self.assertEqual(job['status'], 'done')
        self.assertTrue(self.audio.payload['225']['inputs']['value'])
        self.assertIn('[intro-short]', self.audio.payload['226']['inputs']['value'])
        self.assertIn('Good morning, the day is calling', self.audio.payload['226']['inputs']['value'])
    def test_original_health_unknown_job_and_debug_api_paths_preserved(self):
        client = self.app.app.test_client()
        self.assertEqual(client.get('/health').json, {'ok': True})
        self.assertEqual(client.get('/job/unknown').json, {'status': 'unknown'})
        self.assertEqual(client.get('/debug/comfy_history/unknown').status_code, 400)
        self.post()
        self.wait_job('wake-operation')
        with patch('requests.get') as http:
            http.return_value.json.return_value = {'owned-prompt': {'synthetic': True}}
            response = client.get('/debug/comfy_history/wake-operation')
            self.assertEqual(response.json, {'owned-prompt': {'synthetic': True}})
            self.assertTrue(http.call_args[0][0].endswith('/history/owned-prompt'))
    def test_no_ready_replacement_keeps_current_song_available(self):
        self.new_file.unlink()
        response = self.post()
        self.assertFalse(response.json['promoted'])
        self.assertEqual(self.latest_file.read_bytes(), OLD)
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertEqual(self.latest_file.read_bytes(), OLD)
        self.assertEqual(self.new_file.read_bytes(), SONG)
    def test_archive_collision_preserves_both_songs_and_naming_format(self):
        first = self.app.archive_file(str(self.latest_file))
        self.latest_file.write_bytes(READY)
        second = self.app.archive_file(str(self.latest_file))
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), OLD)
        self.assertEqual(Path(second).read_bytes(), READY)
        self.assertRegex(Path(second).name, r'^wakeup-\d{4}-\d{2}-\d{2}_\d{6}\.mp3$')
    def test_atomic_publisher_never_exposes_partial_replacement(self):
        old_new = self.new_file.read_bytes()
        def blocked_replace(source, destination):
            self.assertEqual(self.new_file.read_bytes(), old_new)
            self.assertEqual(Path(source).read_bytes(), SONG)
            self.assertEqual(stat.S_IMODE(Path(source).stat().st_mode), 0o644)
            raise OSError('synthetic replace failure')
        with patch('os.replace', side_effect=blocked_replace), self.assertRaises(OSError):
            publish_bytes(self.new_file, SONG)
        self.assertEqual(self.new_file.read_bytes(), old_new)
        self.assertEqual(list(self.latest.glob('.tmp-*')), [])
        self.assertEqual(self.latest_file.read_bytes(), OLD)
    def test_late_metadata_failure_never_writes_after_output_guard_release(self):
        store = self.app._output_slot().store
        original = store.write
        fired, later_writes = False, []
        def fail_after_effect(name, value):
            nonlocal fired
            if fired:
                later_writes.append(name)
            original(name, value)
            if name == 'active.json' and value.get('status') == 'ok' and not fired:
                fired = True
                raise OSError('synthetic late status synchronization failure')
        with patch.object(store, 'write', side_effect=fail_after_effect):
            self.post()
            self.assertEqual(self.wait_job('wake-operation')['status'], 'done')
        self.assertTrue(fired)
        self.assertEqual(later_writes, [])
        self.assertEqual(self.new_file.read_bytes(), SONG)
        self.assertEqual(self.latest_file.read_bytes(), READY)
    @unittest.skipUnless(sys.platform.startswith('linux'), 'Native Linux process death')
    def test_killed_music_output_process_is_not_replayed_after_restart(self):
        ready = self.root / 'killed-writer-ready'
        code = ('import importlib.util,pathlib,time; '
                f'spec=importlib.util.spec_from_file_location("synthetic_music_child",{str(ROOT / "app.py")!r}); '
                'app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app); '
                'op=app._output_slot().begin("swap-and-generate","killed-writer"); '
                'app._set_job(op,status="running"); '
                f'pathlib.Path({str(ready)!r}).touch();time.sleep(30)')
        child = subprocess.Popen([sys.executable, '-c', code], env=os.environ,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(child)
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                self.fail(str(child.communicate()))
            time.sleep(.005)
        self.assertTrue(ready.exists())
        child.kill()
        child.communicate(timeout=5)
        self.assertEqual(child.returncode, -9)
        restarted = load_app('killed_writer_restart')
        self.assertEqual(restarted._get_job('killed-writer')['status'], 'error')
        response = restarted.app.test_client().post('/swap_and_generate', headers={'X-Request-ID': 'killed-writer'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual((self.chat.generations, self.audio.submissions), (0, 0))
        self.assertEqual(self.latest_file.read_bytes(), OLD)
    @unittest.skipUnless(sys.platform.startswith('linux'), 'Native Linux meeting contention')
    def test_meeting_transcription_and_music_share_gpu0_without_nested_locks(self):
        self._native_meeting_contention('gpu0')
    @unittest.skipUnless(sys.platform.startswith('linux'), 'Native Linux meeting contention')
    def test_meeting_summary_and_music_share_gpu1_without_nested_locks(self):
        self._native_meeting_contention('gpu1')
    def _native_meeting_contention(self, resource):
        binary = self.root / 'probe'
        binary.mkdir()
        probe = binary / 'nvidia-smi'
        probe.write_text('#!' + sys.executable + '\nprint(0)\n')
        probe.chmod(0o755)
        ready, release = self.root / 'meeting-ready', self.root / 'meeting-release'
        code = f'import pathlib,time;pathlib.Path({str(ready)!r}).touch();\nwhile not pathlib.Path({str(release)!r}).exists():time.sleep(.005)'
        environment = dict(os.environ, AIHUB_GPU_RUNNER_CONFIG=str(self.config), AIHUB_GPU_RUNNER_PYTHON=sys.executable,
                           AIHUB_GPU_STAGE_ID='synthetic-meeting', PATH=str(binary) + ':' + os.environ['PATH'])
        helper = Path(os.environ['MEETING_PIPELINE_SOURCE']) / 'bin/with-gpu-lock.sh'
        process = subprocess.Popen(['bash', str(helper), resource, 'synthetic meeting', sys.executable, '-c', code],
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail(str(process.communicate()))
            time.sleep(.005)
        self.assertTrue(ready.exists())
        self.post()
        self.wait_state('wake-operation', 'waiting', 'music' if resource == 'gpu0' else 'lyrics-tags')
        self.assertEqual(self.audio.submissions, 0)
        if resource == 'gpu1':
            self.assertEqual(self.chat.generations, 0)
        else:
            self.assertEqual(self.chat.generations, 1)
        self.assertEqual(self.latest_file.read_bytes(), READY)
        release.touch()
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(self.wait_job('wake-operation')['status'], 'done')

if __name__ == '__main__':
    unittest.main()
