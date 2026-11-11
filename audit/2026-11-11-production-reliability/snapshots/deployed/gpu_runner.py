"""Thin music adapter; GPU admission, execution and cleanup stay in the runner."""
import os
from aihub_gpu_runner.client import CURRENT_OPERATION, JobClient, CoordinationError


def _run(step, target, operation, payload, artifact=False, on_status=None):
    context = CURRENT_OPERATION.get()
    if context is None:
        raise CoordinationError('owned_music_output_operation_required')
    payload = context.snapshot(step, payload)
    request_id = JobClient.stage_id(context.request_id, step)
    client = JobClient.from_environment()
    result = client.run(request_id, target, operation, payload, artifact=artifact, on_status=on_status)
    return result, client, request_id


def chat(payload, on_status=None):
    result, _, _ = _run('lyrics-tags', os.environ.get('GPU_RUNNER_OLLAMA_TARGET', 'ollama'),
                        'chat', payload, on_status=on_status)
    return result


def music(workflow, on_status=None):
    (data, media), client, request_id = _run('music', os.environ.get('GPU_RUNNER_MUSIC_TARGET', 'music'),
                                            'workflow', workflow, artifact=True, on_status=on_status)
    if media != 'audio/mpeg':
        raise RuntimeError('Expected runner MP3 audio artifact')
    return data, client.result(request_id)['value']
