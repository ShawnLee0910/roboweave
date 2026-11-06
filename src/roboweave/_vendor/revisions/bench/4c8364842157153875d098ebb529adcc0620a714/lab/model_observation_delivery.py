"""Prove a current composite was in the successful provider request before V ack."""
import base64
import hashlib
import json
from pathlib import Path


def acknowledge_model_observation(*, scene, observation, episode_id, sequence, call_dir):
    composite = observation.read_model_composite_png()
    composite_sha = hashlib.sha256(composite).hexdigest()
    encoded_sha = hashlib.sha256(
        b'data:image/png;base64,' + base64.b64encode(composite)
    ).hexdigest()
    manifest = Path(call_dir) / 'request_manifest.json'
    if manifest.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('oversized provider request manifest')
    raw = manifest.read_bytes()

    def contains_image(value):
        if isinstance(value, dict):
            image = value.get('image_url')
            return (value.get('type') in {'image_url', 'input_image'}
                    and isinstance(image, dict) and image.get('encoded_sha256') == encoded_sha
                    or any(contains_image(item) for item in value.values()))
        return isinstance(value, list) and any(contains_image(item) for item in value)

    if not contains_image(json.loads(raw)):
        raise ValueError('provider request does not prove delivery of the current V composite')
    fields = dict(episode_id=episode_id, consumer='MODEL_REQUEST',
                  request_id=Path(call_dir).name, observation_id=observation.observation_id,
                  sequence=sequence, composite_sha256=composite_sha,
                  request_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  image_url_sha256=encoded_sha)
    result = scene.request('observation_delivered', **fields)
    if (result.get('status') != 'PASS' or result.get('accepted') is not True
            or any(result.get(key) != fields[key] for key in
                   ('episode_id', 'consumer', 'observation_id', 'sequence'))
            or type(result.get('episode_terminal')) is not bool):
        raise ValueError('resident rejected current model observation delivery')
    from scene_protocol import _write_json
    _write_json(Path(call_dir) / 'private/model_observation_delivery.json',
                {**fields, 'request_manifest': str(manifest), 'result': dict(result)})
    return result
