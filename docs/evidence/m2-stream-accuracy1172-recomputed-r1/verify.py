"""One finite read-only comparison of the pressure-stop1172 trajectory to s4."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from probe_model import require_bounded_job
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError('metadata exceeds finite byte budget')
    return json.loads(path.read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def prefix(path, count):
    with path.open('rb') as stream:
        lines = [stream.readline(65536) for _ in range(count)]
    require(all(line.endswith(b'\n') for line in lines), 'complete finite prefix unavailable')
    rows = [json.loads(line) for line in lines]
    require([row['step'] for row in rows] == list(range(1, count+1)), 'noncontiguous steps')
    return b''.join(lines), rows


def main():
    require_bounded_job()
    old, new = (ROOT/'runs'/('m2-stream-accuracy4096-'+s) for s in ('s3', 's4'))
    catalogue = '121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b'
    for run in (old, new):
        verify_source(run/'code', catalogue)
    receipt = read(ROOT/'runs/m2-stream-accuracy4096-s3-job.json')
    require(receipt['status'] == 'memory_pressure' and receipt['exit_code'] == 125, 'original stop identity differs')
    pointer = read(old/'latest-training.json')
    manifest_path = old/'training-snapshots'/pointer['generation']/'manifest.json'
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == pointer['manifest_sha256'], 'parent pointer differs')
    snapshot = read(manifest_path)
    require(snapshot['step'] == 1152 and snapshot['contract']['steps'] == 4096, 'parent saved target/cursor differs')
    require(sum(1 for _ in (old/'curve.jsonl').open('rb')) == 1172, 'original final logged cursor differs')
    selected = {}
    for filename in ('training.jsonl', 'curve.jsonl'):
        old_bytes, prior = prefix(old/filename, 1172)
        new_bytes, current = prefix(new/filename, 1172)
        if filename == 'training.jsonl':
            require(old_bytes == new_bytes, 'data recipe bytes differ')
        else:
            strip = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
            require(strip(prior) == strip(current), 'sample/loss/terms/gradients differ')
        selected['old-'+filename] = old_bytes
        selected['new-'+filename] = new_bytes
    evidence = ROOT/'docs/evidence/m2-stream-accuracy1172-recomputed-r1'
    evidence.mkdir(exist_ok=False)
    hashes = {}
    for filename, data in selected.items():
        (evidence/filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
    shutil.copyfile(Path(__file__), evidence/'verify.py')
    proof = {'status': 'passed', 'parent_banked_step': 1152, 'old_logged_steps': 1172,
             'recomputed_updates_compared': 20, 'all_prefix_rows_compared': 1172,
             'source_catalogue_sha256': catalogue, 'parent_pointer': pointer,
             'exact_training_recipe_bytes': True, 'exact_loss_terms_gradients_samples_excluding_elapsed': True,
             'artifact_sha256': hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'scope': 'Read-only fixed prefix of live s4 versus terminal pressure-stopped s3. No old1172 tensor artifact exists for equality. Not a saved1172 checkpoint or final accuracy/completion claim.'}
    (evidence/'scope.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
