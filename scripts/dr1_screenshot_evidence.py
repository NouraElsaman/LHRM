"""Read-only DR1 evidence summary to photograph in a real terminal.

Run from the LHRM project root:
    python scripts/dr1_screenshot_evidence.py
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

ROOT = Path.cwd()
EVIDENCE = ROOT / 'experiments' / 'RUN_0_P1F_EXT1' / 'RUN_0_P1F_EXT1_EVIDENCE_20261009_222831.zip'
EXPECTED_SHA256 = '806fce2e7f001bcabcd4d23f451b7a80f44dd5b5530f38a53d239a66d6544329'
assert EVIDENCE.is_file(), f'Evidence ZIP missing: {EVIDENCE}'
actual_sha = sha256(EVIDENCE.read_bytes()).hexdigest()
assert actual_sha == EXPECTED_SHA256, f'Wrong evidence archive: {actual_sha}'

with ZipFile(EVIDENCE) as z:
    assert z.testzip() is None, 'Corrupt archive'
    prefix = 'RUN_0_P1F_EXT1/'
    pred = json.loads(z.read(prefix + 'variant_L2/val_sample_predictions.json'))
    result = json.loads(z.read(prefix + 'run0_decision.json'))['candidate_results']['L2']
    summary = json.loads(z.read('DR1_STEP1000_SUMMARY.json'))

assert len(pred) == 200, f'Expected 200 dev rows, got {len(pred)}'
ref_labels = pred[0]['labels']
assert all(row['labels'] == ref_labels and len(row['scores']) == len(ref_labels) for row in pred)
ranked = [tuple(sorted(range(len(row['scores'])), key=lambda i: -row['scores'][i])) for row in pred]
top1 = Counter(ref_labels[r[0]] for r in ranked)
top3 = Counter(tuple(ref_labels[i] for i in r[:3]) for r in ranked)

print('=' * 64)
print('DR1 EVIDENCE AUDIT — ACTUAL RUN_0_P1F_EXT1 ARTIFACTS')
print('=' * 64)
print('SHA-256 VERIFIED:', actual_sha)
print('Experiment: LHRM / CSAM variant L2, 1000 optimizer steps')
print('Dev diagnostic documents:', len(pred), '| Labels:', len(ref_labels))
print('V1/V2/V3/V4:', [result[k] for k in ('V1', 'V2', 'V3', 'V4')])
print('V5 (model beats prior):', result['V5'], '— FAIL')
print('Model P@1:', summary['v5_model_hits'], '/ 200 =', f"{summary['v5_model_hits']/200:.1%}")
print('Prior P@1:', summary['v5_prior_hits'], '/ 200 =', f"{summary['v5_prior_hits']/200:.1%}")
print('Distinct top-1 predicted labels:', len(top1))
print('Top-1 label counts:', top1.most_common())
print('Distinct ordered top-3 patterns:', len(top3))
print('Most common top-3:', top3.most_common(1)[0])
print('CONCLUSION: 1000-step L2 did not beat frequency prior on this 200-doc sample.')
print('CAUTION: This does not prove all CSAM implementations cannot learn.')
print('=' * 64)
