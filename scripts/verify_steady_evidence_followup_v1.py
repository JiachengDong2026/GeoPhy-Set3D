"""Refresh local evidence inventory and register a temporal verification handoff.

The generator's reported supporting records are not treated as inspected evidence.
Only metadata is read, including across splits; no test arrays or CFD rerun.
"""
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import audit_dataset_steadiness_v1 as inventory


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/steady_evidence_followup_v1'


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    inventory.OUT = OUT / 'metadata_inventory'
    with (OUT / 'inventory_stdout.json').open('w') as f, redirect_stdout(f):
        inventory.main()
    audit = json.loads((inventory.OUT / 'audit_summary.json').read_text())
    index = [json.loads(s) for s in (inventory.POOL / 'source_index.jsonl').read_text().splitlines() if s.strip()]
    train = [r for r in index if r['split'] == 'train']
    # Deterministic metadata-only coverage, no use of concentration or model errors.
    fields = ('wind_speed_m_s', 'pressure_pa', 'active_hole_count', 'geometry_id')
    tags = lambda r: {(k, str(r['condition_meta'][k])) for k in fields}
    remaining = set().union(*(tags(r) for r in train))
    selected = []
    candidates = sorted(train, key=lambda r: hashlib.sha256(r['case_id'].encode()).hexdigest())
    while remaining:
        best = max(candidates, key=lambda r: len(tags(r) & remaining))
        assert tags(best) & remaining
        selected.append(dict(case_id=best['case_id'], **best['condition_meta']))
        remaining -= tags(best)
        candidates.remove(best)
    status = dict(
        checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        generator_statement_via_user='t=2 s 应已达到稳态；有后段多帧或监测曲线支持',
        statement_class='Reported evidence exists; records and acceptance criteria not yet inspected',
        supporting_record_location='requested_from_user_pending',
        case_count=audit['manifest_records'],
        local_case_frame_counts=audit['frames_per_case_counts'],
        local_CFD_monitor_candidates=audit['cfd_candidate_filenames'],
        temporal_steadiness_status='not_verifiable_from_current_local_delivery',
        transport_balance_status='not_computable_without_u_rho_diffusion_and_boundary_fluxes',
        interpretation='Neither a steady-state pass nor evidence of non-steadiness',
        single_frame_cannot_identify_time_derivative=True,
        final_test_arrays_opened=False, CFD_restarted=False,
        selection_basis='Train metadata greedy coverage of wind, pressure, active-hole count and geometry; ties SHA256(case_id)',
        representative_train_cases=selected,
        requested_records=[
            'Original case IDs, timestamps, export coordinates and transient/steady solver configuration',
            'Aligned concentration snapshots through the late interval; initially 1.5-2.0 s if available, retaining earlier history',
            'Methane inventory, signed input/output species flow histories and probe concentration histories',
            'Per-step equation residuals, time step and inner-iteration convergence criteria',
            'Velocity, density, effective diffusion/turbulence closure, source and boundary definitions for transport verification',
        ],
        temporal_diagnostics=[
            'Full-field window first/last drift and maximum pairwise relative L1 at explicit time separations',
            'Last-frame deviation from time-weighted window mean; accumulated trends vs fluctuations',
            'Density-weighted methane inventory when rho exists; otherwise label integral(C) as concentration proxy',
            'Peak, centroid, height profiles and fixed-ray projections where the necessary fields are supplied',
            'Stationary monitoring values alone support only those monitors, not every spatial degree of freedom',
            'Repeat at longer windows if drift persists; distinguish instantaneous and statistical stationarity',
        ],
        acceptance_thresholds='Not calibrated; report continuous diagnostics first; do not invent a universal pass criterion',
        dataset_wide_claim='Representative training checks cannot certify all 237 cases; complete per-case status before a dataset-wide pass',
        inventory_script_sha256=hashlib.sha256(Path(inventory.__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write(OUT / 'verification_status.json', status)
    (OUT / 'per_case_status.jsonl').write_text(''.join(json.dumps(dict(
        case_id=r['case_id'], split=r['split'], available_times_s=[2.0],
        temporal_status='unverified_missing_local_time_history',
        generator_claim='reported_supporting_records_not_inspected'), ensure_ascii=False) + '\n' for r in index))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
