#!/usr/bin/env python3
"""Append-only erratum of replay metadata; never alter registered evidence/code."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from geophy_set3d.data.build_utils import digest
from geophy_set3d.training.utils import write_json
from run_geometry_augmentation_formal import configuration

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');a=ap.parse_args();c,out=configuration(a.config);reg=out/'registration';source=reg/'control_fresh_numeric_replay.json';data=json.loads(source.read_text());rows=data['first100_steps'];assert len(rows)==100
    assert [r['step'] for r in rows[:83]]==list(range(1,84)) and [r['step'] for r in rows[83:]]==list(range(1,18))
    corrected=[dict(r,registered_global_step_label=r['global_step'],global_step=i+1) for i,r in enumerate(rows)]
    write_json(reg/'control_fresh_replay_index_erratum.json',dict(source_sha256=digest(source),registered_evidence_unchanged=True,actual_optimizer_steps=100,
        reason='Generator extend evaluated len(trace) while extending: original global_step label increments twice; local step/epoch/sequence/hash/loss unchanged.',
        effect='Metadata labels only. Control reuse remains REJECTED; formal training uses independent actual optimizer step counter, no intervention change.',
        corrected_trace=corrected,step1_10_100=[corrected[i-1] for i in (1,10,100)]))
    print('Replay index erratum saved; original registered evidence unchanged')
if __name__=='__main__':main()
