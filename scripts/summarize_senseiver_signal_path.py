#!/usr/bin/env python3
"""Posthoc audit/plots of saved signal diagnosis. No model construction or forward."""
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from diagnose_senseiver_signal_path import config, verify, audited_guard, sha256, ART
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        writer = csv.DictWriter(f, keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def span(values):
    return dict(min=float(min(values)), max=float(max(values)), mean=float(np.mean(values)))


def main():
    c, source = config(REPO / 'configs/ray_senseiver_signal_path_v1.yaml')
    out = Path(c['output'])
    full, local = {}, {}
    stage_rows, measurement_rows, gradient_rows, fd_rows, contribution_rows = [], [], [], [], []
    array_manifest, spatial_rows = {}, []
    complete = json.loads((out / 'complete.json').read_text())
    with audited_guard(c, source) as guard:
        registration = verify(c, source)
        events = [json.loads(x) for x in (out / 'forward_ledger.jsonl').read_text().splitlines()]
        plans = registration['full_forward_plan'] + registration['local_graph_plan']
        assert set(x['id'] for x in events) == set(x['id'] for x in plans)
        for entry in plans:
            identity = entry['id']
            ledger = [x for x in events if x['id'] == identity]
            expected = ['started', 'returned', 'saved']
            if identity == 'local_epoch0_case0':
                expected = ['started', 'aborted'] + expected
            assert [x['event'] for x in ledger] == expected
            path = out / 'records' / (identity + '.json')
            assert sha256(path) == ledger[-1]['record_sha256']
            row = json.loads(path.read_text())
            key, i = row['checkpoint'], row['index']
            assert row['model_identity'] == complete[key]['model_identity_before']
            assert row['parameter_buffers_unchanged'] and row['optimizer_steps'] == 0
            if identity.startswith('local_'):
                local[key, i] = row
                assert row['grad_buffers_none'] and not row['optimizer_created']
                assert row['sampled_queries'] == len(row['query_indices']) == 259
                import hashlib
                assert hashlib.sha256(np.array(row['query_indices'], dtype=np.int64).tobytes()).hexdigest() == registration['fixed_259_indices_sha256']
                for stage, probes in row['stages'].items():
                    assert len(probes) == 4
                    for probe in probes:
                        assert probe['gradient_u']['nonfinite_count'] == probe['invalid_gradient_nonzero_count'] == 0
                        gradient_rows.append(dict(checkpoint=key, index=i, stage=stage, probe=probe['probe'],
                            phi=probe['phi'], gradient_l2=probe['gradient_l2'], gradient_dot_u=probe['gradient_dot_u'], **probe['gradient_u']))
                    for probe in row['finite_difference'][stage]:
                        assert probe['passed'] == (probe['absolute_error'] <= probe['tolerance'])
                        fd_rows.append(dict(checkpoint=key, index=i, stage=stage, **probe))
                continue
            full[key, i, row['mode']] = row
            assert row['original_geometry_path_masks_and_other_fields_same_objects'] and row['H_excluded_from_network']
            assert row['encoder_call_order'] == ['after_cross_1', 'block_1', 'after_cross_2', 'block_2', 'after_cross_3', 'block_3']
            for name, hkey in [('arrays', 'arrays_sha256'), ('baseline_arrays', 'baseline_sha256')]:
                if name in row:
                    path = out / row[name]
                    assert sha256(path) == row[hkey]
                    array_manifest[row[name]] = dict(sha256=row[hkey], bytes=path.stat().st_size)
            for stage, stats in row['stages'].items():
                assert stats['nonfinite_count'] == 0
                assert stats['effective_elements'] == sum(math.prod(x) for x in stats['call_shapes'])
                if stage.startswith('decoder.'):
                    assert sum(x[1] for x in stats['call_shapes']) == 262144
                    assert stats['calls'] == 16
                if stage == 'concentration':
                    assert stats['effective_elements'] == 262144
                if stage in ['measurement.raw', 'measurement.normalized_field']:
                    assert stats['effective_elements'] == row['valid_rays']
                stage_rows.append(dict(checkpoint=key, index=i, mode=row['mode'], stage=stage,
                    shape=json.dumps(stats['shape']), **{k: v for k, v in stats.items() if not isinstance(v, (dict, list))}, **stats['paired']))
            for subset, values in row['measurement'].items():
                normal_path = out / 'records' / f'{key}_case{i}_normal.json'
                normal = json.loads(normal_path.read_text())['measurement'][subset]
                measurement_rows.append(dict(checkpoint=key, index=i, mode=row['mode'], subset=subset, **values,
                    delta_l1=values['relative_l1']-normal['relative_l1'], delta_l2=values['relative_l2']-normal['relative_l2']))
            if row['mode'] == 'normal':
                assert row['decomposition']['reconstruction_passed']
                for group, values in row['decomposition']['groups'].items():
                    contribution_rows.append(dict(checkpoint=key, index=i, group=group,
                        input_rms=values['input']['rms'], mapped_rms=values['mapped_contribution']['rms'], weight_frobenius=values['weight_frobenius']))
                contribution_rows.append(dict(checkpoint=key, index=i, group='bias', input_rms=None,
                    mapped_rms=row['decomposition']['bias_contribution']['rms'], weight_frobenius=None))
        assert len(full) == 30 and len(local) == 6 and len(fd_rows) == 96 and len(array_manifest) == 36
        for key in ['epoch0', 'last']:
            assert complete[key]['model_identity_before'] == complete[key]['model_identity_after']
            assert complete[key]['hooks_removed'] and complete[key]['grad_buffers_none'] and complete[key]['checkpoint_unchanged']
            assert sha256(c['checkpoints'][key]['path']) == c['checkpoints'][key]['sha256']
            for i in range(3):
                normal = full[key, i, 'normal']
                for mode in c['modes']:
                    r = full[key, i, mode]
                    assert r['valid_rays'] == normal['valid_rays']
                    for stage in ['decoder.query_coordinates_fixed', 'decoder.learned_query_fixed', 'decoder.query_preproc_fixed']:
                        assert r['stages'][stage]['tensor_sha256'] == normal['stages'][stage]['tensor_sha256']
                    with np.load(out / r['arrays'], allow_pickle=False) as a:
                        x = a['concentration'].astype(np.float64).reshape(64, 64, 64)
                        assert x.size == 262144
                    with np.load(out / normal['arrays'], allow_pickle=False) as a:
                        base = a['concentration'].astype(np.float64).reshape(64, 64, 64)
                    diff = x-base
                    spatial_rows.append(dict(checkpoint=key, index=i, mode=mode,
                        concentration_mean=float(x.mean()), concentration_std=float(x.std()),
                        difference_mean=float(diff.mean()), difference_rms=float(np.sqrt(np.mean(diff**2))),
                        centered_difference_rms=float(diff.std()),
                        centered_over_total_difference_rms=float(diff.std()/np.sqrt(np.mean(diff**2))) if np.any(diff) else None))
        assert not complete['optimizer_created'] and complete['optimizer_steps'] == 0
        audit = dict(registration_and_amendment_chain_verified=True, all_registered_source_history_data_hashes_unchanged=True,
            all_36_saved_record_hashes_verified=True, all_36_external_array_hashes_verified=True,
            all_30_full_query_counts_262144=True, all_decoder_chunks_covered=True, element_weighted_counts_verified=True,
            all_shared_encoder_call_orders_verified=True, all_fixed_query_tensor_hashes_equal=True,
            all_parameters_and_buffers_unchanged=True, all_grad_buffers_none=True, all_hooks_removed=True,
            only_registered_checkpoint_hashes_verified=True, successful_full_forwards_repeated=0,
            full_counts=dict(Counter(x['event'] for x in events if x['kind']=='full')),
            local_counts=dict(Counter(x['event'] for x in events if x['kind']=='local_sampled_graph')),
            optimizer_created=False, optimizer_steps=0, posthoc_model_constructions=0, posthoc_model_forwards=0,
            partial_attempt_access_counter_limitation=complete['prior_partial_attempt'], posthoc_access=guard.summary())
    # Read scalar records already protected and verified; no subsequent data/checkpoint reads.
    ART.mkdir(parents=True, exist_ok=True)
    for name in ['complete.json', 'forward_ledger.jsonl', 'run.log']:
        shutil.copy2(out/name, ART/name)
    shutil.copytree(out/'records', ART/'records', dirs_exist_ok=True)
    shutil.copytree(out/'access_sessions', ART/'access_sessions', dirs_exist_ok=True)
    for name, rows in [('stages', stage_rows), ('measurement', measurement_rows), ('local_gradients', gradient_rows),
                       ('finite_difference', fd_rows), ('contributions', contribution_rows), ('spatial_response', spatial_rows)]:
        table(ART/(name+'.csv'), rows)
    write(ART/'external_arrays_manifest.json', dict(root=str(out), files=array_manifest))
    write(ART/'final_integrity_audit.json', audit)
    shutil.copy2('/tmp/signal_path_tests_v1_2.log', ART/'tests_v1_2.txt')
    shutil.copy2('/tmp/signal_path_regression.log', ART/'model_regression.txt')
    summary = {}
    for key in ['epoch0', 'last']:
        summary[key] = dict(responses={}, local_gradients={}, measurement_changes={}, output={})
        for mode in c['modes'][1:]:
            summary[key]['responses'][mode] = {stage: span([full[key,i,mode]['stages'][stage]['paired']['relative_RMS'] for i in range(3)]) for stage in full[key,0,mode]['stages']}
            summary[key]['responses'][mode]['concentration_relative_MA'] = span([full[key,i,mode]['stages']['concentration']['paired']['relative_MA_previous_definition'] for i in range(3)])
            summary[key]['measurement_changes'][mode] = {subset: {metric:span([full[key,i,mode]['measurement'][subset][metric]-full[key,i,'normal']['measurement'][subset][metric] for i in range(3)]) for metric in ['relative_l1','relative_l2']} for subset in ['input','H']}
        for stage in local[key,0]['stages']:
            rows = [r for i in range(3) for r in local[key,i]['finite_difference'][stage]]
            summary[key]['local_gradients'][stage] = dict(l2=span([r['gradient_l2'] for i in range(3) for r in local[key,i]['stages'][stage]]),
                fd_pass=sum(r['passed'] for r in rows), fd_total=len(rows), fd_max_absolute_error=max(r['absolute_error'] for r in rows),
                max_local_full_phi_difference=max(abs(r['local_vs_full_normal_phi']) for r in rows))
        summary[key]['output'] = dict(mean=span([full[key,i,'normal']['stages']['concentration']['mean'] for i in range(3)]),
            sigmoid_derivative_mean=span([full[key,i,'normal']['sigmoid_derivative']['mean'] for i in range(3)]),
            dlogC_dlogit_mean=span([local[key,i]['dlogC_dlogit']['mean'] for i in range(3)]),
            same_probe_gradient_C_over_logit=span([r for i in range(3) for r in local[key,i]['same_probe_concentration_over_logit_gradient_l2']]))
    w0=np.array(full['epoch0',0,'normal']['decomposition']['measurement_weight_column'])
    w1=np.array(full['last',0,'normal']['decomposition']['measurement_weight_column'])
    summary['measurement_weight_change']=dict(l2=float(np.linalg.norm(w1-w0)), relative_l2=float(np.linalg.norm(w1-w0)/np.linalg.norm(w0)), cosine=float(w0@w1/np.linalg.norm(w0)/np.linalg.norm(w1)))
    summary['timing']=dict(full_record_seconds=sum(r['seconds'] for r in full.values()), resumed_final_process_seconds=complete['seconds'],
        ledger_start=events[0]['at'], ledger_last_saved=events[-1]['at'], max_full_forward_peak_gpu_bytes=max(r['peak_gpu_allocated_bytes'] for r in full.values()),
        note='Full record times include hook statistics/projector/save; final process seconds is only v1.2, not cumulative. Local graph timing not independently instrumented. Ledger span includes repairs/tests/precommits.')
    summary['finite_difference']=dict(passed=sum(r['passed'] for r in fd_rows), total=len(fd_rows), failed=sum(not r['passed'] for r in fd_rows))
    write(ART/'summary.json', summary)
    # Figures use scalar records and existing outputs only; English labels, Chinese report captions.
    figdir=ART/'figures';figdir.mkdir(exist_ok=True)
    stages=['ray_token','encoder.preproc','encoder.kv_norm_1','encoder.after_cross_1','encoder.block_1',
            'encoder.after_cross_2','encoder.block_2','encoder.after_cross_3','encoder.final',
            'decoder.cross_attention_raw','decoder.cross_attention','decoder.logits','concentration']
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for ax,key in zip(axes,['epoch0','last']):
        for mode in c['modes'][1:]:
            v=np.array([[full[key,i,mode]['stages'][s]['paired']['relative_RMS'] for s in stages] for i in range(3)])
            ax.plot(np.arange(len(stages)),v.mean(0),'.-',label=mode)
            ax.fill_between(np.arange(len(stages)),v.min(0),v.max(0),alpha=.12)
        ax.set_yscale('log');ax.set_ylabel('Paired relative RMS');ax.set_title(key+' (line: mean of 3 cases; band: range)');ax.legend(ncol=4);ax.grid(alpha=.2)
    axes[-1].set_xticks(range(len(stages)),stages,rotation=55,ha='right')
    fig.tight_layout();fig.savefig(figdir/'stage_responses.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    groups=['measurement','raw_geometry','fourier_geometry','bias']
    for j,key in enumerate(['epoch0','last']):
        for i in range(3):
            vals=[next(r['mapped_rms'] for r in contribution_rows if r['checkpoint']==key and r['index']==i and r['group']==g) for g in groups]
            axes[0].plot(np.arange(4)+(j-.5)*.08,vals,'o--' if j else 'o-',label=f'{key} case{i}')
    axes[0].set_yscale('log');axes[0].set_xticks(range(4),groups,rotation=20);axes[0].set_ylabel('Mapped contribution RMS');axes[0].legend(fontsize=7)
    axes[1].plot(w0,label='epoch0');axes[1].plot(w1,label='last');axes[1].set_xlabel('Measurement column coefficient (64)');axes[1].set_ylabel('Weight');axes[1].legend()
    fig.tight_layout();fig.savefig(figdir/'preprocessing_contributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    names=['encoder.preproc','encoder.final','decoder.logits','concentration']
    for j,key in enumerate(['epoch0','last']):
        for i in range(3):
            for probe in range(4):
                ax.plot(np.arange(4)+.03*(i-1),[local[key,i]['stages'][s][probe]['gradient_l2'] for s in names],
                    'o--' if j else 'o-',color=['tab:blue','tab:orange'][j],alpha=.35,label=key if i==probe==0 else None)
    ax.set_yscale('log');ax.set_xticks(range(4),names);ax.set_ylabel('L2 of d phi / d u (4 probes x 3 cases)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'local_sensitivity.png',dpi=160);plt.close(fig)
    # Arrays already verified: allowed own-output reads remain guarded, no new inference.
    with audited_guard(c,source):
        for key in ['epoch0','last']:
            fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained')
            for i in range(3):
                fields=[]
                for mode in ['normal','zero','wrong_case']:
                    with np.load(out/full[key,i,mode]['arrays'],allow_pickle=False) as data:
                        fields.append(data['concentration'].reshape(64,64,64)[:,:,32].T.astype(float))
                lo=min(x.min() for x in fields);hi=max(x.max() for x in fields)
                for j,x in enumerate(fields):
                    im=axes[i,j].imshow(x,origin='lower',vmin=lo,vmax=hi,cmap='viridis');fig.colorbar(im,ax=axes[i,j],shrink=.65,format='%.2e')
                diffs=[fields[j]-fields[0] for j in [1,2]];lim=max(np.max(np.abs(x)) for x in diffs)
                for j,x in enumerate(diffs,3):
                    im=axes[i,j].imshow(x,origin='lower',vmin=-lim,vmax=lim,cmap='coolwarm');fig.colorbar(im,ax=axes[i,j],shrink=.65,format='%.1e')
                for j,title in enumerate(['normal','zero','wrong case','zero - normal','wrong - normal']):
                    axes[i,j].set_title(f'case{i} {title}');axes[i,j].set_xlabel('x index');axes[i,j].set_ylabel('y height index')
            fig.suptitle(f'{key}: z-index=32, XYZ field, y vertical; per-case shared absolute scale; differences separately scaled. No GT.')
            fig.savefig(figdir/f'fields_{key}.png',dpi=160);plt.close(fig)
    shutil.copytree(out/'access_sessions', ART/'access_sessions', dirs_exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['relative_l1','relative_l2']):
        for key in ['epoch0','last']:
            for subset in ['input','H']:
                v=[np.mean([full[key,i,m]['measurement'][subset][metric] for i in range(3)]) for m in c['modes']]
                ax.plot(range(5),v,'o-',label=f'{key} {subset}')
        ax.set_xticks(range(5),c['modes'],rotation=25);ax.set_ylabel(metric+' (3-case arithmetic mean)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/'measurement_metrics.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(audit=audit, summary=summary),indent=2))


if __name__ == '__main__':
    main()
