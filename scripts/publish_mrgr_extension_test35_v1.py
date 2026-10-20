"""Export accepted supplementary35-case results as portable repository evidence."""
from pathlib import Path
import csv
import hashlib
import gzip
import json
import math
import re
import shutil
import sys
from collections import defaultdict

PROJECT=Path(__file__).resolve().parents[1]
BASE=Path('/scratch/djc/GeoPhy-Set3D/mrgr_extension_test35_v1')
DEST=PROJECT/'docs/reports/artifacts/mrgr_extension_test35_v1'
REPORT=PROJECT/'docs/reports/mrgr_extension_test35_v1_zh.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    accepted=json.loads((BASE/'acceptance.json').read_text())
    assert accepted['completed'] and accepted['new_fields']==8330 and accepted['independently_verified_fields']==13090
    for p,h in accepted['evidence'].items():assert sha(p)==h,p
    DEST.mkdir(parents=True,exist_ok=True);manifest=dict(acceptance_sha256=sha(BASE/'acceptance.json'),files={})
    def export(origin,name=None,projection=False):
        origin=Path(origin);p=DEST/(name or origin.name)
        if projection:
            r=json.loads(origin.read_text());r.pop('audit',None);dump(p,r)
        else:shutil.copy2(origin,p)
        manifest['files'][str(p.relative_to(PROJECT))]=dict(sha256=sha(p),origin=str(origin),origin_sha256=sha(origin),transform='remove verbose local access audit paths' if projection else 'exact copy')
    for name in ('final_lock.json','acceptance.json','boundary_checks.json','training_audit.json','reuse_audit.json','reuse_methods.json','pre_GT_implementation_amendment.json','pre_GT_access_amendment.json','source_amendment_audit.json'):
        export(BASE/name)
    for name in ('summary.json','independent_verification.json','independent_fields.json','figure_audit.json'):
        export(BASE/'analysis'/name,projection=name!='summary.json')
    for name in ('all_predictions_sealed.json','gt_access.json'):
        export(BASE/'evaluation/test'/name,projection=True)
    export(REPORT,'source_report.md')
    origin=BASE/'analysis/all_metrics.jsonl';packed=DEST/'all_metrics.jsonl.gz'
    with origin.open('rb') as src, packed.open('wb') as dst:
        with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as zipped:shutil.copyfileobj(src,zipped)
    manifest['files'][str(packed.relative_to(PROJECT))]=dict(sha256=sha(packed),origin=str(origin),origin_sha256=sha(origin),transform='lossless gzip of every raw metric, including all shape/physical centroid/peak-location and runtime fields')
    export(BASE/'pre_GT_implementation_revision1/final_lock.json','initial_final_lock.json')
    export(BASE/'pre_GT_implementation_revision1/finish.py','initial_finish.py.txt')
    for source,target in [('final_lock.json','intermediate_final_lock.json'),('seal_and_score.json','pre_GT_seal_failure.json'),('seal_and_score.log','pre_GT_seal_failure.log'),('runtime.py','initial_runtime.py.txt'),('controller.py','initial_controller.py.txt')]:
        export(BASE/'pre_GT_implementation_revision2'/source,target)
    for p in (BASE/'analysis').glob('fixed_case_*'):
        if p.suffix in ('.png','.pdf'):export(p)
    # Preserve every value while pivoting long case/metric rows into a compact wideCSV.
    p=BASE/'analysis/per_case_metrics.csv';wide={};metrics=set()
    with p.open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('method','arm','seed','stage','condition','draws','case_id'))
            row=wide.setdefault(key,dict(zip(('method','arm','seed','stage','condition','draws','case_id'),key)))
            assert r['metric'] not in row;row[r['metric']]=r['value'];metrics.add(r['metric'])
    dest=DEST/'per_case_metrics.csv'
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','arm','seed','stage','condition','draws','case_id',*sorted(metrics)],lineterminator='\n')
        writer.writeheader();writer.writerows(wide.values())
    manifest['files'][str(dest.relative_to(PROJECT))]=dict(sha256=sha(dest),origin=str(p),origin_sha256=sha(p),transform='long-to-wideCSV pivot; identical case/metric values, LF newlines')
    summary=json.loads((DEST/'summary.json').read_text());values=defaultdict(list)
    for row in wide.values():
        for metric in metrics:
            if row[metric]:values[row['method'],row['stage'],row['condition'],row['draws'],metric].append(float(row[metric]))
    checked=0
    for panel in summary['panels']:
        for metric,s in panel['metrics'].items():
            arr=values.get((panel['method'],panel['stage'],panel['condition'],','.join(map(str,panel['draws'])),metric))
            if arr:
                assert math.isclose(sum(arr)/len(arr),s['mean'],rel_tol=1e-12,abs_tol=1e-12);checked+=1
    manifest['panel_means_recomputed_from_exported_csv']=checked
    for p in (PROJECT/'scripts/mrgr_extension_test35_v1').glob('*.py'):
        lock=json.loads((BASE/'final_lock.json').read_text());assert sha(p)==lock['sources'][str(p)]
        manifest['files'][str(p.relative_to(PROJECT))]=dict(sha256=sha(p),transform='exact frozen evaluation source')
    manifest['files'][str(REPORT.relative_to(PROJECT))]=dict(sha256=sha(REPORT),transform='exact accepted report; already uses portable evidence links')
    dump(DEST/'export_manifest.json',manifest)
    missing=[]
    for link in re.findall(r'\]\(([^)]+)\)',REPORT.read_text()):
        if not (REPORT.parent/link).exists():missing.append(link)
    assert not missing,missing
    print('Export accepted; case table rows',len(wide),'verified panel means',checked,'files',len(manifest['files']))

if __name__=='__main__':main()
