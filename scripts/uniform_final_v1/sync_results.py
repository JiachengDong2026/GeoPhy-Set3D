"""Publish only independently accepted artifacts; preserve unrelated repository work."""
from runtime import *
from scope import evaluation_stage
import re
import gzip

REPORTS=('mrgr_uniform_final_v1_zh.md','mrgr_uniform_final_model_v1_zh.md',
    'mrgr_uniform_final_ablations_v1_zh.md','mrgr_uniform_oari_v1_zh.md','mrgr_uniform_final_resources_v1_zh.md')

def git(*values):return subprocess.check_output(['git',*values],cwd=PROJECT,text=True).strip()

def main():
    accepted=verified(BASE/'acceptance.json');assert accepted['completed'] and not accepted['GT128_read']
    verify_hashes(accepted['evidence'])
    verify_hashes(verified(BASE/'final_lock.json')['sources'])
    artifacts=PROJECT/'docs/reports/artifacts/mrgr_uniform_final_v1'
    manifest=read(artifacts/'export_manifest.json')
    for name,item in manifest['files'].items():assert sha(PROJECT/name)==item['sha256'],name
    for name in ('all_metrics.jsonl','paired_statistics.jsonl'):
        p=artifacts/(name+'.gz');item=manifest['files'][str(p.relative_to(PROJECT))]
        assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==item['origin_sha256']
    for name in REPORTS:
        report=PROJECT/'docs/reports'/name
        for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
            if target.startswith(('http://','https://','#')):continue
            assert (report.parent/target).exists(),(name,target)
    assert git('branch','--show-current')=='main','Preserve current branch; do not switch someone else’s work'
    summary=read(artifacts/'summary.json')
    control=[];treatment=[]
    for condition in ('Val-A-V4','clustered-V4'):
        means=[]
        for arm in ('uniform','uniform_feedback_oari'):
            vals=[p['metrics']['volume.relative_l1']['mean'] for p in summary['panels'] if p['arm']==arm and p['stage']=='C1'
                and p['condition']==condition and len(p['draws'])==(1 if condition=='Val-A-V4' else 5)]
            means.append(None if not vals else sum(vals)/len(vals))
        control.append(means[0]);treatment.append(means[1])
    note='2026-10-20 **MRGR-UniformFusion最终候选及补充评价已完成**：主线四种子、均匀融合骨干匹配消融、四种经典算法比较与独立验收均归档；反馈OARI保持独立支线。沿用此前已使用的35例，未获得新的独立案例确认，不按本次测试重选部署种子。[中文总报告](docs/reports/mrgr_uniform_final_v1_zh.md)、[模型身份](docs/reports/mrgr_uniform_final_model_v1_zh.md)、[执行协议](docs/mrgr_uniform_final_v1_protocol_zh.md)。'
    if evaluation_stage():
        assert accepted['result_scope']=='four_run_priority_3D_evaluation'
        assert not accepted['overall_work_complete'] and len(accepted['deferred_ablation_runs'])==7
        note='2026-10-20 **均匀融合与反馈OARI双种子三维配对评价已完成**：两种子Control和两种子Treatment的35例/17布局评价、四种经典算法历史对比及独立验收归档。7个骨干消融暂缓，是否将反馈OARI作为最终模型等待审阅，未自动晋级。沿用此前已使用的35例，属于补充评价。[阶段总报告](docs/reports/mrgr_uniform_final_v1_zh.md)、[OARI结果](docs/reports/mrgr_uniform_oari_v1_zh.md)、[执行协议](docs/mrgr_uniform_final_v1_protocol_zh.md)。'
    readme=PROJECT/'README.md';content=readme.read_text()
    if note not in content:
        assert content.startswith('# GeoPhy-Set3D\n')
        content=content.replace('# GeoPhy-Set3D\n','# GeoPhy-Set3D\n\n'+note+'\n',1)
        readme.write_text(content)
    allowed=['README.md','configs/mrgr_uniform_fusion_v1.json','scripts/uniform_final_v1',
        'scripts/uniform_feedback_oari_v1','docs/mrgr_uniform_final_v1_protocol_zh.md',
        'docs/reports/mrgr_uniform_final_progress_v1_zh.md',
        'docs/reports/artifacts/mrgr_uniform_final_v1',*[str(Path('docs/reports')/n) for n in REPORTS]]
    def authorized(name):return any(name==p or name.startswith(p+'/') for p in allowed)
    staged=git('diff','--cached','--name-only').splitlines()
    assert all(authorized(p) for p in staged),'Unrelated staged changes: preserve and block automatic commit'
    subprocess.run(['git','add','--',*allowed],cwd=PROJECT,check=True)
    staged=git('diff','--cached','--name-only').splitlines()
    assert all(authorized(p) for p in staged)
    assert not any(Path(p).suffix in ('.pt','.pth','.npy','.npz') for p in staged)
    check=subprocess.run(['git','diff','--cached','--check'],cwd=PROJECT,capture_output=True,text=True)
    assert check.returncode==0,check.stdout[:5000]
    if staged:
        title='Evaluate uniform feedback OARI on matched seeds; defer remaining ablations' if evaluation_stage() else 'Freeze MRGR-UniformFusion and complete matched supplementary evaluation'
        subprocess.run(['git','commit','-m',title],cwd=PROJECT,check=True)
    subprocess.run(['git','fetch','origin'],cwd=PROJECT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=PROJECT,check=True)
    subprocess.run(['git','push','origin','main','experiment/uniform-feedback-oari-v1'],cwd=PROJECT,check=True)
    head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/main').split()[0]
    assert remote==head
    write(BASE/'github_sync.json',dict(at=now(),complete=True,head=head,remote_main=remote,
        source_branch='experiment/uniform-feedback-oari-v1',unrelated_work_preserved=True,
        post_sync_status=git('status','--short')))
    print('Accepted results synchronized to GitHub main',head,flush=True)

if __name__=='__main__':main()
