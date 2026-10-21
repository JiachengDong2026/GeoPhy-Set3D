"""Final editorial review only; numerical sources and older reports are retained."""
from runtime import *
import shutil


def run():
    validate()
    parent = verified(BASE / 'acceptance_final.json')
    original = verified(BASE / 'acceptance.json')
    verify_hashes(original['exports'])
    source = Path(parent['final_report'])
    assert sha(source) == parent['final_report_sha256']
    reviewed = source.with_name('mrgr_oari_mechanism_v2_reviewed_zh.md')
    text = source.read_text().replace('直接事实是上述有限输出响应、历史污染、确定性冗余',
                                     '直接事实是上述有限输出响应、历史污染审计为零、确定性冗余')
    text = text.replace('以及历史P1的attention主要来自污染。',
                        '以及其他含局部缺视角输入中P1可能受mask污染影响（本轮数据未触发）。')
    text += '\n\n## 后续实验的资源优先级\n\n优先复用已完成的 Uniform Fusion `uniform_no_ray_relations_seed20261011`，在固定案例0、17、34及普通/聚簇draw0上，只补缺失中间量和6份obs_token冻结输出；本轮主模型结果及无射线模型已有基准全部复用。先判断训练后token路径依赖是否重分配，暂不新增训练。方位角训练对照仅为未来明确需要重训练增量价值时的条件性备选。详细方案见证据目录 `next_step_zh.md`；本轮没有执行这些后续实验。\n'
    assert '直接事实是上述有限输出响应、历史污染、' not in text
    if not reviewed.exists():
        reviewed.write_text(text)
    else:
        assert reviewed.read_text() == text
    destination = ARTIFACT / reviewed.name
    if not destination.exists():
        shutil.copy2(reviewed, destination)
    receipt = dict(at=now(), completed=True, scope='report editorial review only; no numerical or experimental change',
                   report=str(reviewed), report_sha256=sha(reviewed), report_artifact_sha256=sha(destination),
                   source_sha256=sha(__file__), parent_report_sha256=sha(source),
                   parent_acceptance_sha256=sha(BASE / 'acceptance_final.json'),
                   numerical_acceptance_sha256=sha(BASE / 'acceptance.json'), protocol_sha256=sha(BASE / 'protocol.json'),
                   corrections=['zero observed historical contamination distinguished from hypothetical risk',
                                'prioritize reuse of existing Uniform Fusion no-ray checkpoint before any new training'],
                   original_exports_still_valid=True, new_forwards=0, GT64_read=False, GT128_read=False, training=False)
    if not (BASE / 'acceptance_reviewed.json').exists():
        freeze(BASE / 'acceptance_reviewed.json', receipt)
    for name in ('acceptance_reviewed.json', 'acceptance_reviewed.json.sha256'):
        if not (ARTIFACT / name).exists():
            shutil.copy2(BASE / name, ARTIFACT / name)
    current = ARTIFACT / 'CURRENT_REPORT.json'
    if not current.exists():
        freeze(current, dict(report=reviewed.name, acceptance='acceptance_reviewed.json',
                             report_sha256=sha(reviewed), acceptance_sha256=sha(BASE / 'acceptance_reviewed.json')))
    print('FINAL EDITORIAL REVIEW PASSED; NUMERICAL EVIDENCE UNCHANGED')


if __name__ == '__main__':
    run()
