# 普通集合交互最终主模型冻结报告

生成时间：2026-10-23T14:29:41.705536+08:00。所有数字由本轮机器证据自动生成。

已冻结 `uniform_plain_set_interaction_seed20261011`，最佳epoch 298，参数量 342,209。主种子由用户指定，不按历史测试误差重新排名。

| 身份 | 种子 | 最佳epoch | 参数量 | checkpoint SHA256 |
|---|---|---|---|---|
| primary_user_designated | 20261011 | 298 | 342209 | `0ccea9b24a77e9d5fac025c626a98b167c3ab57fa202688e3e4acc75261bc18c` |
| second_seed_reference | 20261019 | 279 | 342209 | `fe19095926c6fad4194cfd6addcf96fed523b94631e754d021600f1b7313525d` |

两份checkpoint均实际加载并严格校验state dict、内部config、protocol哈希、epoch及GT标记，文件大小与实际路径见[model_manifest](artifacts/mrgr_plain_final_v1/model_manifest.json)。完整300epoch/24,900次提交更新及日志最低验证H4选轮已逐行复核；last.pt保留优化器/调度器/RNG恢复状态。

架构：Encoder → GeometryLifting → Plain Set Interaction → 多尺度MRGR → UniformFusion → PRF。当前主模型不是历史MRGR、MRGR-UniformFusion或LR4E4。继承config里的`geolift_oari_mrgr_res3d`名称不能独立决定工厂；必须通过普通交互arm分派。

训练166案例、开发验证36案例。线性p99.5归一化标尺32284.894580078173，仅训练统计；数据划分/数据文件与统计哈希已登记。batch2，300轮，AdamW lr2e-4/wd1e-4，cosine300，FP32/TF32关闭，clip1；损失hold+0.5cross+0.1observed+0.05consistency，全部沿用原始配置。

C0为初始解码；C1为带符号测量残差反馈后的解码。输出[B,1,64,64,64]、XYZ端点[0,0.6]m、CH4质量分数[0,1]。正式P_diff在质量分数体上三线性采样后转换摩尔分数，K96中点积分，观测单位ppm·m。

访问日志、训练摘要及源码显示这两次正式训练和验证选轮未读取测试GT；本轮无optimizer、无反向训练、无种子/超参搜索。历史35案例已参与模型研发，故不能据此声称整个架构开发完全未利用测试信息。

CPU与独占GPU smoke均通过：严格加载、前向形状/有限值/范围、eval/no_grad及权重buffer规范张量哈希前后一致。主模型全部595组预测已封存，原checkpoint哈希在最终验收再次核验。

复现（在仓库根目录）：
```python
import sys, torch
sys.path.insert(0, "scripts/plain_final_v1")
from context import verified, BASE, model_for, PLAIN
m = verified(BASE / "model_manifest.json")["models"][0]
model = model_for(m["config"], PLAIN, "cpu").eval()
ck = torch.load(m["checkpoint"], map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=True)
# 测量输入及布局由context.dataset和batch_for构造，无GT输入
```

环境、完整配置、源码快照与协议位于`/scratch/djc/GeoPhy-Set3D/mrgr_plain_final_v1`；Python可执行文件`/scratch/djc/envs/gas-torch/bin/python`。轻量身份及哈希已导出到[证据目录](artifacts/mrgr_plain_final_v1/model_manifest.json)。

本轮沿用已参与研发/比较的35个历史案例，全部比较与区间均属探索性结果，不能替代独立盲测。bootstrap单位为案例；多draw先在案例内平均；两个训练种子分别报告，案例区间不是跨种子不确定性。
