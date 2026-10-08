# CAD50：Graph顺序与激活的四支单因素训练对照

## 目标

各2000次五mesh更新，固定种子0、同一份随机参数，比较完整V2、仅撤回Graph LN位置、仅撤回Graph GELU、仅撤回Encoder FFN GELU。保留无bias邻居投影。有限预算只说明该起点与配方下的效果。

## 固定配置

V2 Encoder512、latent512、12组Graph＋Transformer，Decoder1024、16完整Transformer块，所有Decoder FFN保持4096，Fourier39，两个共享32维head。全部重建层训练；logvar冻结，μ路径，KL=0。

fresh AdamW lr=1e-4、betas=(0.9,0.999)、eps=1e-8、wd=0.01；100次有效更新warmup，clip=1。microbatch1，五条完整mesh等权累积后一次更新。Edge全pair、Face全部GT＋每epoch/UID固定seed抽取ceil(1.5F)唯一负例。Hard4整mesh四组归约、空组0、内部/4。FP32 SDPA MATH、确定性Graph、TF32/autocast关闭、逐块重计算。

每支2000更新＝200完整CAD50 epoch＝每条200次直接参与。四支步数不相加为同一模型进度。所有分支参数名称、形状、初值和参数量相同。原项目源码、原100条、VAE、diffusion与历史模型保持原状态。

## 正式训练命令

初始化只生成一次；读取的旧目录仅用于无损CAD数据和历史规则，不加载其已训练权重。

```bash
/opt/conda/bin/python -B -u launch_all.py --source /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01
```

## 验收与保存

新增0、500、1000、1500、2000冻结同一模型，对50条真实Encoder→μ→Decoder评价。全部Edge pair，Face从预测Edge图完整流式枚举triangle，未进入候选GT面计FN，logit>0。逐mesh错误、完整预测分片与严格成功UID都保存。500/1000/1500/2000保存完整model/AdamW/RNG和数据游标，可明确恢复。每支到2000停止。

## 已执行测试

CPU小型合成测试验证V2基线与原源码前向及梯度逐位一致、四支初值一致、梯度连通、logvar冻结、数据顺序与负例恢复、Adam更新边界恢复、Hard4分块梯度和完整三角枚举。CPU合成测试不称为真实CAD训练结果。

## 结论边界

旧A/B整体版本对照不能孤立归因Graph或激活。预训练权重上撤回操作属于推理依赖检查；本轮从相同随机参数训练，才能比较操作对学习过程的影响。单种子短预算不等于全局因果证明。
