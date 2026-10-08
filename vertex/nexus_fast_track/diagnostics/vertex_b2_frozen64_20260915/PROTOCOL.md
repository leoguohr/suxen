# B2冻结64与后续C执行协议

用户授权：先冻结累计step2000/B2 update1000，UID nexus_2k_000105、depth9、GT parents、种子14000000..14000063、20步Euler、threshold0.5、无参数更新。唯一通过门槛是64/64整数坐标集合正确。点数、漏点、多余点、连续占据误差和阈值余量均为诊断，不追加门槛。
权重SHA256：9dc3cc9ad8ba64a6aaf947df37f1451dc8e4532c2a8eb1f779e989042edc2f3c。
原持久路径 /guohaoran/tmp/vertex_b2_restart1_20260915/checkpoint-last.pt，当前本地盘副本 /tmp/vertex_b2_step2000.pt；已逐字节流式复制、校验原文件SHA并回读新副本校验。
原B2 noise13000002从上一轮NPZ原样读取，20/40步×BF16/FP32配对；禁止CPU按seed重生噪声替代。新64使用CUDA私有Generator产生并在采样前保存实际噪声。失败则对同份保存噪声追加20 FP32、40 BF16/FP32诊断，不重训。
用户提及的自备脚本未实际附上，本次实现evaluate_vertex_b2_frozen.py；不宣称运行了用户脚本。

C只有本轮64/64通过才启动，继续同B2权重/Adam/CPU及CUDA RNG。新增1800更新，lr1e-5、WD0、clip1、accum8、BF16，无warmup或结构/loss/阈值改动。每micro独立随机t、噪声；depth=1+((update-1)*8+micro)%9。每9更新各层8micro；完整预算每层1600micro。
GT_d=unique(vertices//2**(9-d))。根层1parent/8占据，其他层8parents/8占据，均实际断言。
每100更新评估并备份新C文件，保留原B2文件。C初始也评估，不消耗恢复的训练RNG。
开发评估固定4个种子15000000..15000003，每seed分别做9层GTparents独立采样和从根全树20步Euler采样；终验固定8个种子16000000..16000007仅在1800更新完成后执行一次。每层实际噪声种子=base_seed+depth*1000。这些数量和种子是本轮预先固定的工程选择，不是论文规定。主报完整生成坐标集合成功数，局部9层成功率独立报告，连续误差不设门槛。
全树路径不插入GTparents、不强制非空、不topk修复。保存所有层输入parents、实际noise、连续estimate、输出cells、GT_d、最早失配层。为防止失控树OOM，输入parents超过4096时将该轨迹记为失败并保存当层父格，不裁剪后继续或伪装成功。空树继续记录后续空层。
模型恢复后严格比对Adam全部状态、参数组与RNG；保持已加载权重。测试用CPU小模型结果不可当成完整GPU训练结果。
C完成后停止，先报告，不在本轮自动进入D；后续按2→4→10/20扩展。
