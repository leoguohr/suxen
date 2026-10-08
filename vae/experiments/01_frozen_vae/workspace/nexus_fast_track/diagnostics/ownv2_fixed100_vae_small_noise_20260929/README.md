# OwnAE-v2：小噪声与小KL接入

用户已明确确认从step34220接入。这个AE起点尚未达到原Face F1≥0.997门槛，不能称为100条AE已通过。其实际Face F1为0.9960223877867064，Edge/Face联合严格成功28/100。

## 固定方案

独立分支完整继承AE权重、AdamW、RNG和调度游标；仅尚未训练的logvar专属weight置0、bias置-6，新组AdamW状态为空。原logvar clamp[-20,10]保留。训练完整网络，z=mu+exp(logvar/2)*epsilon，每条训练forward产生一份新epsilon，在同次重计算中复用。KL=0.5*mean_vertex_channel(mu^2+exp(logvar)-1-logvar)，每mesh内部平均再随五mesh等权平均，beta固定1e-6。

重建目标保持原Hard4、全部Edge pair与按epoch/UID生成的Face负例。原AdamW实际LR=1e-4、betas=(0.9,0.999)、eps=1e-8、wd=.01；logvar独立组同LR和优化选项，global clip=1。无新warmup。数学后端、评分、输入编号、数据及模型结构保持现有定义。

## 有限预算与验收

新增500次有效optimizer更新，每次microbatch=1、累积5条；原100条每epoch20次更新，本轮25epochs，每mesh新增25次参与，末尾累计34720。新增0/100/250/500保存完整状态并全100条验收mu；起点和末尾使用同一预先固定5组噪声，每组评价全100条实际Edge与预测Edge图枚举的Face。另记录posterior、KL分解、真实扰动、梯度与实际参数位移，起点比较独立重建与KL梯度量级。NaN/Inf或协议不符保存现场并停止；到500结束，不自动延长。

## 起点证据

- source/ae-step34220.pt位于服务器独立分支根目录；原文件只读保留。
- SHA256：c9cd45dbd011c4adec9f3916bcdeb85dbe4bb2dcd3949db313356f1909f5ce95。
- model SHA256：508cb0f40560a2ea060d0f6cfbd6131a4a7aa34b42e02898a0c6280d063409d5。
- 原412个AdamW参数状态的step均34220；logvar不在optimizer。
- 原cursor：epoch1711，group0；每条participation1711。
- evidence/source_checkpoint_audit.json包含CPU实核记录；source/ae_mu_baseline.json为完整原AE验收。

## 模型分工

Astra xhigh负责协议与代码审查；gpt-6-sol xhigh负责独立code/实现；主控核验、集成、运行与交付。调用标识见execution_roles.json。

## 状态

代码准备中；尚未执行本轮GPU前向或optimizer更新。实际状态以repro_outputs/status.json和服务器run/status.json为准。
