# 固定100条 Decoder hidden 的共同 Edge 读出

源checkpoint固定为第二轮Face难负例分支epoch900/update22500，SHA256为
`428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79`。
100 UID沿用原selection.json，保持原局部顶点编号。先运行真实math00 Encoder→μ→Decoder，缓存末端LayerNorm输出H。初始化只使用此源checkpoint的原Edge head。

唯一可训练变量是一份W(32×1024)和bias(32)。每mesh独立中心化 `E = HW^T+b - mean_vertex(HW^T+b)`，然后执行原16+16空间/时间平方距离差评分和原scale。所有i<j pair均参与。GT来自原完整mesh边图，判正定义为logit>0。

每条Edge Soft4的membership为sigmoid(logit)，不detach。四组各用 `sum(w*BCE)/(sum(w)+1e-8)`，再除4；FP32 reduction。有效归档函数中的旧注释不决定计算图：执行语句不含detach，预检还对照源网络验证了head梯度。

每次optimizer更新都完整遍历100条。在不改变head的情况下顺序计算每条loss，按1/100反向累积，最后统一global clip=1与一次Adam step。目标是 `mean100(Edge Soft4)`，没有额外外层1/4，没有Face、KL或MSE。

先完成无更新成本预检，再固定2000更新的预算。fresh Adam LR=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0。预算与LR是候选实验设置；不因中途指标调整，不自动延长。

evaluations.jsonl中的step=t表示已执行t次更新的同一head状态，其100条计数全部在下一次step之前得到。updates.jsonl的update=t记录从t-1到t的梯度、clip与实际参数位移。每条直接参与2000次，不是每条独立拥有2000步的专属head。

严格成功只能来自同一head状态下，所有100条Edge FP=FN=0。逐mesh历史曾成功次数不能替代此标准。此实验不执行实际Face验收，也不把新head自动安装到主模型。
