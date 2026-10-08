# POINT_NATIVE_XYZ_GATE

本轮结论：**当前 depth-9 叶格中心 XYZ 表示未达到老师连续坐标标准；D2 既有整数生成成绩没有被推翻。** 保留 R1，没有训练、改主干/loss/先验或触碰 VAE/AE V2。工作均在本独立目录。

**执行边界：D2 完整模型本轮尚未重放。** 最终 checkpoint 已在新服务器两次 CPU 读取核验。GPU0 的独立分配与最多 30 GPU 分钟预算已一次性询问，尚未获答复；两张卡均未使用。不能将下面历史数组复算、表示往返或小模型测试写成新模型成功。

## 结果与数据边界

|执行内容|本轮结果|证据|
|---|---|---|
|浮点 GT 表示往返，CPU|D2/D4 4 个 + CAD50 50 个|`outputs/representation.json`、逐样本 CSV、`roundtrip/*/arrays.npz`|
|D2 历史真实预测复算|32/32 整数集合、32/32 全树正确；288 个逐层源文件匹配原 final_once 哈希|`outputs/B_historical_D2/evaluation.json`；没有重新运行大模型|
|独立生成接口 CPU 测试|9 层 parents/noise/estimate/prediction 与原采样函数逐数组一致，180 调用；另做缩小 R1 前向|`outputs/interface_test/result.json`；解析 oracle 与 smoke 均不计生成成绩|
|老师 A：完整文本条件模型，保留坐标先验|seed34567 复用已核验历史50份；seed98765 本轮 CPU 实跑50份；100/100通过原门槛|`outputs/teacher_A/generation_manifest.json`、`evaluation.json`|
|我方 B：CAD50 真实生成|未运行；所查 CAD50 缓存仅有浮点 vertices/faces、文本特征等，没有已核验的点云/法向条件|没有用 CAD 顶点冒充点云；没有训练或伪造条件|

真实预测与哈希保留在 `outputs/teacher_A/seed-*/*.npy` 和 `outputs/B_historical_D2/seed-*/UID/prediction.npz`，后者明确标记为**历史预测解码**。每份 `generation.json` 记录原数组路径/哈希、噪声、顺序、mask 来源、固定变换；独立评价前后预测哈希一致。汇总逐样本表：`outputs/native_and_xyz_per_sample.csv`。

原始 GT 是自己 stage1 的 **canonical_world** 浮点顶点，通过已保存 stage2 变换到最长边2；不是整数 GT 自比较，也未宣称取回更早的原始 CAD 文件。CAD50 使用老师缓存中已归一化的真实浮点 GT（最长边2，中心0，逐条核验）；更早世界坐标未提供，当前到老师空间为恒等变换。CAD50 `sample_id=index`、prompt 及两份文本特征数组逐项相等；未假设原100条有文本。

|UID|原始记录 → 解码点数|重复记录数 / 真量化碰撞数|原记录逐顶点对应 RMSE|最大欧氏误差|
|---|---:|---:|---:|---:|
|nexus_2k_000105|24 → 8|16 / 0|0.00125557|0.00217472|
|nexus_2k_000195|140 → 52|88 / 0|0.00126709|0.00289814|
|nexus_2k_001045|194 → 54|140 / 0|0.00119743|0.00304656|
|nexus_2k_001885|672 → 176|496 / 0|0.00118631|0.00334285|

这4份原始记录含重复坐标，原 GT 最小间距均为0，误差/间距比未定义；老师一一匹配要求点数相等，故**原始记录集合的匹配 RMSE 不填写、不宣告通过**。上表 RMSE 用编码时明确的原记录→格子映射计算，属于表示诊断。预处理日志确认这4份没有额外未引用顶点删除或面删除。

下面仅为**显式去除完全重复 GT 坐标后的诊断**，不替代原始点数验收。D2 历史每个 seed 解码 XYZ 均等于前两行结果。

|UID|去重 GT 匹配 RMSE|去重 GT 最小间距|最大误差/最小间距|
|---|---:|---:|---:|
|nexus_2k_000105|0.00125557|0.19270240|0.011285|
|nexus_2k_000195|0.00127163|0.04839939|0.059880|
|nexus_2k_001045|0.00121437|0.01672588|0.182147|
|nexus_2k_001885|0.00120453|0.04716915|0.070869|

CAD50：48份点数不变，匹配 RMSE 范围 **0.0008802869–0.001953125**，均大于1e-4；另外2份存在真实量化碰撞，点数不符，因此没有一一匹配 RMSE。`sample_id=0` **68→8**、`sample_id=13` **16→8**；对应最大逐原点误差约0.00292203、0.00287357，GT最小间距约0.000575225、0.000815766。所有54份的完整指标、碰撞映射和数组均保留。**此为实际规则往返误差，不称理论最小误差。**

评价直接复用 `teacher_candidate/replay.py:25` 的 `point_metrics`：Hungarian 最小化总平方欧氏代价；RMSE=√(Σ三坐标误差²/(3N))；max_error=max逐点L2；spacing=min GT 两点L2；ratio=max_error/spacing。点数不同不匹配，spacing=0的比例不定义。来源门槛为RMSE1e-4、ratio0.1、点数全对。没有重新归一化、拟合、修正或排序后回写预测。

## 当前冻结代码定位

读取了交接 `STATUS_AND_EVIDENCE.md`、`NETWORK_AND_PROTOCOL.md`，副本在 `evidence/source_docs/`。**本轮使用 D2 snapshot 冻结代码，而非最后 D4 部署版**；`evidence/code_provenance.json` 逐文件确认冻结 Python 与 D2 snapshot 相同。

|路径与函数|本轮相关结论|
|---|---|
|`evidence/preprocessing/stage2_surface_condition.py:112 normalize_vertices`|c=(bbox_min+bbox_max)/2，h=max边长/2，u=(world-c)/h；先float64计算再存float32；逆变换world=u·h+c。导出只用固定 c,h。|
|`evidence/preprocessing/stage3_octree_labels.py:117 quantize_vertices`；`:169 merge_and_clean_quantized_mesh`|q=clip(floor((float64(u)+1)·256),0,511)，同格合并再显式清理；对4份真实数据与保存的原始q逐元素一致。|
|`frozen_code/mini_nexus/octree.py:21 quantize_vertex_cells`、`:41 decode_leaf_centers`、`:65 build_octree_levels`、`:98 expand_occupied_children`|x̂=-1+(q+0.5)/256；子格ID=4x+2y+z；训练去重、父格分组，生成按父行再子ID展开；不能把原生顺序当XYZ字典序。|
|`frozen_code/mini_nexus/data_2k.py:224`；`:338 collate_nexus2k_samples`|实际条件是固定8192×(XYZ+normal)；训练顶点/父格 padding 的 mask 来自真实长度。新推理导出单物体无padding，mask是模型生成N个点的全True，未读GT mask。|
|`frozen_code/mini_nexus/training.py:54 VertexStageSystem.forward`；`flow.py:11 flow_matching_batch`、`:63 euler_integrate`|Y为0/1占据，ε~N(0,I)，x_t=(1-t)ε+tY，监督Y-ε；VecSet真实进入flow；mask内FP32速度MSE。Euler t=i/S，dt=1/S，从0噪声到1数据，方向和尺度一致。|
|`frozen_code/scripts/train_vertex_d2.py:213 evaluate`、`:315`；`train_vertex_c.py:29 sample_tree`；`mini_nexus/vertex_evaluation.py:119 sample_level`|D2真实终验调用链：seed外层、A/B内层；9层20步；每层private generator(seed+1000·depth)，从根用模型父层。旧函数读GT仅做目标/指标及噪声模板shape，新接口shape只取生成parents。|
|`scripts/native_export.py:export_tree`；`scripts/evaluate_exports.py:evaluate`|生成只读条件NPZ、检查点、seed/协议/固定变换；先保存整数顶点、XYZ、N、mask、原序/字典序置换、逐层占据与哈希，再由独立入口读GT。容量超4096中止计失败，无top-k/强制非空。|

D2 原协议 20步是**每层**，非空9层每树180次网络调用，32树5760次；本轮CPU oracle确实计数180。CAD50 若沿用此八叉树接口配置100步，是**每层100、每树最多900次**（空层跳过）；老师 A 是每点集100次。没有为了表面对齐数字而更改D2协议。双方若比较CAD50，应标为**同一GT、不同条件任务**，本轮没有B条件适配或成绩。

## 权重及恢复边界

D2实际可读：`/guohaoran/tmp/vertex_d2_resume3000_20260916/checkpoint-last.pt`，27,990,380,250 bytes，SHA256 **2384b430fa52d5012cd793fc2781c79791814f6a4b2e8ac120a711fb91112e1b**，累计7400 / D2 3600。model 907张量、2,332,430,344元素；Adam 1组907参数及907状态，moment shape与model登记顺序一致，step全部7400；lr1e-5、WD0。CPU RNG5056 bytes、CUDA RNG 1×16；没有scheduler字段（原恒定LR）。只做CPU读取/覆盖检查，没有恢复运行，跨环境逐位续训未验证；参数组未保存名称。详见两个 checkpoint audit JSON。

老师完整点checkpoint `teacher_checkpoint_audit.json`：18层、宽144、prior512、最大274点，step160000；含完整denoiser、coordinate_prior、residual_scale。该文件没有完整Adam/RNG，不把专门先验训练状态当完整去噪网络续训状态。教师seed98765是本轮CPU随机流，不能称老师原GPU噪声逐位复现。

## 已执行命令与下一步

实际CPU命令、日志与未执行的GPU方案见 `COMMANDS.md`；当前状态见 `RUN_STATUS.json`。代码/模型原件没有修改。交付不含大权重、凭据或独立V2工程。

**唯一下一步：获得已询问的独立GPU0与30分钟预算后，用已核实D2权重完成原16对种子冻结重放，逐层比较保存的实际噪声与整数输出。** 不重新训练，不先切到CAD50。当前连续坐标门槛不满足已有直接的表示检查证据，不能靠继续压占据loss宣称该门槛通过。
