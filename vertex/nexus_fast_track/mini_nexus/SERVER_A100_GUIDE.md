# Nexus A100 服务器使用说明（纯新手版）

## 1. 本次服务器布局

本项目只把长期文件写入服务器的持久化目录 `/guohaoran`：

```text
/guohaoran/
├── envs/nexus-algo/                 # Nexus 独立 Python 环境
├── cache/                           # pip、Torch、Hugging Face 等缓存
└── nexus_fast_track/
    ├── data_pilot32/                # 32 条 pilot 原始数据及 Stage 1/2 结果
    ├── external_references/         # Hunyuan3D-2.1、MeshFlow 参考代码
    ├── mini_nexus/                  # Nexus 独立教学复现
    └── Nexus算法复现_纯新手讲义.md
```

不要把训练数据、权重或输出放进 `/root`、`/tmp`、`/workspace` 或服务器根目录。这些位置可能位于随实例释放而消失的 overlay 文件系统。

## 2. 每次登录后怎么进入环境

```bash
source /guohaoran/nexus_fast_track/mini_nexus/scripts/server_env.sh
```

这一行依次做四件事：

1. 把各种缓存位置指向 `/guohaoran/cache`；
2. 把 `mini_nexus` 加入 Python 的模块搜索路径；
3. 激活 `/guohaoran/envs/nexus-algo`，不修改服务器 base 环境；
4. 进入算法项目目录。

看到下面两行，说明路径正确：

```text
Nexus root: /guohaoran/nexus_fast_track/mini_nexus
Python: /guohaoran/envs/nexus-algo/bin/python
```

## 3. 首次安装（只需要管理员执行一次）

服务器镜像已经带有针对 A100/CUDA 12.4 编译的 PyTorch。为避免重新下载一套庞大的 CUDA 环境，本项目创建独立 venv，但只读复用镜像的 PyTorch：

```bash
mkdir -p /guohaoran/envs /guohaoran/cache/{pip,huggingface,torch,torch_extensions,xdg}
/opt/conda/bin/python -m venv --system-site-packages /guohaoran/envs/nexus-algo
source /guohaoran/envs/nexus-algo/bin/activate
python -m pip install --upgrade "pip<26"
python -m pip install -r /guohaoran/nexus_fast_track/mini_nexus/server-requirements.txt
```

`--system-site-packages` 的含义是：独立环境可以看见服务器镜像已有的 PyTorch/CUDA，但新增和固定的包只写入 `/guohaoran/envs/nexus-algo`，不会写入 `/opt/conda`。

这台 NVIDIA 镜像的旧 ONNX protobuf 文件不能用 protobuf 4 的 C++ 解析器，
否则创建 AdamW 时会在导入 ONNX 处失败。`server_env.sh` 因此设置
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，同时保留与 cuDF 兼容的
`protobuf==4.24.4`；它不修改服务器 base。

## 4. 三项验证

### 4.1 验证确实使用 A100

```bash
source /guohaoran/nexus_fast_track/mini_nexus/scripts/server_env.sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

必须看到：

- `True`；
- 设备名包含 `NVIDIA A100-SXM4-80GB`。

### 4.2 跑单元测试

```bash
python -m pytest -q
```

测试必须全部通过。它验证的是张量 shape、flow matching、八叉树 round-trip、拓扑解码等最小数学合同。

### 4.3 用真实 pilot 数据跑 GPU smoke test

```bash
python scripts/audit_pilot_for_algorithm.py
python scripts/smoke_real_pilot.py --uid nexus_pilot_0032 --device cuda
```

第一行检查 32 条数据中哪些已经具备算法所需的 Stage-2 文件；第二行让一条真实样本通过条件编码、顶点阶段、拓扑阶段和反向传播。它只证明数据与代码能接通，不证明训练已收敛。

## 5. 输出放哪里

所有训练输出统一放到：

```text
/guohaoran/nexus_fast_track/mini_nexus/outputs/<实验名>/
```

实验目录至少保存：

- 配置和随机种子；
- checkpoint；
- loss/metric；
- 输入 UID；
- Git commit 或源代码 hash（若有）；
- 运行日志。

不要只保存最终 OBJ。没有配置、UID 和 checkpoint 的 OBJ 无法证明实验可复现。

## 6. 这台机器能复现到什么程度

当前服务器实测只有 **1 张 A100 80GB**。它足够运行 mini-Nexus、单样本过拟合、小批量实验和模块消融；不等于论文约 20 亿参数、每个阶段 32 张 A100 的原规模训练。当前目标是先通过真实数据的最小闭环 Gate，再决定是否申请多卡资源。
