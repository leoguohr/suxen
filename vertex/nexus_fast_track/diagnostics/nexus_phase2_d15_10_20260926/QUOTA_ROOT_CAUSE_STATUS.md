# GPFS 写入受限排查：2026-09-27

结论：已独立复现 `EDQUOT / errno 122`。尚未取得 GPFS 后台的实际配额记录，因此不能确定是仍然超额、计账未更新，还是容器外的占用导致。不能把已删除文件的逻辑大小当作已返还的可写额度。

## 已执行及证据

- `findmnt` 与 `/proc/self/mountinfo`：`/guohaoran` 挂载自 `storage[/home/huangxiangru/guohaoran]`，类型 GPFS，读写挂载；设备号 `0:112`。未访问此挂载外的其他账户目录。
- `/proc/self/uid_map`、`gid_map` 均为 `0 0 4294967295`；当前 UID/GID 为 0/0，没有容器 UID 重映射。抽查的新旧 checkpoint 均为 UID/GID 0/0。
- 全文件系统 `statvfs`：仍有约 3.6 PiB 全局可用空间，且数十亿全局空闲 inode。这个值不是当前账户或 fileset 的可写配额。
- 清理清单中 22 个 R1-D15 与 6 个 D2/D4 中间 checkpoint 路径均已不存在。当前容器的 `/proc/*/fd` 和 `lsof +L1` 没发现已删除但仍打开的文件；不覆盖宿主机及其他节点。
- 本日此前的独立写入实验：Phase2 的 `run_10` 与旧 `vertex_d2_20260916` 目录均在写入 1,325,400,064 字节（1264 MiB）后报 errno 122。两个探针文件已删除，见服务器 `quota_directory_recheck_20260927.json`。这不是模型保存函数独有的错误。
- `mmlsquota`、`mmlsattr`、`mmrepquota`、`libgpfs.so` 和 GPFS 管理日志在容器内均不可用。尝试只读 Linux `quotactl(Q_GETQUOTA)`，对 `/guohaoran` 返回 errno 15（需要块设备），对 `storage` 返回 errno 2，未取得配额值。未更改配额。
- 只读统计整个 `/guohaoran` 的 `du` 未及时完成，已中断；没有得到整个目录占用总量，不据此推断是否低于后台限额。
- 另有独立 GPFS 挂载 `/ssdwork/guohaoran`，源为 `ssdwork[/home/huangxiangru/guohaoran]`，设备 `0:113`；目录内存在 `nexus_fast_track`。仅查看，未写入、未迁移；其实际配额和保留期限尚未核实。

## 需要学校存储管理员读取的记录

请在具有 GPFS 管理工具的节点上，先定位实际路径对应的 fileset（管理节点挂载点可能不同）：

```sh
mmlsattr -L /home/huangxiangru/guohaoran/tmp/nexus_phase2_d15_10_20260926/run_10
```

然后用输出中的真实 fileset 名代入以下命令，不要根据目录名猜测：

```sh
mmlsquota -j FILESET_NAME storage
mmlsquota -u 0 storage:FILESET_NAME
mmlsquota -g 0 storage:FILESET_NAME
```

需要返回：配额类型/归属、current usage、soft limit、hard limit、in_doubt、grace，以及该 fileset 的计账范围。若读数明显陈旧，请管理员按本校运维流程确认是否刷新或检查计账；本轮不自行运行全局配额重算、不调整限额、不删除快照。

只有取得这些值，才能判断删除后不恢复写入是“实际仍超额”还是“占用/预留计账未释放”。如果配额覆盖挂载之外的同一 fileset，还需管理员核实该范围的占用；当前容器不扫描其他账户目录。

参考：[IBM mmlsquota](https://www.ibm.com/docs/en/storage-scale/6.0.0?topic=reference-mmlsquota-command) 说明 `current usage + in_doubt` 也会限制可分配空间；这是一种待核实机制，不是本机已确认根因。[IBM mmlsattr](https://www.ibm.com/docs/en/storage-scale/6.0.1?topic=reference-mmlsattr-command) 说明 `-L` 可显示文件所属 fileset。

## 训练状态边界

本轮最后读取时，进程 2036 存活、训练 update 2211/6000。临时盘 update 2200 / step 8200 的完整 checkpoint SHA256 为 `8deeaa3c7d567d2d4977afc8d51384f8533082b64b4c582a201ff0a6e80951c8`，字节数 27,990,506,330，NVME 回读校验完成。持久盘最近的完整 checkpoint 仍是 update 1600；本轮未修改训练代码或协议，也未再删除训练文件。

## 另一个任务成功保存的对照

用户随后提供另一个任务的“两份轮换恢复＋每1000步永久留档”描述。本次只读核实该任务实际恢复点为 `/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_resume16760_20260927/run/recovery-a.pt`，完成更新次数 20180，文件大小 2,960,559,104 字节（2.7572 GiB）。独立回读 SHA256 为 `a3402fa9a924120eaa18b9ca4b4c3f081e36631378df43ffe2635649f89be43c`，与 `recovery-latest.json` 完全一致，读取期间 inode、mtime 和大小未变化。另一个恢复槽 b 的清单记录 update 20160，其权重未在本轮重新计算哈希。

`findmnt` 确认它使用 `ssdwork` 文件系统，本任务原保存位置使用 `storage` 文件系统。成功的 2.76-GiB 文件大于此前 `storage` 写入实验约 1.23-GiB 的失败边界，因此不能只用“文件更小”解释；两者实际写入不同的挂载。轮换保留策略控制长期占用，不能直接解除另一个文件系统已经返回的配额错误。

此对照证明 `/ssdwork` 上存在完整可核验的恢复文件，不证明本任务约 26.1-GiB（28 GB）的完整状态目前一定可写，也没有核实 SSD 的平台保留期限。未修改 VAE/AE V2 文件、保存协议或 GPU 进程。本任务状态在对照检查时为 update 2346/6000，仍使用原 NVME 保存入口。原 `storage` 的具体配额记录仍待管理员查询。

用户随后授权实际尝试。本任务 update 2400 的 27,990,506,330 字节 checkpoint 已成功复制至 SSD 本任务独立目录，fsync 和完整回读 SHA256 均通过（63.596 秒）。这是本模型真实大文件的成功写入证据；详见 `SSD_WRITE_VALIDATION.md`。后续自动保存目的地尚未切换，原存储配额的后台原因仍未查明。

随后用户要求完成自动保存切换：已在 update 2600 完整保存后迁移，自动保存、日志和最终评估均改到 SSD。update 2601 的首份实际自动保存 checkpoint 经独立哈希与 CPU 状态检查通过，最后查看训练已到 update 2629；详见 `SSD_MIGRATION.md`。原 `storage` 配额的后台原因仍未查明，但当前训练已不依赖该挂载写入完整 checkpoint。
