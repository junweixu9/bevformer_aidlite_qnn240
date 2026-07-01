# BEVFormer AidLite QNN2.40

多相机 BEV 时空表征 3D 检测，在 QCS8550 HTP 上端到端推理。

## 模型信息

- 架构：BEVFormer (Backbone → Temporal Encoder → Snapshot Decoder)
- 输入：六路相机预处理张量 (6×3×450×800 uint8)
- 输出：300 个九维 3D 检测框 (boxes + scores + labels)
- 目标 SoC：QCS8550 (HTP V73)
- QNN SDK：2.40
- AidLite SDK：2.4.0.265
- 框架类型：TYPE_QNN240
- 三模型常驻，同一 Python 进程十帧时序递归

## 正式性能基线

| 环节 | 耗时 |
|------|------|
| Backbone invoke mean | 22.47 ms |
| Encoder invoke mean | 355.98 ms |
| Decoder invoke mean | 37.69 ms |
| 三模型 Invoke 合计 | 416.14 ms |
| NMSFreeCoder (CPU) | 0.68 ms |
| **总墙钟 Mean** | **474.94 ms** |
| **总墙钟 P95** | **481.73 ms** |

测试合同：3 帧 warmup + 10 帧 measured，不含 Python 启动/Context 加载/原图解码。

## 目录结构

```
bevformer_aidlite_qnn240/
├── README.md
├── python/                         # 所有核心代码
│   ├── run_e2e.py                  # ★ 本地入口 (参数说明、参考验证)
│   ├── bevformer_aidlite_qnn240_e2e_performance_v1.py  # 主 Runner
│   ├── functional_mother.py        # 工具函数
│   ├── portable_numpy_nmsfreecoder.py  # 板端 NumPy NMSFreeCoder
│   ├── verify_contract.py          # float32 ε 修正合同验证
│   ├── frame009_numpy_native_reference.npz  # Frame009 参考坐标
│   └── run_bevformer_aidlite_qnn240_e2e_performance_v1.sh  # 板端启动器
├── tools/                          # 编排脚本
│   ├── run_board.sh                # ★ 板端一键远程执行
│   ├── preflight_host.sh           # Host 资产检查
│   ├── preflight_board.sh          # 板端环境检查
│   └── board.env                   # 板端连接 + 模型路径 + SHA256
├── models/
│   └── EXPECTED_SHA256.txt         # 三个 Context 期望 SHA256
└── outputs/                        # 运行结果 (每次一个 run_YYYYMMDD_HHMMSS/)
```

## 一条命令运行

### 本地 (只检查资产和连接)

```bash
bash tools/preflight_host.sh
bash tools/preflight_board.sh
```

### 板端完整执行 (六路张量 → 3D 坐标)

```bash
bash tools/run_board.sh
```

自动完成：
1. Host 资产检查 (Python 编译、SSH 连通)
2. 板端环境检查 (AidLite、TYPE_QNN240、Context SHA256)
3. 资产 Manifest 解析 (自动查找十帧资产清单)
4. 代码部署到板端
5. 板端执行 (3 warmup + 10 measured + Frame009 验证)
6. 结果回拉到 Container B
7. float32 epsilon 修正合同验证
8. 输出 FINAL_DELIVERY_ACCEPTANCE_GATE

### 仅验证已有坐标 (不重新推理)

```bash
python3 python/verify_contract.py \
  --reference python/frame009_numpy_native_reference.npz \
  --candidate outputs/run_YYYYMMDD_HHMMSS/frame009_final_coordinates.npz \
  --report-json outputs/run_YYYYMMDD_HHMMSS/corrected_report.json \
  --report-txt  outputs/run_YYYYMMDD_HHMMSS/corrected_report.txt
```

## 精度合同

| 检查项 | 容差 |
|--------|------|
| labels | 逐索引完全一致 |
| scores | max abs error ≤ 2 × float32 ε (≈2.38e-07) |
| boxes | max abs error ≤ 8 × float32 ε (≈9.54e-07) |
| shape | 与参考完全一致 |

## 文件角色速查

| 文件 | 输入 | 输出 | 执行位置 |
|------|------|------|----------|
| `tools/run_board.sh` | 无(自动) | FINAL_DELIVERY_ACCEPTANCE_GATE | Container B |
| `tools/preflight_host.sh` | 无 | HOST_PREFLIGHT_GATE | Container B |
| `tools/preflight_board.sh` | 无 | BOARD_PREFLIGHT_GATE | Container B→Board |
| `python/run_bevformer_...sh` | 5 个环境变量 | Python Runner 进程 | QCS8550 Board |
| `python/bevformer_...e2e...py` | asset_manifest, models, reference | performance_result.json, coordinates.npz | QCS8550 Board |
| `python/functional_mother.py` | (被 Runner import) | 工具函数 | QCS8550 Board |
| `python/portable_numpy_nmsfreecoder.py` | cls_scores, bbox_preds | 300 个 3D 检测框 (boxes, scores, labels) | QCS8550 Board CPU |
| `python/verify_contract.py` | reference.npz, candidate.npz | CORRECTED_VERIFICATION_GATE, 报告 | Container B (本地) |
| `python/frame009_numpy_native_reference.npz` | (只读参考) | — | — |

## 当前边界

- ✅ 六路相机张量 → Backbone → Encoder → Decoder → NMSFreeCoder → 3D 坐标
- ✅ 三 Interpreter 单进程常驻，NumPy 内存交接
- ✅ 十帧真实 prev_bev 时序递归
- ✅ 正式稳态性能 (3 warmup + 10 measured)
- ❌ 尚未集成原图解码/resize/normalize (V2 计划)
- ❌ 尚未覆盖完整官方 nuScenes val
