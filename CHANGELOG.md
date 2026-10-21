# Changelog

All notable changes to `topologies` are listed here.

## 0.0.1 — 2026-10-20

### Added
- **Models**: `xmem` (XMem) ([paper](https://arxiv.org/abs/2207.07115))
- **Models**: `consistency-model` (Consistency Models) ([paper](https://arxiv.org/abs/2303.01469))
- **Models**: `distributional-q` (A Distributional Perspective on Reinforcement Learning) ([paper](https://arxiv.org/abs/1707.06887))
- **Models**: `biggan` (Large Scale GAN Training for High Fidelity Natural Image Synthesis) ([paper](https://arxiv.org/abs/1809.11096))
- **Models**: `dqn-family` (Playing Atari with Deep Reinforcement Learning) ([paper](https://arxiv.org/abs/1312.5602))
- **Models**: `latent-diffusion` (High-Resolution Image Synthesis with Latent Diffusion Models) ([paper](https://arxiv.org/abs/2112.10752))
- **Models**: `rgcn` (R-GCN) ([paper](https://arxiv.org/abs/1703.06103))
- **Models**: `graphsage` (GraphSAGE Mean Aggregator) ([paper](https://arxiv.org/abs/1706.02216))
- **Models**: `qmix` (QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning) ([paper](https://arxiv.org/abs/1803.11485))
- **Models**: `lightgcn` (LightGCN) ([paper](https://arxiv.org/abs/2002.02126))
- **Models**: `tft` (TFT) ([paper](https://arxiv.org/abs/1912.09363))
- **Models**: `dcrnn` (DCRNN diffusion-GRU sequence primitive) ([paper](https://arxiv.org/abs/1707.01926))
- **Components**: `cbam-gate` (CBAM channel-and-spatial attention block) ([paper](https://arxiv.org/abs/1807.06521))
- **Components**: `topk-moe-layer` (Deterministic top-k MoE core; excludes noisy gating and balance losses) ([paper](https://arxiv.org/abs/1701.06538))
- **Components**: `meshcnn` (MeshCNN: edge convolution and collapse pooling) ([paper](https://arxiv.org/abs/1809.05910))
- **Components**: `crestereo` (CREStereo local correlation/disparity-offset primitive) ([paper](https://arxiv.org/abs/2203.11483))
- **Components**: `paged-attention` (Paged Attention) ([paper](https://arxiv.org/abs/2309.06180))
- **Components**: `conformer` (Conformer Convolution Module) ([paper](https://arxiv.org/abs/2005.08100))
- **Components**: `v-trace` (IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures) ([paper](https://arxiv.org/abs/1802.01561))
- **Components**: `cluster-gcn` (Cluster-GCN) ([paper](https://arxiv.org/abs/1905.07953))
- **Components**: `film` (FiLM) ([paper](https://arxiv.org/abs/1709.07871))
- **Losses**: `a2c` and `poly1-loss`.
- **Training/docs**: Add a guide and an end-to-end `poly1-loss` quickstart.

### Changed
- Document how to enumerate catalog entries from Python (TPC-ZYNND4GA).
- Add a licensed catalog-listing example (TPC-TPATES06).

### Fixed
- Validate synthetic loader sample counts (TPC-XK5Q2N6V).
