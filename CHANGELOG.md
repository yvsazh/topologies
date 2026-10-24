# Changelog

All notable changes to `topologies` are listed here.

## 0.0.4 — 2026-10-24

### Added
- **Models**: `phasenet` (PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method) ([paper](https://arxiv.org/abs/1803.03211))
- **Models**: `dkt` (Deep Knowledge Tracing) ([paper](https://arxiv.org/abs/1506.05908))
- **Models**: `aae` (AAE Network and Objective Primitives) ([paper](https://arxiv.org/abs/1511.05644))
- **Models**: `autoencoder-mlp` (Shallow MLP Autoencoder) ([paper](https://www.science.org/doi/10.1126/science.1127647))
- **Components**: `capsnet` (Dynamic Routing Between Capsules) ([paper](https://arxiv.org/abs/1710.09829))
- **Components**: `lottery-ticket` (lottery-ticket) ([paper](https://arxiv.org/abs/1803.03635))
- **Components**: `crnn` (Convolutional Recurrent Neural Network readout) ([paper](https://arxiv.org/abs/1507.05717))
- **Components**: `dncnn` (Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising) ([paper](https://arxiv.org/abs/1608.03981))
- **Components**: `magnitude-pruning` (magnitude-pruning) ([paper](https://arxiv.org/abs/1506.02626))
- **Components**: `llm-int8` (LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale) ([paper](https://arxiv.org/abs/2208.07339))
- **Components**: `wanda` (Wanda pruning score) ([paper](https://arxiv.org/abs/2306.11695))
- **Components**: `squared-relu` (squared-relu) ([paper](https://arxiv.org/abs/2109.08668))
- **Components**: `randomized-smoothing` (Randomized Smoothing (Cohen)) ([paper](https://arxiv.org/abs/1902.02918))
- **Components**: `fgsm` (Fast Gradient Sign Method: adversarial input perturbation) ([paper](https://arxiv.org/abs/1412.6572))
- **Components**: `neural-process` (Neural Processes) ([paper](https://arxiv.org/abs/1807.01622))
- **Components**: `deepsurv` (deepsurv) ([paper](https://arxiv.org/abs/1606.00931))
- **Components**: `rigl` (rigl) ([paper](https://arxiv.org/abs/1911.11134))
- **Components**: `grl` (grl) ([paper](https://arxiv.org/abs/1505.07818))
- **Losses**: `asl`
- **Losses**: `attention-transfer`


## 0.0.3 — 2026-10-23

### Added
- **Models**: `reverse-distillation` (Reverse distillation) ([paper](https://arxiv.org/abs/2201.10703))
- **Models**: `abmil` (Attention-based Deep Multiple Instance Learning) ([paper](https://arxiv.org/abs/1802.04712))
- **Components**: `gem-pooling` (Generalized Mean Pooling) ([paper](https://arxiv.org/abs/1711.02512))
- **Components**: `rl-iqm` (Interquartile Mean for Reinforcement Learning Evaluation) ([paper](https://arxiv.org/abs/2108.13264))
- **Components**: `graph-drop` (DropEdge edge-mask component) ([paper](https://arxiv.org/abs/1907.10903))
- **Components**: `maml` (Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks) ([paper](https://arxiv.org/abs/1703.03400))
- **Components**: `prototypical-networks` (Prototypical Networks for Few-shot Learning) ([paper](https://arxiv.org/abs/1703.05175))
- **Components**: `reptile` (Reptile outer step) ([paper](https://arxiv.org/abs/1803.02999))
- **Components**: `tabr` (TabR) ([paper](https://arxiv.org/abs/2307.14338))
- **Components**: `continual-eval` (Continual Evaluation Metrics) ([paper](https://arxiv.org/abs/1706.08840))
- **Components**: `matching-networks` (Matching Networks for One Shot Learning) ([paper](https://arxiv.org/abs/1606.04080))
- **Components**: `laplace-redux` (Laplace Redux) ([paper](https://arxiv.org/abs/2106.14806))
- **Components**: `colbert` (ColBERT) ([paper](https://arxiv.org/abs/2004.12832))
- **Components**: `cqr` (Conformalized Quantile Regression) ([paper](https://arxiv.org/abs/1905.03222))
- **Components**: `sngp` (Spectral-normalized Neural Gaussian Process) ([paper](https://arxiv.org/abs/2006.10108))
- **Components**: `bbb` (Bayes linear Gaussian-prior primitive) ([paper](https://arxiv.org/abs/1505.05424))
- **Components**: `mc-dropout` (Monte Carlo Dropout) ([paper](https://arxiv.org/abs/1506.02142))
- **Components**: `seco` (Seasonal Contrast) ([paper](https://arxiv.org/abs/2103.16607))
- **Components**: `chexnet` (CheXNet-style multi-label head) ([paper](https://arxiv.org/abs/1711.05225))
- **Components**: `optnet` (Equality projection) ([paper](https://arxiv.org/abs/1703.00443))
- **Components**: `octo` (Octo-style Policy Readout) ([paper](https://arxiv.org/abs/2405.12213))
- **Components**: `minnen-context` (Minnen causal context model) ([paper](https://arxiv.org/abs/1809.02736))
- **Components**: `notears` (NOTEARS Smooth Acyclicity Constraint) ([paper](https://arxiv.org/abs/1803.01422))
- **Components**: `dag-gnn` (DAG-GNN-style Linear SEM) ([paper](https://arxiv.org/abs/1904.10098))
- **Components**: `midas` (MiDaS Scale-Shift Loss) ([paper](https://arxiv.org/abs/1907.01341))
- **Components**: `fedprox` (Proximal federated penalty) ([paper](https://arxiv.org/abs/1812.06127))
- **Components**: `tcav` (TCAV Concept Score) ([paper](https://arxiv.org/abs/1711.11279))
- **Components**: `fedavg` (Count-weighted federated averaging) ([paper](https://arxiv.org/abs/1602.05629))
- **Components**: `grad-cam` (Grad-CAM) ([paper](https://arxiv.org/abs/1610.02391))
- **Components**: `darts` (DARTS Mixed Operation Primitive) ([paper](https://arxiv.org/abs/1806.09055))
- **Losses**: `der-replay`
- **Losses**: `lwf`
- **Losses**: `evidential`
- **Losses**: `temperature-scaling`
- **Losses**: `fixmatch`
- **Losses**: `raft-epe`
- **Losses**: `infoxlm`


## 0.0.2 — 2026-10-22

### Added
- **Models**: `occupancy-networks` (Occupancy Networks: Learning 3D Reconstruction in Function Space) ([paper](https://arxiv.org/abs/1812.03828))
- **Models**: `ds-cnn` (Hello Edge: Keyword Spotting on Microcontrollers) ([paper](https://arxiv.org/abs/1711.07128))
- **Models**: `deeponet` (DeepONet) ([paper](https://arxiv.org/abs/1910.03193))
- **Models**: `perceiver` (Perceiver) ([paper](https://arxiv.org/abs/2103.03206))
- **Models**: `chronos` (Chronos) ([paper](https://arxiv.org/abs/2403.07815))
- **Models**: `barlow-twins` (Barlow Twins) ([paper](https://arxiv.org/abs/2103.03230))
- **Components**: `alibi-bias` (ALiBiBias) ([paper](https://arxiv.org/abs/2108.12409))
- **Components**: `tent` (Tent: Fully Test-time Adaptation by Entropy Minimization) ([paper](https://arxiv.org/abs/2006.10726))
- **Components**: `nerf` (NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis) ([paper](https://arxiv.org/abs/2003.08934))
- **Components**: `classifier-free-guidance` (Classifier-Free Diffusion Guidance) ([paper](https://arxiv.org/abs/2207.12598))
- **Losses**: `byol-loss`


## 0.0.1 — 2026-10-20

First public release: the training engine, data utilities, tokenizers, model constructor,
generation, evaluation, experiments and introspection, plus the experimental catalog of
reference implementations in `topologies.models` and `topologies.layers`.

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
- Document how to enumerate catalog entries from Python.
- Add a licensed catalog-listing example.

### Fixed
- Validate synthetic loader sample counts.
