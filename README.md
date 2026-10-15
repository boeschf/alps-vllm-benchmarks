# GLM-5.3 inference on Alps

Reproducible vLLM serving and benchmarking for `zai-org/GLM-5.3` on Alps:

- NVIDIA GH200 over Slingshot/CXI;
- AMD MI300A over Slingshot/CXI.

The repository consumes the vLLM 0.30 images produced by
[`eth-cscs/alps-extended-images#63`](https://github.com/eth-cscs/alps-extended-images/pull/63).
It does not rebuild or patch those images. Optional UCCL-EP, Mooncake, and offload components are
present in the images but are enabled only by explicit scenarios.

## Images

| Platform | Profile | Model loader |
|---|---|---|
| GH200 | [`config/platforms/cuda.env`](config/platforms/cuda.env) | Run:ai Model Streamer |
| MI300A | [`config/platforms/rocm.env`](config/platforms/rocm.env) | Standard vLLM loader |

`IMAGE_REF` in each platform profile selects the current image tag. At job startup the launcher
resolves that tag through the registry and records its immutable `IMAGE_DIGEST`; failure to resolve
the digest fails the job before serving. Each EDF expands `IMAGE_REF` and otherwise contains only
runtime settings. Changing an image tag remains an explicit repository change, while the resolved
digest detects registry-side tag movement.

## Repository layout

| Path | Purpose |
|---|---|
| `config/platforms/` | Image, EDF, loader, NUMA, and runtime defaults for each accelerator platform |
| `config/scenarios/` | Declarative, composable serving scenarios |
| `edf/` | Container-engine runtime definitions |
| `scripts/submit-serve.py` | Scenario-aware Slurm submission entrypoint |
| `sbatch/serve.sbatch` | Multi-node serving and benchmark workload |
| `sbatch/download.sbatch` | Resumable model download job |
| `benchmarks/` | Standard short-context and production-parity long-context suites |
| `results/` | Versionable benchmark records |
| `logs/` | Ignored Slurm, server, smoke, and cache logs |
| `tools/install-bencher.sh` | Pinned repository-local Bencher CLI installer |

## Prerequisites and model

Install `uv` once. Host-side Python uses a managed Python 3.12 because the compute-node system
Python is too old for the scripts; no virtual environment is required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the Hugging Face CLI once:

```bash
uv tool install 'huggingface_hub[cli]'
```

Run submission commands from the repository root. The wrapper always passes
`--uenv-passthrough=ignore`: an active submitter uenv is never inherited, because serving uses the
repository EDF.

The native FP8 checkpoint is pinned to revision
`aca966e4e02791568aa6a4ced368624b3d897f42` and is approximately 704 GiB. Download it to the
default ignored directory, `models/GLM-5.3`:

```bash
sbatch sbatch/download.sbatch
```

Download progress is written to `logs/download-<jobid>.out` and failures to
`logs/download-<jobid>.err`. The download is resumable. Set `MODEL_REVISION` deliberately when
testing another revision.

Serving defaults to `models/GLM-5.3`. A different checkpoint must be visible from every allocated
node:

```bash
./scripts/submit-serve.py --model /path/to/checkpoint
```

The EDF mounts the resolved host directory read-only at `/models/GLM-5.3`. `MODEL_CONTAINER` may
override that absolute container path when an image requires a different location.

## Scenarios

`scripts/scenario.py` resolves layers in this order: `config/scenarios/baseline.json`, the selected
`config/platforms/<platform>.json`, then scenario components from left to right. Platform defaults
own the common CUDA and ROCm NUMA maps, CUDA's 0.88 memory budget, and
`VLLM_ROCM_USE_AITER=1`. A scenario component can still override those defaults; its selected
`platform_overrides` layer is applied immediately after that component's general settings. The
resolver then validates platform and topology constraints and emits the exact vLLM YAML plus
`scenario.env`.

| Scenario | Change | Constraints or notes |
|---|---|---|
| `baseline` (default) | `allgather_reducescatter` | TP=2; CUDA GPU memory 0.88; BF16 KV cache; external KV and prefix caching off |
| `uccl-ll` | `deepep_low_latency` | UCCL-EP over CXI; CUDA GPU memory 0.90; decode-oriented 1024-token scheduler cap |
| `uccl-ht` | `deepep_high_throughput` | UCCL-EP over CXI; CUDA pins `deep_gemm`, while ROCm retains its platform backend selection |
| `moe-deep-gemm` | `deep_gemm` expert kernels | CUDA-only; composable with baseline or low-latency all-to-all |
| `mooncake-cpu` | `MooncakeStoreConnector` | Embedded 8 GB CPU segment per rank and prefix caching |
| `eplb` | Expert load balancing | 32 redundant experts with the Gloo communicator |
| `offload-uva` | UVA weight offload | 16 GiB per GPU |
| `offload-prefetch` | Async layer-group offload | Two of every eight layers offloaded; one layer prefetched |
| `speculative-mtp` | MTP speculative decoding | Three speculative tokens |
| `prefix-caching` | Local GPU prefix cache | No external KV connector |
| `dual-batch-overlap` | Dual batch overlap | Overlaps communication and computation |
| `async-scheduling` | Async scheduling | Overlaps CPU scheduling and accelerator execution |
| `disaggregated-pd` | Mooncake prefill/decode split | Four prefill plus four decode nodes; CUDA only; not composable |
| `production` | External production vLLM parity | Exactly four CUDA nodes, TP=4/DCP=4, 1M context, FP8 KV cache, 32 sequences, 4096-token scheduler cap, and one MTP token |
| `million-context` | Experimental vLLM 0.30 capacity reproducer | Exactly four CUDA nodes, TP=4, FP8 KV cache, native CPU KV offload; currently fails startup without DCP |

`production` tracks the active vLLM parameters in `glm-5-3-fp8_tp_dp.yaml`: FlashMLA sparse
attention, DCP=4 with `ag_rs`, custom all-reduce enabled, MTP=1, and EPLB and Mooncake disabled.
The benchmark image, Slurm runtime, network environment, and resource settings remain owned by
this repository's platform configuration rather than the Kubernetes deployment.

The resolver rejects the incompatibilities it explicitly encodes; ordered overlays otherwise use
normal last-component-wins deep-merge semantics. On ROCm, `uccl-ll` uses 0.70 GPU memory
utilization and eager execution. On CUDA, `uccl-ht` pins DeepGEMM because automatic selection
chose FlashInfer CUTLASS and stalled distributed engine initialization; the working path used
DeepGEMM with `DeepEPHTPrepareAndFinalize`. ROCm retains its platform-native MoE selection. Use
`moe-deep-gemm` or `uccl-ll+moe-deep-gemm` to pin DeepGEMM in other CUDA runs.

`uccl-ll` retains `max-num-batched-tokens=1024`. Job 5037516 tested 2048 with CUDA GPU
memory utilization 0.90 and still exhausted GPU memory during startup. DeepEP sizes its
low-latency RDMA workspace from this capacity, so raising the scheduler cap consumes memory before
the server can allocate its required KV cache. The one-million-token profile also uses the proven
1024-token cap.

`uccl-ll+million-context` records the attempted native 1,048,576-token target on four GH200 nodes;
it is a failure reproducer, not a supported serving profile. Job 5037564 measured only
15.86 GiB/GPU available for KV cache without offload, versus 52.68 GiB required, and estimated a
315,776-token ceiling. Job 5037589 tried 42 GiB/GPU of UVA weight offload, reached
405,443,520 KiB (about 387 GiB) step RSS while loading, and was OOM-killed. Job 5037650 accepted
a 40 GiB native CPU KV-offload configuration but applied the same local-KV startup check:
15.85--15.87 GiB available and an
estimated 315,392--315,904-token ceiling. Native KV offload therefore did not extend the maximum
single active request on this vLLM/model path.

The existing `mooncake-cpu` component is complementary, not a demonstrated solution to this
limit. Its `MooncakeStoreConnector` externalizes reusable prefix blocks and currently has
`enable_offload=false`; it does not satisfy the observed startup requirement that one active
`max-model-len` request fit local KV capacity. Do not combine it with the reproducer until active
single-request paging is supported and verified.

Evidence from the predecessor investigation found the high-throughput backend unsuitable for
decode, but `uccl-ht` remains available as a prefill-heavy experiment on either platform.
`disaggregated-pd` uses high throughput for prefill and low latency with the same 1024-token cap
and 0.90 CUDA memory utilization for decode.

## Submit serving

Use the wrapper rather than invoking `sbatch/serve.sbatch` directly. It resolves the scenario
before submission and requests the correct allocation.

```bash
./scripts/submit-serve.py
```

Node selection follows the scenario constraints:

- exact-size scenarios request their required node count;
- scalable scenarios default to `min_nodes` and accept `--nodes`;
- `disaggregated-pd` derives its allocation from its prefill and decode role counts.

Examples:

```bash
./scripts/submit-serve.py --scenario uccl-ll
./scripts/submit-serve.py --scenario uccl-ll+eplb --nodes 8
./scripts/submit-serve.py --scenario uccl-ll+million-context --smoke-only --time 04:00:00
./scripts/submit-serve.py --scenario production
./scripts/submit-serve.py --scenario disaggregated-pd
./scripts/submit-serve.py --platform rocm
```

Common Slurm overrides are explicit wrapper options:

```bash
./scripts/submit-serve.py --platform rocm --nodes 8 \
  --account csstaff --partition <partition> --time 02:00:00
```

The default CUDA allocation is four nodes with four GPUs each:

```text
TP = 2
DP = 8
EP = DP x TP = 16
```

Distributed serving steps use `--distribution=block:cyclic` and leave Slurm CPU binding disabled;
vLLM binds accelerator workers to their local NUMA resources.
The default single service uses all allocated nodes; its OpenAI-compatible endpoint is served on
port 8000 of the first allocated node. After readiness, the job performs one real chat completion.
Without a benchmark suite or `--smoke-only`, the server remains active until the job is cancelled.

Logs are split by purpose:

```text
logs/serve-<jobid>.out       Slurm standard output
logs/serve-<jobid>.err       Slurm standard error
logs/run-<jobid>/            Scenario, per-node runtime, server, smoke, and cache records
```

Scenario materialization writes one complete vLLM YAML per role-local node under
`logs/run-<jobid>/scenario/`, for example `vllm-single-node0.yaml` or
`vllm-decode-node3.yaml`. Each file includes the model, topology, coordinator, rank, and API
settings used by that process. Workers launch only `vllm serve --config <node-yaml>`; no vLLM
settings are appended by the Slurm launcher.

## Run the standard benchmark

Pass the suite to the submission wrapper. The job exits after readiness, smoke completion, cache
publication, and all benchmark points complete.

```bash
./scripts/submit-serve.py --benchmark-suite benchmarks/standard.tsv
./scripts/submit-serve.py --platform rocm \
  --benchmark-suite benchmarks/standard.tsv
```

The standard suite contains 11 points:

- concurrency scaling at 1024 input and 128 output tokens;
- prefill scaling at concurrency 8 with 1K, 4K, and 16K input tokens;
- decode scaling at concurrency 8 with 128, 512, and 1024 output tokens.

The long-context suite preserves the standard suite for historical comparison and separately tests
32K, 128K, and 256K input contexts with 128 output tokens at concurrency 8. Run it against the
four-node TP=4/DCP=4 production-parity scenario, whose 1,048,576-token context limit,
32-sequence scheduler, and 4096-token batch cap match the external deployment's active vLLM
parameters:

```bash
./scripts/submit-serve.py --scenario production \
  --benchmark-suite benchmarks/long-context.tsv --time 08:00:00
```

Long-context points retain 128 requests and three repetitions so their p99 values have the same
meaning as the standard suite; the explicit eight-hour allocation accounts for that measurement
volume.

Every point uses deterministic greedy sampling (`temperature=0`) with random prompt generation seeded
to zero. Each measured repetition contains at least 128 requests (256 at concurrency 64), so every
reported p99 is derived from at least 128 request observations.

The benchmark requires the server `/tokenize` endpoint, verifies raw and chat-template tokenization
before measurement, and fails if measured input-token totals differ from the expected
post-template lengths. `summary.tsv` records both requested `input_len` and
`effective_input_len`.

Before measurement, the harness runs every suite point once with the same input/output lengths,
concurrency, seed, and sampling settings but an explicit smaller `warmup_prompts` count. The
standard suite uses `min(prompts, max(16, 2 * concurrency))`: at least two concurrency waves and
at least 16 requests, without paying for statistically meaningful sample counts in an unmeasured
phase. This full shape pass retains every suite shape and at least two scheduler waves; its
diagnostic results are retained under `warmup/`.

Measurement then runs three complete suite passes. `summary.tsv` reports the arithmetic mean and
sample standard deviation of each run-level metric. `bencher.json` publishes that mean with
`lower_value` and `upper_value` set to one sample standard deviation below and above it. The batch
script requests four hours to accommodate the warmup and three separate timed passes.

Slurm output uses UTC `[status ...]` lines for scenario materialization, worker launch, readiness,
smoke, cache publication, and benchmark phases. Readiness reports once per minute. Every warmup,
measurement, or cache-publication subprocess reports a heartbeat every 60 seconds and a completion
line with elapsed time.

Each completed run is written under `results/<run-id>/`:

```text
manifest.json                    Run state, provenance, topology, commands, and fingerprints
summary.tsv                      Human-readable metrics for every point
bencher.json                     Bencher custom-measure payload
submitted-batch-script.sbatch    Exact Slurm spool script executed by the job
raw/<point>/repetition-<n>.json  Original vLLM result for each measured repetition
warmup/<point>.json              Unmeasured full-suite warmup result for each point
```

## Persistent JIT cache

Compiler hot paths stay in container-local `/tmp`. Benchmark jobs run the full-suite warmup, then
capture every node's local artifacts and publish an immutable generation under `.cache/jit` before
measurement starts. Jobs without a benchmark suite publish after smoke instead. A compatible later
job hydrates that generation before vLLM starts. Compatibility includes the image, Python and
package versions, CUDA or ROCm version, and accelerator identity; scenarios are not a compatibility
dimension.

Cache capture is a strict barrier for benchmark jobs: capture errors stop the run before the first
measured request. A publication conflict caused by another job advancing the same namespace keeps
that complete generation and records the conflict, but does not discard the already-warmed
allocation or block measurement.

The wrapper exposes the operational cache modes directly:

- `readwrite` (default): hydrate and publish;
- `read`: hydrate without publishing;
- `off`: use clean node-local caches without reading or publishing.

```bash
./scripts/submit-serve.py --jit-cache-mode off
./scripts/submit-serve.py --jit-cache-mode read
./scripts/submit-serve.py --smoke-only
```

`--smoke-only` exits after smoke and cache publication when no benchmark suite is selected.
FlashInfer's rank-local autotune JSON is excluded because partial hits can desynchronize collective
profiling. Snapshots include FlashInfer, Triton, TorchInductor, Torch extensions, vLLM, and
TensorRT-LLM DeepGEMM artifacts.

Controls:

- `JIT_CACHE_ROOT`: shared root, default `.cache/jit`;
- `JIT_CACHE_LOCK_SECONDS`: publication lock deadline, default 300;
- `JIT_CACHE_BUNDLE_SECONDS`: node-bundle deadline, default 1800;
- `JIT_CACHE_KEEP_GENERATIONS`: retained generation count, default 3;
- `JIT_CACHE_SEED_IMAGE_REFS`: explicit comma-separated source images for a migration.

The current CUDA profile carries one explicit seed from its preceding image. Seed reuse still
requires every non-image compatibility field to match, and reused artifacts are republished under
the current image namespace. Missing, invalid, or conflicting snapshots fall back to clean local
caches without changing inference correctness.

## Benchmark provenance and Bencher

`manifest.json` schema 2 records the repository state, the exact Slurm spool script, model identity,
image tag and resolved OCI manifest digest, EDF, resolved scenario, topology, per-node runtime
identities, complete executed server and benchmark commands, cache status, and result hashes.

Benchmark initialization also creates `reproduction/` in the result directory. Its `vllm/`
subdirectory contains byte-for-byte copies of every role/node YAML, `environment.json` contains
the allowlisted environment captured for each process, and `bundle.json` records their checksums.
The allowlist includes every EDF runtime setting, the image tag and digest, and scenario-declared
variables; unrelated process environment is excluded. These artifacts are created before warmup
and remain available when a later benchmark phase fails.

Two fingerprints separate experiment identity from record integrity:

- `experiment_fingerprint` covers relevant tracked and untracked source files, the executed Slurm
  batch-script checksum, model metadata and shard inventory, image, EDF, scenario, topology, stable
  runtime identity, the reproduction bundle's normalized identity, and suite definition;
- `record_fingerprint` additionally covers the completed output artifacts, including the archived
  batch script and exact reproduction bundle.

Hostnames, timestamps, concrete node names, coordinator addresses, and run-local cache paths are
excluded from experiment identity so equivalent runs on the same allocation shape retain the same
fingerprint. Their exact measured values remain checksummed in the reproduction bundle and are
therefore covered by record integrity.

Install the pinned Bencher CLI into `tools/bin/<arch>/`:

```bash
tools/install-bencher.sh
```

The installer supports aarch64 and x86_64, verifies the downloaded binary against the checksum in
the script, and does not modify `$HOME` or `PATH`. To upgrade, change `BENCHER_VERSION` and the
corresponding checksums in `tools/install-bencher.sh`.

Upload a completed result; only `BENCHER_API_KEY` is required at publication time:

```bash
BENCHER_API_KEY=bencher_run_... scripts/publish-bencher.sh results/<run-id>/
```

Publication metadata comes from the manifest:

- project: `glm-5-3-inference` by default;
- testbed: `alps-gh200` or `alps-mi300a`;
- benchmark variant: `glm53-<platform>-<nodes>n-<gpus>g-tp<tp>-dp<dp>-ep<ep>-<scenario>`;
- branch and hash: the measured repository state;
- publishable: true only for a clean worktree.

The publisher prefixes every benchmark name with the variant. It does not pass the variant as
Bencher's `--spec`: that option selects remote Runner hardware, requires `--image`, and would
incorrectly rerun rather than upload the measurements produced on Alps.

Known generated paths and root-level Linux core dumps are excluded from source identity. Other
untracked files remain fingerprinted and make the run non-publishable because they may affect the
measured code or configuration.

`BENCHER_PROJECT`, `BENCHER_TESTBED`, `BENCHER_BRANCH`, `BENCHER_SPEC`, and `BENCHER_HASH` may
deliberately override manifest values. The publisher rejects incomplete runs and dirty-tree runs
unless `BENCHER_HASH` is supplied explicitly.

Custom measure names encode their units:

- `request-throughput-req-per-sec`;
- `output-throughput-tok-per-sec`;
- `total-throughput-tok-per-sec`;
- `mean-ttft-ms`, `p99-ttft-ms`;
- `mean-tpot-ms`, `p99-tpot-ms`.

## Measurement boundaries

This repository provides reproducible serving and measurement plumbing, not performance claims.
The four-node baseline floor is currently applied to both platforms. It comes from predecessor
GH200 work; MI300A memory fit and optimal parallelism remain unmeasured. Results belong in
`results/`, and launcher defaults should change only after measurement.
