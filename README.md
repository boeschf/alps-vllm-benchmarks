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

`IMAGE_REF` in each platform profile is the sole current-image pin. Each EDF expands that value
and otherwise contains only runtime settings. Both current pins are content-addressed development
images from the open pull request, so changing an image remains an explicit repository change.

## Repository layout

| Path | Purpose |
|---|---|
| `config/platforms/` | Image, EDF, and loader selection for each accelerator platform |
| `config/scenarios/` | Declarative, composable serving scenarios |
| `edf/` | Container-engine runtime definitions |
| `scripts/submit-serve.py` | Scenario-aware Slurm submission entrypoint |
| `sbatch/serve.sbatch` | Multi-node serving and benchmark workload |
| `sbatch/download.sbatch` | Resumable model download job |
| `benchmarks/standard.tsv` | Standard 11-point benchmark suite |
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

Run submission commands from the repository root in a shell without an active uenv. Slurm rejects
nested `sbatch` calls from an active uenv unless a passthrough policy is supplied; serving uses the
repository EDF instead of inheriting the submitter's uenv.

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

A scenario overlays `config/scenarios/baseline.json`. `scripts/scenario.py` deep-merges composed
components, applies platform overrides, validates its encoded platform and topology constraints,
and emits the exact vLLM YAML plus `scenario.env`. Components compose with `+`.

| Scenario | Change | Constraints or notes |
|---|---|---|
| `baseline` (default) | `allgather_reducescatter` | TP=2; BF16 KV cache; external KV and prefix caching off |
| `uccl-ll` | `deepep_low_latency` | UCCL-EP over CXI; intended for decode-heavy traffic |
| `uccl-ht` | `deepep_high_throughput` | `deep_gemm`, GPU memory 0.80; intended for prefill-heavy experiments |
| `mooncake-cpu` | `MooncakeStoreConnector` | Embedded 8 GB CPU segment per rank and prefix caching |
| `eplb` | Expert load balancing | 32 redundant experts with the Gloo communicator |
| `offload-uva` | UVA weight offload | 16 GiB per GPU |
| `offload-prefetch` | Async layer-group offload | Two of every eight layers offloaded; one layer prefetched |
| `speculative-mtp` | MTP speculative decoding | Three speculative tokens |
| `prefix-caching` | Local GPU prefix cache | No external KV connector |
| `dual-batch-overlap` | Dual batch overlap | Overlaps communication and computation |
| `async-scheduling` | Async scheduling | Overlaps CPU scheduling and accelerator execution |
| `disaggregated-pd` | Mooncake prefill/decode split | Four prefill plus four decode nodes; CUDA only; not composable |
| `production` | Production K8s parity settings | Exactly eight CUDA nodes, TP=4, MTP, EPLB, and MooncakeStore |

The resolver rejects the incompatibilities it explicitly encodes; ordered overlays otherwise use
normal last-component-wins deep-merge semantics. On ROCm, `uccl-ll` uses 0.70 GPU memory
utilization and eager execution. Evidence from the predecessor investigation found the
high-throughput backend unsuitable for decode, but `uccl-ht` remains available as a standalone
prefill-heavy experiment. `disaggregated-pd` uses high throughput for prefill and low latency for
decode.

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

All points use deterministic greedy sampling (`temperature=0`) with random prompt generation seeded
to zero.

The benchmark requires the server `/tokenize` endpoint, verifies raw and chat-template tokenization
before measurement, and fails if measured input-token totals differ from the expected
post-template lengths. `summary.tsv` records both requested `input_len` and
`effective_input_len`.

Each completed run is written under `results/<run-id>/`:

```text
manifest.json               Run state, provenance, topology, commands, and fingerprints
summary.tsv                 Human-readable metrics for every point
bencher.json                Bencher custom-measure payload
raw/<point>.json            Original vLLM result for each benchmark point
```

## Persistent JIT cache

Compiler hot paths stay in container-local `/tmp`. After readiness and smoke success, every node
captures its local artifacts and the job publishes an immutable generation under `.cache/jit`.
A compatible later job hydrates that generation before vLLM starts. Compatibility includes the
image, Python and package versions, CUDA or ROCm version, and accelerator identity; scenarios are
not a compatibility dimension.

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

`manifest.json` schema 2 records the repository state, model identity, image and EDF, resolved
scenario, topology, per-node runtime identities, exact server and benchmark commands, cache
status, and result hashes.

Two fingerprints separate experiment identity from record integrity:

- `experiment_fingerprint` covers relevant tracked and untracked source files, model metadata and
  shard inventory, image, EDF, scenario, topology, stable runtime identity, and suite definition;
- `record_fingerprint` additionally covers the completed output artifacts.

Hostnames, timestamps, and concrete node names are excluded from the fingerprints so equivalent
runs on the same allocation shape retain the same experiment identity.

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
- spec: `glm53-<platform>-<nodes>n-<gpus>g-tp<tp>-dp<dp>-ep<ep>-<scenario>`;
- branch and hash: the measured repository state;
- publishable: true only for a clean worktree.

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
