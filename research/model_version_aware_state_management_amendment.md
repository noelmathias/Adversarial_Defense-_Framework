# Amendment: Model-Version-Aware Adaptive DA-LinTS State Management

## Status and purpose

This amendment supplements, but does not replace or modify, the frozen preregistration in `research/model_version_aware_state_management_preregistration.md` and `.json`. Its sole purpose is to document a failed execution-integrity gate and required runtime-provenance recording before a future final execution.

## 1. Failed initial execution

An initial full-run attempt stopped at the retained Phase-A trajectory-integrity gate for seed `20260822`. The first meaningful recorded divergence was episode 49 (one-based): `adversarial_confidence` was `0.8756623268127441` for `full_reset` and `0.8733007311820984` for `version_indexed_learner_state_isolation`. Later differences propagated into residual, posterior diagnostics, sampled estimates, and defended confidence.

The stop was correct. The full experiment was not continued, no completed policy/seed was silently rerun, and existing artifacts are to be preserved.

## 2. Controlled follow-up diagnostic

A research-only diagnostic reproduced the sequential Phase-A prefix for seed `20260822` through episode 49, whose manifest entry was image index `4705`, PGD, epsilon `0.03`. It compared independently loaded policy executions at the earliest relevant points.

In the current CPU environment, the two executions were identical for:

- standard checkpoint SHA-256;
- model parameter/state digest before and after attack;
- model evaluation state (`training=False`) and device;
- clean input tensor and target;
- Torch CPU RNG state before the clean forward, before attack, and after attack;
- clean logits and confidence;
- adversarial tensor, label, probabilities, and confidence; and
- model state after the attack.

CUDA was unavailable in the diagnostic environment. The version registry is not created or activated until after Phase-A episode 100, and the diagnostic therefore ruled out the currently testable Phase-A causes involving the intervention registry, current input generation, checkpoint/model identity, model mutation, current Torch CPU RNG state, and attack generation.

## 3. Interpretation

The original seed-`20260822` discrepancy could not be reproduced from the preserved information available to the diagnostic. It is therefore classified as an execution-environment/runtime artifact whose exact cause is not identifiable from the preserved artifacts. This classification is not a claim that the original mismatch was harmless: the trajectory equality gate remains mandatory.

## 4. No design changes

This amendment changes none of the following:

- research question, hypotheses, intervention, or baseline;
- seeds, episode counts, phases, manifests, or model-version transition;
- classifier checkpoints or checkpoint hashes;
- DA-LinTS mathematics, hyperparameters, action space, or learner update path;
- reward, risk computation, attacks, attack severity, defenses, thresholds, or evaluation methodology;
- CUSUM implementation, configuration, residual, startup/cooldown behavior, or posterior-discount path; or
- primary/secondary metrics, advancement criteria, failure criteria, or interpretation limits.

The original preregistration remains frozen. This amendment does not authorize parameter tuning, selective reruns, artifact deletion, or relaxation of any integrity requirement.

## 5. Required runtime-provenance record before final execution

Before beginning a future final experiment, create and retain an immutable runtime-provenance record in its new non-overwriting results directory. It must include:

- Python version and executable;
- PyTorch version and torchvision version, if installed;
- NumPy version;
- platform, OS, architecture, processor/CPU information, and hostname where permitted by the execution environment;
- active device, CUDA availability, CUDA runtime/device name where available, and CUDA RNG state availability;
- `torch.are_deterministic_algorithms_enabled()`;
- `torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark`;
- checkpoint SHA-256 hashes;
- git revision/dirty-state information when available;
- SHA-256 hashes of the experiment module and relevant imported research/episode-runner modules; and
- dependency metadata (for example `pip freeze` or an equivalent installed-package listing).

Recording this metadata is a reproducibility/provenance requirement only. It does not enable deterministic algorithms, change backend settings, alter any experiment condition, or amend the frozen analysis.

## 6. Retained integrity gate and failure handling

The Phase-A trajectory equality gate remains exact, not approximate. For each seed, `full_reset` and `version_indexed_learner_state_isolation` must have identical Phase-A records under the matched manifest and RNG protocol before either policy may proceed to Phase B.

If the gate fails again, execution must stop immediately, preserve all partial artifacts and runtime-provenance records, and report the failure. Do not silently rerun the seed, alter parameters, disable or weaken the gate, replace the manifest, change seeds, or continue the full experiment.

## 7. Scope

This is a research-only documentation amendment. It authorizes no production-code, Streamlit, adaptive learner, detector, defense, reward, risk, attack, checkpoint, or original-preregistration modification. It does not authorize a full rerun.
