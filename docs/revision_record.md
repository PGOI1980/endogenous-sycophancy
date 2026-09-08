# Revision record: 3.0.0-review

The manuscript review identified discrepancies between the stated models, reported tests, and supplied code. This revision addresses those discrepancies and regenerates the results from a common implementation.

## Model changes

The main signal is now defined as agreement between the response and the expressed opinion, matching the original numerical mechanism. Actual posterior movement remains a separate graded-feedback experiment. The informed-user likelihood targets the naïve listener's confirming datum, while the informed listener reasons about that policy.

The diffusion learner uses a reflecting Gaussian transition. Unlike zero padding followed by joint renormalisation, it preserves the probability assigned to each world state before a new observation. Both fixed-unknown-rate and known-feedback-rule inference are included. The latter keeps uncertainty about the initial propensity; an oracle additionally knows that initial value.

The logistic interior update is unchanged in form, but exact zero and one are absorbing endpoints. The earlier floor that silently introduced a small positive propensity at zero is removed. Near-zero experiments are reported separately.

## Experimental changes

Every active condition consumes the same number of random inputs per round. The revised seed-42 runs consequently supersede the earlier separate-stream numbers. Exact McNemar tests use paired crossing outcomes, with declared Holm families. Wilson intervals describe each marginal rate and are not used as a substitute for a paired difference test.

The intervention suite includes matching fixed-policy controls, smoothing alone, and cap-plus-exploration. The symmetric cap is explicitly identified as a smaller learning rate for binary linear feedback. Independent donor schedules, donor means, shuffled donor signals, and fair signals test which effects require dependence on the current recipient.

## Changes to the manuscript's conclusions

The new text confines the main increase to the conditions supported by the saved comparisons. It distinguishes absolute from proportional protection, and sycophantic response propensity from false-confidence crossings. It removes claims of an intrinsic failure of informed reasoning, universal superiority of the combined intervention, and a mathematical upper bound supplied by binary feedback.

Donor replay produces more crossings than live feedback in the tested naïve, hallucinating conditions. The revised paper therefore does not attribute the entire fixed-versus-coupled excess to recipient-specific dependence. Exact-rule and oracle comparisons likewise prevent treating the static learner's failure as a limit of all informed inference.

The mitigation experiment is moved into Results, the figure numbering follows presentation order, and the first 15 traces in Figure 4 come directly from the saved main run. Supplementary methods supply likelihoods, transition rules, secondary parameters, sampling conventions, grid checks, control results, and provenance.

## Earlier material

`legacy/v2` retains the source and saved outputs provided for review. It includes the reconstruction notes for the earlier mitigation and graded-signal scripts. Those historical scripts remain separate from the active pipeline, and their outputs are not used to tune the revised estimates.

## Validation

The model checks cover conservation of probability, preservation of world-state marginals under diffusion, observation-likelihood normalisation, equivalent impartial baselines, rate-cap equivalence, the absorbing zero-logistic endpoint, zero-diffusion identity, and the impartial oracle. An independent absorbing-random-walk calculation gives the exact impartial threshold-crossing probability, 0.0067661269, for the stated horizon and threshold.

Grid-resolution checks compare 21, 51, and 101 points at initial propensity 0.3 for static, diffusion, and exact-rule inference with each bot type. They do not establish numerical convergence outside those settings. Selected model identities and principal results are checked again after the saved-result pipeline completes.
