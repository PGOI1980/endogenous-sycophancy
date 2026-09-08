# Supplementary methods

Supplementary methods and results

S1. Observation likelihoods and informed inference

Let pH(d) denote the world likelihood of datum d under H, and let A = 1[d = h*]. For the hallucinating sycophant, L(H, π; d, h*) = πA + (1 - π)pH(d). For the factual sycophant with two independent private data points, the sycophantic likelihood is 1 - (1 - pH(h*))² when d = h*, and pH(d)² otherwise; its mixture with impartial reporting uses the same weights π and 1 - π. Datum indices are exchangeable and are marginalised rather than supplied as an additional observation.

For static and diffusion learners, the joint posterior J(H, πj) is multiplied by L(H, πj; d, h*) and normalised after each observation. The Gaussian transition uses grid spacing Δπ = 1/(B - 1), radius ceil(4σ/Δπ), and weights proportional to exp[-(rΔπ)²/(2σ²)]. Half-sample reflection, implemented by scipy.ndimage.convolve1d with mode="reflect", confines the transition to the grid while preserving ΣjJ(H, πj) separately for each H. No diffusion is applied before the first observation. The original zero-padding implementation is retained only in the archived v2 scripts.

The exact-rule learner labels posterior components by θj, the initial propensity. It begins with θj uniformly distributed over the B-point grid and sets πj(0) = θj. After observing d and h*, each component follows πj(t + 1) = clip(πj(t) + αs(t), 0, 1). The likelihood uses that component’s current πj before the update. Thus no interpolation or diffusion is required for this deterministic rule. The oracle is the single-component case θ = π₀. Both know α and the agreement definition; the oracle additionally knows the true initial propensity. These knowledge assumptions are not inferred from a warning or estimated from user data.

At π₀ = 0.3, grid refinement gives the following coupled threshold-crossing estimates. Matching random inputs across resolutions permits a numerical comparison, but these checks are limited to the displayed starting value. All values are percentages.

The automated checks verify preservation of each H marginal under diffusion, normalisation of both response likelihoods, identity of factual and hallucinating impartial baselines, equivalence of the binary rate cap and a smaller α, the absorbing logistic endpoint, identity of zero diffusion and static inference, and agreement between the oracle and naïve learner for an impartial bot. These are checks of model identities, not tests that require the proposed effect to occur.

S2. Feedback specifications

The main signal is s = 2 × 1[d = h*] - 1. This operationalises agreement, not measured satisfaction and not necessarily the sign of an informed user’s posterior change. The table gives updates before final clipping to [0, 1]. For the confidence gate, c is the larger posterior probability of H immediately after the observation.

The interior logistic formula is evaluated with scipy.special.logit and expit; no positive numerical floor is substituted for zero. The earlier implementation clipped inputs before taking log odds, thereby introducing a tiny positive value at the boundary. The revised exact-zero convention makes the zero-start experiment an identity check. The near-zero cases in sensitivity.json document the separate finite-interior calculation.

Momentum and smoothing can retain a positive update after disagreement. Confidence-gated updating can leave π unchanged, and decay adds a term whose sign depends on displacement from π₀. A negative signal therefore need not lower π under every alternative rule. The main linear rule has a downward response to disagreement except at its lower boundary.

S3. Sampling, pairing, and statistical comparisons

All revised conditions use N = 10,000, T = 100, true H = 1, and a uniform world-state prior. One call to numpy.random.default_rng(42) starts each condition. Every round consumes six uniform arrays in the same order: expressed opinion, first datum, second datum, sycophantic-policy selection, impartial datum selection, and exploration. Drawing the last array even when exploration is absent preserves pairing across interventions. NumPy, SciPy, Python versions, and source hashes are recorded in results/manifest.json. New paired estimates supersede the earlier manuscript’s separate-stream numbers rather than being selected to match them.

Rate intervals use the 95% Wilson score formula. For two paired conditions, a lost crossing is a conversation that crosses only in the comparator; a gained crossing crosses only in the intervention. Conditional on their sum, the exact McNemar test is a two-sided binomial test with success probability one half. A comparison with no discordant pairs has p = 1. The saved summaries include both discordant counts, the rate difference, and raw and Holm-adjusted p-values. Adjustment families comprise 44 main comparisons, 88 diffusion-versus-static comparisons, 44 exact-rule/oracle fixed-versus-coupled comparisons, 18 update-variant comparisons, 11 graded comparisons, 66 matched mitigation comparisons, and 12 feedback-control comparisons. Grid checks and component rankings are descriptive.

At π₀ = 0 and α = 0, naïve belief updating is a biased random walk in log odds with step log(1.5). The 99% false-confidence threshold is first crossed when the signed evidence count reaches -12. Propagating the non-absorbed probability over 100 rounds yields 0.0067661269. This independent calculation lies inside the revised simulation’s Wilson interval and provides a check beyond comparison with the published figure.

S4. Feedback controls and interpretation

The donor sample uses seed 4242 and otherwise the same naïve, hallucinating model. Full per-round donor propensities are recorded before response generation. Replay assigns donor i’s complete schedule to recipient i, whose opinion and observations use seed 42; the recipient cannot alter that schedule. Mean replay assigns the donor mean at each round to all recipients. Signal shuffling uses seed 987 to permute donor agreement signals across conversations independently at each round. It preserves the donor sample’s per-round positive frequency, while breaking within-conversation signal sequences and any link to the recipient. Fair feedback supplies independent ±1 signals with equal probabilities, retaining the same clipping rule.

At zero initial sycophancy, fair signals yield 2.45% threshold crossings, compared with the 0.53% fixed baseline. Clipping and randomly varying exposure can therefore produce excess risk without an agreement-sensitive return path. Donor replay, mean replay, and shuffled donor signals yield still higher rates than live feedback at the displayed starting values. The direction of this contrast prevents attributing the whole fixed-versus-coupled excess to recipient-specific reinforcement; it is compatible with that reinforcement concentrating higher π in conversations whose beliefs are already better supported by the true state.

Figure S1. Feedback controls for the naïve, hallucinating condition. Rates use 10,000 recipient conversations; donor schedules and signals are generated independently. These controls retain selected marginal properties, rather than matching the full joint distribution of opinions, observations, and response propensities.

S5. Reproduction and revision provenance

The accompanying code archive is version 3.0.0-review. Running scripts/check_model.py verifies the stated model identities; scripts/run_all.py regenerates numerical summaries and paired event arrays; scripts/make_figures.py draws all nine manuscript figures and Figure S1 from those summaries. The first 15 traces in Figure 4 are saved directly from the main run. No figure substitutes a newly sampled illustrative conversation for a claimed member of that run. The archive can be supplied through anonymous supplementary materials without exposing identifying repository metadata in the manuscript.

The supplied v2 scripts are retained under legacy/v2, including their comments that scripts 09 and 10 reconstruct code lost between revision rounds. The revised analyses use a common implementation rather than those reconstructions as numerical targets. The changes are probability-conserving diffusion, explicit agreement feedback, exact logistic endpoints, shared random inputs and paired tests, matched mitigation controls, exact-rule inference, and independent-donor feedback controls. The current results are not exact reruns of the historical random streams; they are fully specified reruns of the revised models. The source and results needed to evaluate that distinction accompany the manuscript.

The accompanying manuscript contains the formatted rule and result tables. Numerical table sources are `results/grid_checks.json`, `results/sensitivity.json`, and `results/feedback_controls.json`.
