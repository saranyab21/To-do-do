# Are the Pillar-1 gate checks standard practice? (research brief)

Date: 2026-11-26. Scope: quick literature brief (not a systematic review) to decide which
evaluation checks the thesis keeps, changes or adds for a query-conditioned assessor that
outputs a distribution over expert ratings. Sources were retrieved from arXiv/PubMed via a
paper index and their metadata checked; 2026 items are mostly preprints (not yet peer
reviewed) and are marked as such.

**Question.** Do established groups run these four checks on a model that predicts a
distribution over expert ratings, and how do they report them?

- G1 calibration against the annotators' label distribution
- G2 alignment of model uncertainty with annotator disagreement
- G3 query specificity (does the answer depend on the question asked?)
- G4 grounding (does the answer depend on the image?)

## Short answer

The checks are "common sense", but the literature shows that models routinely fail them,
which is why careful groups measure them explicitly instead of assuming them. G1, G2 and
G4 are established practice with standard metrics. G3 is the least standardised: the
closest work (prompt sensitivity, counterfactual prompting, multi-dimension medical
probing) shows that a question-swap effect only means something when compared with a
paraphrase baseline. That is the main change the thesis should make.

| Check | Standard? | Typical metrics / figures | Thesis action |
|---|---|---|---|
| G1 calibration to annotator distribution | Yes | cross-entropy or KL to soft labels, soft ECE, Brier, reliability diagram | keep; add KL and Brier; stratify by agreement level |
| G2 uncertainty vs disagreement | Yes (growing) | Spearman/Pearson of model entropy vs annotator entropy | keep; report the probe as a reference ρ |
| G3 query specificity | Partly | flip rates / output change under prompt edits, compared with paraphrase edits | keep swap test; **add a paraphrase control** |
| G4 grounding | Yes (very common in VLM work) | blind (no-image) baseline, blank / shuffled / hard-negative image, "visual dependency" gap | keep; **add a hard-negative swap**; use the field's names |

## G1 · Calibration against the annotator distribution

**Evidence.** Peterson, Battleday, Griffiths and Russakovsky (ICCV 2019; Princeton)
released CIFAR-10H, a full distribution of human labels per image, and argued models
should be evaluated against it. Nie, Zhou and Bansal (EMNLP 2020; UNC) collected 100
annotations per item (ChaosNLI) and recommended "evaluating model outputs against the
distribution over collective human opinions"; they found models near-perfect on
high-agreement items and near chance on low-agreement ones. Kerrigan, Smyth and Steyvers
(NeurIPS 2021; UC Irvine) work on CIFAR-10H and treat calibration as the condition for
combining human and model predictions. The LeWiDi shared tasks (Leonardelli, Plank,
Poesio and colleagues, 3rd edition 2025) make soft-label evaluation (cross-entropy
against annotator distributions) the official protocol. Tao et al. (2026, preprint) show
that temperature scaling fitted on majority labels stays miscalibrated against the
annotator distribution, and that miscalibration grows with annotation entropy. Closest to
the thesis: Mittmann et al. (Nature Communications 2025) train Gleason-pattern models on
soft labels from 54 pathologists and report calibration to the pathologists' annotation
distribution.

**How it is presented.** A table of cross-entropy or KL against the soft labels (often
with Brier and ECE), reliability diagrams, and results split by annotator-agreement level
(ChaosNLI's key figure).

**For the thesis.** G1 as built matches practice; fitting the temperature on soft labels
rather than majority labels is exactly what Tao et al. argue for. Add: KL and Brier next
to soft NLL, and a split of every metric by pathologist agreement (all agree vs not).

## G2 · Does uncertainty follow disagreement?

**Evidence.** Singh, Tiwari et al. (2026, preprint) report per-sample correlation between
model entropy and annotator entropy as a primary metric (soft-label training raises it by
61%). Singh and Pakrashi (2026, preprint) measure the same correlation on CIFAR-10H and
FER+ and find it weak (ρ = 0.24–0.55) for standard models. Mendes, Romano and Garlan
(2025, preprint) find model and human uncertainty only weakly aligned. In medical
imaging, Baur, ..., Samek and Ma (2026, preprint; Fraunhofer HHI) show that entropy tracks
boundary noise but not case-level "is it present?" disagreement, and that the information
sits in frozen encoder features but is discarded by the output, which parallels the
thesis's probe result.

**How it is presented.** A correlation coefficient with an interval, and a scatter or
binned plot of model uncertainty against annotator entropy.

**For the thesis.** G2 matches practice. Because even good models reach only moderate ρ,
the pass rule "interval above 0" is sensible; report the linear probe's ρ as a reference
for what the image features allow.

## G3 · Does the answer depend on the question?

**Evidence.** There is no standard named "query-swap" test. Related work:

- Prompt sensitivity: Promptception (Ismithdeen, Khattak and Khan, 2025, preprint;
  MBZUAI) shows accuracy swings up to 15% from prompt wording in multimodal models.
- Counterfactual prompting: Yang, Levy, Goldberg and Wallace (2026, preprint) show that
  a targeted prompt edit (patient gender in MedQA) flips 14.9% of answers, statistically
  no different from simple paraphrasing (14.1%). Their rule: a targeted effect counts only
  if it exceeds the paraphrase baseline.
- Multi-dimension medical probing: ProbMed (Yan et al., 2024; UC Santa Cruz) asks several
  diagnostic questions per image ("procedural diagnosis") and pairs each with a negated
  variant; top models score worse than random on fine-grained questions.

**How it is presented.** Flip rates or output change under each edit type, with the
paraphrase edit as the reference row.

**For the thesis.** G3 is the most original check and the one reviewers will question
most. The swap form (matched minus swapped pairs) cancels question-level bias, which the
first version did not. Following Yang et al., add a paraphrase control: score each
question in two meaning-preserving wordings; question specificity should exceed the
change caused by rewording the same question. This costs two extra passes per core and
question at 448 px.

## G4 · Does the answer depend on the image?

**Evidence.** This is the most established check in multimodal evaluation.

- Blind baselines: MMStar (Chen et al., NeurIPS 2024; Shanghai AI Lab) shows Gemini Pro
  scores 42.9% on MMMU with no image and builds a benchmark of image-dependent items only.
- Paired images: NaturalBench (Li, ..., Neubig and Ramanan, NeurIPS 2024; CMU) pairs each
  question with two images that need different answers and reports models that "choose
  the same answer regardless of the image", which is the zero-shot failure seen here.
- Eyes Wide Shut (Tong, ..., LeCun and Xie, CVPR 2024; NYU) finds visual shortcomings
  using pairs of images CLIP cannot tell apart.
- Medical and pathology: CORAL (Zafar et al., 2026, preprint) substitutes blank,
  pixel-shuffled and retrieved hard-negative images for Qwen2.5-VL-7B on PathVQA and
  others, reports a Visual Reliance Score, and finds that swapping the image costs at most
  6.5 points on medical benchmarks. Hasan et al. (2026, preprint) show a frontier model
  describing histology on blank white images. Hong and Quan (2026, preprint) define a
  Visual Necessity Score from blank images, including for Qwen2.5-VL.

**How it is presented.** Accuracy or answer change with the real image vs no, blank,
shuffled or hard-negative image, often as a "visual dependency gap" table, with
significance tests.

**For the thesis.** G4 matches practice. Two changes:

1. Add a hard-negative swap, a core with a different pathologist distribution that looks
   similar (nearest neighbour in vision-encoder features). A random cross-slide image is
   the easy test.
2. Name the metrics as the field does (visual dependency or reliance) alongside the JS
   divergence.

## What careful groups do that the thesis should copy

1. State each check, its manipulation and its pass rule before running it (done: the
   gate config and the 0.01-nat margin). Report amendments openly (done for G3).
2. Compare every targeted manipulation with a nuisance manipulation of the same size
   (paraphrase for G3, hard negative vs random image for G4).
3. Split results by human agreement level, because averages hide that models fail
   exactly where humans disagree.
4. Show a human reference next to every model number (done: pathologists' floor,
   pathologists-as-predictor, partner disagreement).
5. Report the same table for several models. A second small model (Qwen2.5-VL-3B or
   MedGemma-4B, zero-shot) would show that the failure is not specific to one model.

## Limitations

This is a quick brief, not a systematic review: retrieval used one paper index with a few
queries per check, and the newest items are unreviewed preprints, so their numbers may
change. Venue information for 2024 papers is from memory where the index did not state
it. No paper was found that runs all four checks together on a multi-rater medical
dataset, but absence from a quick search is not proof that none exists.

## References

- Baur, S., Schernich, A., Böke, E., Samek, W., & Ma, J. (2026). Beyond boundary noise: Aggregated aleatoric uncertainty fails to capture presence ambiguity in 3D lung nodule segmentation. arXiv:2608.14766 (preprint).
- Chen, L., Li, J., Dong, X., et al. (2024). Are we on the right way for evaluating large vision-language models? NeurIPS 2024. arXiv:2403.20330.
- Hasan, M. M., Tozal, M. E., & Ayhan, M. S. (2026). Seeing nothing, saying something: The lack of visual grounding and confabulation in Gemini models for histopathology. medRxiv, doi:10.64898/2026.07.04.26357257 (preprint).
- Hong, R., & Quan, S. (2026). To see or to please: Uncovering visual sycophancy and split beliefs in VLMs. arXiv:2603.18373 (preprint).
- Ismithdeen, M. I., Khattak, M. U., & Khan, S. (2025). Promptception: How sensitive are large multimodal models to prompts? arXiv:2509.03986 (preprint).
- Kerrigan, G., Smyth, P., & Steyvers, M. (2021). Combining human predictions with model probabilities via confusion matrices and calibration. NeurIPS 2021. arXiv:2109.14591.
- Leonardelli, E., Casola, S., Peng, S., et al. (2025). LeWiDi-2025 at NLPerspectives: Third edition of the learning with disagreements shared task. arXiv:2510.08460.
- Li, B., Lin, Z., Peng, W., et al. (2024). NaturalBench: Evaluating vision-language models on natural adversarial samples. NeurIPS 2024 Datasets and Benchmarks. arXiv:2410.14669.
- Mendes, P., Romano, P., & Garlan, D. (2025). Uncertainty estimation by human perception versus neural models. arXiv:2506.15850 (preprint).
- Mittmann, G., Laiouar-Pedari, S., Mehrtens, H. A., et al. (2025). Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer. Nature Communications. doi:10.1038/s41467-025-64712-4.
- Nie, Y., Zhou, X., & Bansal, M. (2020). What can we learn from collective human opinions on natural language inference data? EMNLP 2020. arXiv:2010.03532.
- Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty makes classification more robust. ICCV 2019. arXiv:1908.07086.
- Singh, A., Tiwari, A., Hasanbeig, H., & Gupta, P. (2026). Distributions in, distributions out: The case for soft-label training. arXiv:2511.14117 (preprint).
- Singh, M., & Pakrashi, A. (2026). Does model uncertainty track human ambiguity? Evidence from multi-annotator vision benchmarks. arXiv:2609.34506 (preprint).
- Tao, L., Luo, H., Dong, M., & Xu, C. (2026). Confidence calibration under ambiguous ground truth. arXiv:2603.22879 (preprint).
- Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., & Xie, S. (2024). Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. CVPR 2024. arXiv:2401.06209.
- Yan, Q., He, X., Yue, X., & Wang, X. E. (2024). Worse than random? An embarrassingly simple probing evaluation of large multimodal models in medical VQA. arXiv:2405.20421.
- Yang, Z., Levy, M., Goldberg, Y., & Wallace, B. C. (2026). Compared to what? Baselines and metrics for counterfactual prompting. arXiv:2605.01048 (preprint).
- Zafar, A., Murali, L. K., Bharadwaj, S., Vashist, A., & Wu, J. (2026). Do medical vision language models actually see? A counterfactual grounding framework and hard-negative contrastive training for visually-reliant medical VLMs. arXiv:2607.03647 (preprint).

AI disclosure: this brief was compiled with an AI assistant (Claude) using a paper-search
index; every listed source was retrieved and its metadata checked, but claims should be
verified against the full texts before they are cited in the thesis.
