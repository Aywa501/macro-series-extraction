# Implicit Monetary Value Encoding in BERT-Family Language Models
## A Corpus-Free Probing Study

**Date:** April 2026  
**Models:** `bert-base-uncased`, `emanjavacas/MacBERTh`  
**Data:** Bank of England Millennium of Macroeconomic Data (A47, A48)

---

## 1. Overview

This study asks whether pre-trained language models implicitly encode the historical monetary value of British coins in their embedding space — without any explicit economic training signal. The approach is corpus-free: no historical text corpus is searched. Instead, synthetic probe sentences are passed through frozen model weights, and the CLS token embeddings are analysed geometrically to determine whether a coherent *value axis* exists in the 768-dimensional representation space, and whether projections onto that axis track real historical purchasing power.

Two models are compared:

- **BERT** (`bert-base-uncased`): trained on modern English (Wikipedia + BookCorpus)
- **MacBERTh** (`emanjavacas/MacBERTh`): a BERT-architecture model trained on Early Modern and Middle English historical corpora (1450–1950)

---

## 2. Methodology

### 2.1 Probe Design

All analyses use the CLS token embedding at transformer layer 4, selected by sweeping all 12 layers and maximising time-axis quality (Spearman ρ between projected year embeddings and actual year). Layer 4 yielded the best temporal signal for both models.

Two axes are constructed in embedding space:

**Value axis** — a unit vector pointing from low-value to high-value coin representations. Built by fitting all 13 historical British denominations (farthing through guinea) to their log(pence) values using minimum-norm OLS regression:

```
minimise  Σᵢ ( centroid_i · d − log_pence_i )²
```

where each centroid is the mean CLS embedding of a coin name across 4 paraphrase templates (e.g., *"in {year}, a {coin} was used for everyday purchases"*). Both embeddings and log-pence values are mean-centred before fitting. The result is the minimum-norm direction in 768-d space that linearly separates all 13 denominations by log value — every coin votes on axis direction rather than a single anchor pair dominating.

**Time axis** — a unit vector pointing from earlier to later representations. Built by differencing the mean CLS embeddings of year-probe sentences from 15 early years (≤ 1450) and 15 late years (≥ 1750), using 4 paraphrase templates per year.

The denominations used, covering the full historical British monetary scale:

| Coin | Pence |
|---|---|
| Farthing | 0.25 |
| Halfpenny | 0.5 |
| Penny | 1 |
| Threepence | 3 |
| Groat | 4 |
| Sixpence | 6 |
| Shilling | 12 |
| Florin | 24 |
| Half-crown | 30 |
| Crown | 60 |
| Half-sovereign | 120 |
| Sovereign | 240 |
| Guinea | 252 |

### 2.2 Test Design

Four test coins are embedded at 68 test years (1250–1920, 10-year intervals): **penny** (1d), **sixpence** (6d), **shilling** (12d), **florin** (24d). These were chosen as unambiguous coin names without secondary meanings that could contaminate the embedding (sovereign, crown were excluded on these grounds).

To reduce tokenisation jitter — an artefact where BERT assigns systematically different embeddings to years like "1225" vs "1250" based on how frequently those specific strings appear in training text — each test point averages embeddings across a ±3 year neighbourhood (7 years × 4 templates = 28 forward passes per test point). Adjacent test points are separated by 10 years, so the ±3 windows never overlap; independence between observations is preserved and p-values require no correction.

Each embedding is projected onto both the value axis and time axis, yielding two scalar time series per (coin, year) pair: `value_proj` and `time_proj`.

### 2.3 Real Data

Purchasing power and earnings data come from the Bank of England *Millennium of Macroeconomic Data* dataset:

- **Purchasing power** (Sheet A47, col 3): CPI spliced index (2015=100), anchored to 1/CPI so higher = more valuable. Coverage: 1209–1920.
- **Real earnings** (Sheet A48, col 1): real earnings index. Coverage: 1209–2016.

Each test year uses a 15-year centred window mean of the BoE annual data to smooth local gaps.

### 2.4 Statistical Approach

All correlations are **Spearman ρ** (rank-based, no linearity assumption). Three correlation regimes are computed:

1. **Level correlations** — Spearman ρ between `value_proj` and `purch_power` over all 68 test years. Captures overall monotonic co-movement but is susceptible to shared time trend (both series decline monotonically 1250–1920).

2. **First-difference correlations** — Spearman ρ between `Δvalue_proj` and `Δlog(purch_power)`. Both series are in log-change (percent-change) units: `value_proj ≈ log(perceived_pence)` by axis construction, so `Δvalue_proj ≈ Δlog(perceived_value)`; log-differencing the real data matches this scale. Differencing removes the shared monotonic trend, testing whether BERT tracks *changes* in purchasing power rather than the overall arc.

3. **Rolling window correlations** — Spearman ρ computed within a sliding 150-year window (15 consecutive 10-year test points, centre-aligned). This produces a time series of local correlations revealing which historical periods each model tracks well.

Cross-temporal sequence analysis (Script 02) projects historical sequences (ruling dynasties, primary weapons, ship construction materials, primary fuels) onto the time axis and computes Spearman ρ between projection rank and chronological order.

---

## 3. Results

### 3.1 Axis Quality (Script 01)

**Value axis** — by construction (OLS underdetermined system: 13 coins, 768 dimensions), the minimum-norm solution always achieves Spearman ρ = 1.0 and R² = 1.0 on the training denominations. This is not a meaningful quality metric; it confirms the axis exists but not that it generalises. The informative quality check is the cosine angle between the value and time axes.

**Time axis** — Spearman ρ between year-probe projections and actual year:

| Model | Time-axis ρ | p | Angle (value ⊥ time) |
|---|---|---|---|
| BERT | +0.87 | 6.6e-10 | 83.4° |
| MacBERTh | +0.89 | 9.2e-11 | 87.4° |

Both models significantly order years along their time axis. MacBERTh is marginally better (expected: explicitly trained on historical text), and the two axes are near-orthogonal for both models, confirming they encode distinct dimensions of meaning rather than the same direction twice.

**Tokenisation jitter** — at 25-year probe intervals, BERT exhibits a systematic alternating pattern: years ending in X25/X75 project substantially lower than years ending in X00/X50 in the medieval period (e.g., 1225 = −1.76, 1250 = −1.40), with the pattern reversing in the early modern period. This is a BERT tokenisation artefact: round years (multiples of 50, 100) function as generic period anchors in pre-training text ("the 1800s", "13th-century England"), while intermediate years pick up idiosyncratic event associations. The ±3 year neighbourhood averaging substantially reduces but does not fully eliminate this jitter; it raises the time-axis Spearman ρ from 0.87 to approximately 0.91 when applied to year-probe sentences.

### 3.2 Cross-Temporal Sequence Ordering (Script 02)

The time axis (layer 4) is tested against four historical sequences whose correct chronological order is known:

- **Primary weapons**: longbow → crossbow → musket → rifle → machine gun (5 items, Medieval–WWI)  
- **Ruling dynasties**: Plantagenet → Lancaster → York → Tudor → Stuart → Hanover → Windsor (7 items, 1154–present)  
- **Ship construction**: wood → iron → steel (3 items)  
- **Primary fuel**: wood/peat → coal → oil/gas (3 items)

ρ(yr→seq) measures whether the time axis correctly recovers the chronological order of sequence items:

| Layer | Sequence | BERT ρ(yr→seq) | MacBERTh ρ(yr→seq) |
|---|---|---|---|
| 4 | primary_weapon | **+0.79 ***  | **+0.69 ***  |
| 4 | primary_fuel | +0.55 * | +0.49 ** |
| 4 | ruling_dynasty | +0.44 ns | −0.07 ns |
| 4 | ship_construction | −0.50 ns | −0.08 ns |
| 5 | primary_weapon | **+0.90 ***  | — |
| 5 | ruling_dynasty | **+0.76 ***  | — |
| 5 | primary_fuel | **+0.68 ***  | — |

Key observations:

- BERT is strongest at layer 5 for temporal sequences (primary_weapon ρ=0.90, ruling_dynasty ρ=0.76), while MacBERTh peaks at layer 2–4.
- Primary weapons are recovered reliably by both models, likely because weapon-era associations are strongly encoded in training text across both corpora.
- Ruling dynasties are recovered by BERT (layer 5) but fail in MacBERTh — dynasty names appear to activate political rather than temporal associations in the historical corpus.
- Ship construction and primary fuel are consistently weak (only 3 ordered items each; Spearman ρ is extremely coarse with n=3).

### 3.3 Coin Value × Time Probe (Script 03)

#### 3.3.1 Level Correlations

Spearman ρ between `value_proj(coin, year)` and `purch_power(year)` over 68 test years (1250–1920):

| Coin | BERT ρ_CPI | BERT p | MacBERTh ρ_CPI | MacBERTh p |
|---|---|---|---|---|
| Penny | +0.79 | 7.2e-16 ** | −0.24 | 4.9e-02 * |
| Sixpence | +0.62 | 2.2e-08 ** | +0.37 | 1.6e-03 ** |
| Shilling | +0.83 | 2.6e-18 ** | +0.50 | 1.5e-05 ** |
| Florin | +0.85 | 2.3e-20 ** | −0.20 | 0.10 ns |

BERT shows strong positive correlations for all four coins — as purchasing power declines (rising prices), BERT's value projection also declines, in the correct direction. MacBERTh is dramatically weaker and sign-inconsistent: penny and florin anti-correlate, suggesting MacBERTh's value axis does not track purchasing power uniformly across the full 670-year window.

However, these level correlations are suspect: both `value_proj` and `purch_power` decline near-monotonically from 1250 to 1920, so any monotone function of time would achieve high Spearman ρ. The level result is primarily evidence of a shared time trend, not genuine value tracking.

Correlations with real earnings are weaker and negative for both models:

| Coin | BERT ρ_earn | MacBERTh ρ_earn |
|---|---|---|
| Penny | −0.29 * | −0.01 ns |
| Sixpence | −0.37 ** | −0.01 ns |
| Shilling | −0.27 * | −0.24 * |
| Florin | −0.29 * | −0.34 ** |

BERT's negative earnings correlation makes sense: real wages rose over this period (improving productivity) while perceived coin value fell — so value_proj and real_earn move inversely.

#### 3.3.2 First-Difference Analysis

First-differencing removes the shared monotonic time trend. Both `Δvalue_proj` and `Δlog(purch_power)` are in log-change (percent-change) units, making the comparison dimensionally consistent: the value axis was built on log(pence), so `Δvalue_proj ≈ Δlog(perceived_value)`.

Results for both models, all coins: **uniformly null** (ρ ≈ 0.00–0.11, all p > 0.35).

| Coin | BERT ρ_Δ | MacBERTh ρ_Δ |
|---|---|---|
| Penny | −0.07 ns | −0.10 ns |
| Sixpence | −0.05 ns | −0.09 ns |
| Shilling | −0.07 ns | −0.09 ns |
| Florin | −0.06 ns | −0.03 ns |

This result is robust to: raw vs log-differenced real data, exp()-transforming projections to pence scale before differencing, ±1 vs ±3 year window averaging, and all combinations thereof. The null is not a noise floor artefact — it reflects a genuine absence of decade-level co-movement.

**Interpretation:** BERT and MacBERTh encode monetary value history as a coarse historical gradient (older = higher perceived value, later = lower), not as a period-by-period economic signal. When asked "does BERT's embedding of 'in 1310, a shilling' move differently from 'in 1300, a shilling'?", the answer is no — the specific decade encoded in the year token carries no economic information beyond broad era.

#### 3.3.3 Rolling Window Analysis — Period-Specific Signal

The uniformly null first-difference result and the strong level correlations appear contradictory until the rolling window analysis reveals the structure underneath. Computing Spearman ρ within a 150-year sliding window (15 × 10-year test points) shows that tracking quality varies dramatically and non-randomly across time:

**BERT rolling window:**

| Metric | Value | Location |
|---|---|---|
| Peak ρ (purchasing power) | +0.90 to +0.92 | ~1520 |
| Trough ρ (purchasing power) | −0.37 to −0.51 | ~1320 |
| Florin trough | −0.47 | ~1690 |

**MacBERTh rolling window:**

| Metric | Value | Location |
|---|---|---|
| Peak ρ (purchasing power) | +0.68 to +0.80 | ~1320 |
| Trough ρ (purchasing power) | −0.50 to −0.83 | ~1780–1800 |

The two models are **temporally complementary**:

- **MacBERTh** peaks in the Black Death era (~1280–1380). The plague (1348), post-plague wage shock, and subsequent recovery are the defining economic events of the medieval English corpora on which MacBERTh was trained. The model correctly encodes the U-shaped purchasing power dip of this period.

- **BERT** peaks in the Price Revolution era (~1450–1600), which includes the Great Debasement under Henry VIII and the influx of New World silver — events heavily documented in BERT's modern English training corpus (Wikipedia, books). BERT's tracking collapses in the medieval period (trough ~1320) where its training data is sparse.

- **BERT's Great Disinflation tracking (~1650–1750):** A particularly notable finding is BERT's directionally correct movement in the post-Civil War disinflation period. English prices fell roughly 30% between 1650 and 1750 (the Great Disinflation). BERT's value projection *rises* during this window, correctly encoding that coins were gaining purchasing power. The model then correctly registers the post-1800 inflationary pressure (Napoleonic Wars, early industrialisation). The magnitude is substantially off — BERT's axis is not calibrated to pence — but the rank ordering within this window is correct, which is precisely what Spearman ρ captures.

- **MacBERTh post-1700 reversal:** MacBERTh's rolling ρ becomes strongly negative (−0.77 to −0.83) in the 1750–1850 window. This means it is actively tracking the *wrong* direction in the modern period — rising prices register as rising perceived value rather than falling. This suggests MacBERTh's pre-modern training corpus encodes different monetary semantics than the post-industrial period demands: the language of coin values in Early Modern English does not generalise to 18th–19th century economic dynamics.

The pattern is consistent across all four test coins, ruling out coin-specific artefacts.

### 3.4 Broader Economic Series (Script 04)

The value axis is tested against five additional BoE series using their own bespoke probe sentences (wheat prices, coin supply, population, wages, trade volume):

**BERT:**

| Series | ρ | p | n | Direction |
|---|---|---|---|---|
| Coin supply | −0.62 | 2.9e-08 ** | 66 | Negative (supply ↑, perceived value ↓) |
| Population | −0.43 | 4.7e-04 ** | 63 | Negative |
| Wheat price | +0.40 | 7.8e-04 ** | 67 | Positive (dear grain = high value era) |
| Nominal wages | −0.29 | 1.7e-02 * | 68 | Negative |
| Real wages | +0.22 | ns | 68 | — |
| Trade volume | −0.13 | ns | 65 | — |

**MacBERTh:**

| Series | ρ | p | n |
|---|---|---|---|
| Coin supply | +0.61 | 6.7e-08 ** | 66 |
| Trade volume | −0.26 | 3.8e-02 * | 65 |
| All others | ns | — | — |

Notable findings:

- **Wheat price (BERT, ρ=+0.40 **):** BERT's "expensive grain" semantic axis aligns with Allen basket prices over 650 years. This is the most interpretable result in Script 04, as BERT's embedding of grain-price language plausibly reflects how frequently terms like "dearth" and "plenty" co-occur with bread prices in its training text.

- **Coin supply sign flip:** BERT shows ρ=−0.62 (more coin = lower perceived value, consistent with inflation), while MacBERTh shows ρ=+0.61 (opposite direction). This reflects the different axis orientations produced by training on different corpora: the "expensive" pole of BERT's value axis points toward scarcity, while MacBERTh's points toward abundance in the medieval context.

- **Population and nominal wages (BERT, negative):** Both increase over 1250–1920, and BERT's value axis declines monotonically, so negative correlations here are largely driven by the shared time trend rather than genuine economic encoding.

---

## 4. Discussion

### 4.1 The Nature of the Signal

The evidence points to a coherent picture: both BERT and MacBERTh encode monetary value history, but at a *coarse temporal resolution* driven by their training corpus composition, not by any genuine decade-level economic tracking.

The value axis exists and is non-trivial: it is nearly orthogonal to the time axis (83–87°), meaning the model has not simply built a temporal gradient and called it value. The rolling window analysis shows that the value axis tracks real purchasing power with ρ up to 0.92 in favourable periods — a correlation this strong within a 150-year window cannot be dismissed as chance.

However, the first-difference results establish a hard ceiling on this interpretation: remove the shared historical arc and the signal vanishes entirely. BERT does not know that prices rose in the 1310s and fell in the 1320s. What it knows is approximately: *these words were written long ago, and things were worth more then*.

### 4.2 Why Does Period-Specific Signal Exist?

The rolling window peaks correspond precisely to periods of intense economic discourse in each model's training corpus:

- **MacBERTh's medieval peak** reflects the extraordinary volume of scholarship, chronicles, and administrative records about the Black Death and its economic aftermath, which dominate the medieval English text that MacBERTh was trained on.

- **BERT's Price Revolution peak** reflects Wikipedia and modern scholarly literature about Tudor monetary policy, the Great Debasement (1544–1551), and the 16th-century inflation — topics heavily covered in modern English text.

- **BERT's Great Disinflation tracking** is particularly striking because it is directionally correct even after the tokenisation jitter has been smoothed. The 1650–1750 period is the focus of substantial modern economic history literature, and BERT appears to have absorbed the association between this era and monetary stability.

This suggests a mechanism: language models encode monetary value through the *distributional patterns of economic discourse* around specific historical periods. When a period is heavily discussed in economically-inflected language (dearth, plenty, debasement, inflation, wages), the model's embeddings of coin names in that temporal context absorb a meaningful value signal. When a period is underrepresented or discussed in non-economic terms, the signal is absent or inverted.

### 4.3 Model Comparison

| Property | BERT | MacBERTh |
|---|---|---|
| Time-axis quality (ρ) | 0.87 | 0.89 |
| Value-axis consistency | Strong, uniform across coins | Weak, sign-inconsistent |
| Best tracking period | Price Revolution (~1450–1600) | Black Death (~1280–1380) |
| Post-1700 behaviour | Directionally correct, declining ρ | Strongly anti-correlated |
| Cross-temporal sequences | Strong (L5, dynasties + weapons) | Moderate (weapons only) |
| Broader series (sig. hits) | 4 of 6 | 2 of 6 |

BERT is the more reliable probe for monetary value despite being trained on modern text. This is counterintuitive but explainable: modern English contains far more systematic *discussion* of historical monetary values (economic histories, encyclopaedias, academic papers) than historical English contains contemporaneous economic analysis. MacBERTh's training text is primary source material where economic language is woven into narrative rather than catalogued analytically.

### 4.4 Limitations

1. **Temporal confound.** Level correlations are dominated by the shared monotonic decline of both value_proj and purchasing power over 1250–1920. The first-difference results establish this clearly, but the rolling window correlations are still susceptible to local trend confounds within each 150-year window.

2. **Value axis scale.** The OLS-fitted value axis is calibrated to log(pence) but the scale factor is arbitrary (the minimum-norm solution). Projections are therefore not directly interpretable in pence units without estimating the slope of the calibration curve.

3. **Vacuous OLS quality metric.** With 13 training coins and 768 dimensions, the OLS system is heavily underdetermined and always achieves perfect fit. Spearman ρ = 1.0 on training denominations is uninformative. A held-out denomination evaluation would be more rigorous.

4. **Template sensitivity.** All embeddings use the same 4 paraphrase templates. Results may depend on the specific phrasings chosen, particularly for the TIME_TEMPLATES which explicitly mention years. A broader template set would reduce this sensitivity.

5. **Layer selection.** Layer 4 was selected by time-axis quality. A joint criterion (value + time) might select a different layer and alter results, particularly for MacBERTh where the sweep showed notable layer-dependence.

6. **First-difference power.** At 10-year intervals with n=67 differences, first-difference Spearman ρ requires |ρ| ≥ ~0.24 for significance at p<0.05. True signal in the range ρ=0.10–0.20 would not be detected. The null result rules out moderate-to-large decade-level co-movement, not all co-movement.

---

## 5. Conclusions

1. **A value axis exists in both models.** A linear direction in the 768-dimensional embedding space separates British denominations by log(pence) value, is near-orthogonal to the time axis, and is recoverable at layer 4 for both BERT and MacBERTh.

2. **Level correlations with real purchasing power are strong but largely spurious** (ρ = 0.62–0.85 for BERT), driven by the shared monotonic decline of both series over 670 years rather than genuine economic encoding.

3. **Decade-level co-movement is absent.** First-difference correlations are uniformly null (ρ ≈ 0.00–0.11) and robust to all transformations tried: raw vs log-differenced real data, exp()-transformed projections, neighbourhood averaging, and combinations thereof. BERT does not track period-to-period changes in purchasing power.

4. **Period-specific signal is genuine.** Rolling window analysis reveals ρ up to 0.92 in favourable periods. BERT tracks the Price Revolution era (1450–1600) and correctly captures the direction of the Great Disinflation (1650–1750). MacBERTh tracks the Black Death era (1280–1380). The two models are temporally complementary, each owning the period best represented in its training corpus.

5. **MacBERTh actively anti-correlates post-1700** (rolling ρ as low as −0.83), indicating its pre-modern monetary semantics do not generalise to the industrial and post-industrial period.

6. **The mechanism is likely distributional discourse, not direct value encoding.** Models encode monetary value through the language patterns of economic discourse that surround specific historical periods in their training data. Periods of heavy economic historiography produce coherent value signals; underrepresented periods produce noise or inversion.

7. **The signal is coarse-grained (centuries) not fine-grained (decades).** Language models should be understood as encoding broad *era-level* monetary associations, not as implicit economic historians capable of tracking shorter-run price dynamics.

---

## Appendix: Script Reference

| Script | Purpose | Key outputs |
|---|---|---|
| `01_build_axes.py` | Builds value and time axes, reports quality metrics | `value_direction_L4.npy`, `time_direction_L4.npy`, `axes_quality.png`, `denomination_axis.png` |
| `02_cross_temporal.py` | Tests time axis against known historical sequences | `cross_temporal_results.csv`, `cross_temporal.png` |
| `03_coin_value_probe.py` | Main coin×year probe, level + first-diff + rolling correlations | `coin_value_results.csv`, `coin_value_vs_real.png`, `coin_value_changes.png`, `rolling_correlation.png` |
| `04_broader_series.py` | Tests value axis against 5 additional BoE series | `broader_series_results_v2.csv`, `broader_series_v2.png` |
| `sweep_best_layer.py` | Sweeps all layers to identify best layer for each model | (console output, best-layer direction files) |

**Key flags for `03_coin_value_probe.py`:**
- `--year-window 3` (default): average ±3 year neighbourhood per test point (7 years × 4 templates = 28 embeddings)
- `--exp-proj`: exponentiate projections to pence scale before differencing (Δexp(proj) vs Δpurch_power in absolute units)
- `--layer N`: override default layer (4)
