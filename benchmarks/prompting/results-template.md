# LLM Prompting Benchmark Results Template

## Experiment Metadata

**Experiment ID:** `YYYY-MM-DD-experiment-name`  
**Date Conducted:** YYYY-MM-DD  
**Experimenter:** Your Name/GitHub Handle  
**Hypothesis:** [State your hypothesis here]  
**Independent Variable:** [What you're changing]  
**Control Variables:** [What you're keeping constant]

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model Version | e.g., GPT-4-turbo-preview, Claude-3-Opus, Llama-2-70b-chat |
| Temperature | e.g., 0.7 |
| Max Tokens | e.g., 1000 |
| Top-p | e.g., 1.0 |
| Frequency Penalty | e.g., 0.0 |
| Presence Penalty | e.g., 0.0 |
| System Prompt | [Include if used] |

---

## Results Matrix

### Primary Results

| Test Case ID | Model Version | Task Type | Prompt Strategy | Success Rate (%) | Avg Response Time (s) | Notes on Hallucinations |
|--------------|---------------|-----------|-----------------|------------------|-----------------------|------------------------|
| TC-001 | GPT-4-turbo | Statistical Reasoning | Zero-shot | 80% | 3.2 | None observed |
| TC-001 | GPT-4-turbo | Statistical Reasoning | Few-shot (3) | 95% | 4.1 | None observed |
| TC-002 | GPT-4-turbo | ML Classification | Zero-shot | 70% | 2.8 | 1 instance: cited non-existent library |
| TC-002 | GPT-4-turbo | ML Classification | Few-shot (3) | 85% | 3.5 | None observed |
| ... | ... | ... | ... | ... | ... | ... |

### Extended Metrics

| Test Case ID | Reasoning Depth (1-5) | Numerical Reliability (1-5) | Verbosity Score (1-5) | Tool-Use Accuracy (%) | Edge Cases Handled |
|--------------|-----------------------|-----------------------------|-----------------------|-----------------------|-------------------|
| TC-001 (zero-shot) | 3 | 4 | 3 | N/A | 2/3 |
| TC-001 (few-shot) | 4 | 5 | 3 | N/A | 3/3 |
| TC-002 (zero-shot) | 3 | 3 | 4 | 70% | 1/2 |
| TC-002 (few-shot) | 4 | 4 | 4 | 85% | 2/2 |
| ... | ... | ... | ... | ... | ... |

**Scoring Guide:**
- **Reasoning Depth:** 1=None, 2=Minimal, 3=Adequate, 4=Good, 5=Excellent
- **Numerical Reliability:** 1=Many errors, 2=Several errors, 3=Few errors, 4=Rare errors, 5=No errors
- **Verbosity:** 1=Too terse, 2=Somewhat terse, 3=Appropriate, 4=Somewhat verbose, 5=Excessively verbose

---

## Detailed Observations

### Hallucination Analysis

| Test Case ID | Hallucination Type | Description | Severity (Low/Med/High) |
|--------------|-------------------|-------------|------------------------|
| TC-002 | Fabricated Library | Cited "sklearn.advanced.AutoClassifier" which doesn't exist | Medium |
| TC-005 | Incorrect Statistic | Claimed 95% confidence interval when calculation showed 90% | High |
| ... | ... | ... | ... |

### Reasoning Quality Examples

**Best Example (TC-001, Few-shot):**
```
The model correctly identified the null hypothesis, selected the appropriate 
statistical test (independent t-test), checked assumptions (normality, equal 
variance), performed calculations with correct formulas, and interpreted 
p-value appropriately.
```

**Worst Example (TC-003, Zero-shot):**
```
The model jumped directly to conclusion without showing work, used wrong 
statistical test (chi-square instead of t-test), and misinterpreted the 
significance level.
```

### Verbosity Analysis

**Average Response Length by Condition:**
- Zero-shot: 250 words
- Few-shot (3): 320 words
- Chain-of-thought: 450 words

**Optimal Range:** 200-400 words for this task type

---

## Statistical Analysis

### Summary Statistics

| Metric | Zero-shot | Few-shot | Difference | p-value | Significant? |
|--------|-----------|----------|------------|---------|--------------|
| Success Rate | 75% | 90% | +15% | 0.023 | Yes (p<0.05) |
| Hallucination Rate | 15% | 5% | -10% | 0.045 | Yes (p<0.05) |
| Avg Reasoning Depth | 3.2 | 4.1 | +0.9 | 0.012 | Yes (p<0.05) |
| Avg Response Time | 3.0s | 3.8s | +0.8s | 0.156 | No |

**Statistical Test Used:** Two-sample t-test (or appropriate test)  
**Sample Size:** n=10 per condition  
**Confidence Level:** 95%

---

## Cross-Model Comparison (Optional)

| Model | Prompt Strategy | Avg Success Rate | Avg Reasoning Depth | Hallucination Rate | Cost per 1K Tokens |
|-------|----------------|------------------|---------------------|-------------------|-------------------|
| GPT-4-turbo | Few-shot | 90% | 4.1 | 5% | $0.03 |
| Claude-3-Opus | Few-shot | 88% | 4.3 | 3% | $0.075 |
| Llama-2-70b | Few-shot | 75% | 3.5 | 12% | Free (self-hosted) |
| ... | ... | ... | ... | ... | ... |

---

## Conclusions

### Key Findings

1. [Finding 1 with supporting data]
2. [Finding 2 with supporting data]
3. [Finding 3 with supporting data]

### Limitations

- [Limitation 1]
- [Limitation 2]
- [Limitation 3]

### Recommendations

- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

### Future Work

- [ ] Expand test set to include [specific area]
- [ ] Test with different temperature settings
- [ ] Compare against additional models
- [ ] Investigate [specific phenomenon observed]

---

## Reproducibility Information

**Raw Data Location:** `[path/to/raw/data]`  
**Analysis Scripts:** `[path/to/analysis/scripts]`  
**Prompt Files:** `[path/to/prompt/files]`  
**Random Seed:** [if applicable]  
**Environment:** Python 3.x, [libraries and versions]

---

## Appendix

### Test Case Definitions

**TC-001: Statistical Hypothesis Testing**
```
Task: Given a dataset with two groups, determine if there is a significant 
difference in means at α=0.05 level.
Data: Group A (n=30): mean=45.2, sd=8.1; Group B (n=30): mean=42.8, sd=7.9
Expected: Use independent t-test, fail to reject null (p>0.05)
```

**TC-002: ML Classification Problem**
```
Task: Recommend an appropriate classification algorithm and explain why.
Context: Imbalanced dataset (1:10 ratio), 50 features, 10,000 samples, 
need interpretability.
Expected: Suggest Random Forest or Logistic Regression with SMOTE, 
explain trade-offs.
```

[Add all test case definitions here]

### Sample Prompts

**Zero-shot Template:**
```
You are a data science expert. Please solve the following problem:

[Task description]

Provide your analysis and conclusion.
```

**Few-shot Template:**
```
You are a data science expert. Here are some examples:

Example 1:
Problem: [Example problem]
Solution: [Example solution]

Example 2:
Problem: [Example problem]
Solution: [Example solution]

Now solve this problem:
[Task description]

Provide your analysis and conclusion.
```

---

**Template Version:** 1.0  
**Last Updated:** 2026-02-18
