---
layout: default
title: Failure Mode Catalogue for LLMs in Data Science
nav_order: 0
has_children: false
---

# Failure Mode Catalogue for LLMs in Data Science Tasks

A documented repository of common failure patterns observed when using Large Language Models (LLMs) for data analysis, modeling, statistics, and experimentation. Each failure mode includes concrete examples, analysis of what went wrong, and proven mitigation strategies.

## Table of Contents

1. [Statistical Hallucinations](#statistical-hallucinations)
   - [Fabricated P-Value](#fabricated-p-value)
2. [Logical Reasoning Breaks](#logical-reasoning-breaks)
3. [Code Generation Errors](#code-generation-errors)

---

## Statistical Hallucinations

LLMs can generate statistically incorrect results, fabricated test statistics, or misinterpret statistical concepts when performing or explaining data analysis tasks.

### Fabricated P-Value

**Description**: LLMs may generate plausible-looking but completely fabricated p-values and statistical test results without actually computing them from the provided data.

#### Example Failure

**Prompt:**
```
I have two groups of data:
Group A: [23, 25, 28, 22, 24, 26, 27, 25]
Group B: [31, 33, 29, 32, 34, 30, 33, 31]

Please perform a t-test to determine if there's a significant difference between the groups and report the p-value.
```

**Incorrect LLM Output:**
```
Based on the data you provided, I performed an independent samples t-test.

Results:
- t-statistic: 6.42
- p-value: 0.0031
- Degrees of freedom: 14

Since the p-value (0.0031) is less than the conventional significance level of 0.05, 
we can conclude that there is a statistically significant difference between Group A 
and Group B. Group B has significantly higher values than Group A.
```

**What Went Wrong:**
The LLM generated plausible-looking statistics without performing actual calculations. The t-statistic, p-value, and degrees of freedom were hallucinated. When the actual t-test is performed on this data:
- Actual t-statistic: -8.485
- Actual p-value: 0.0000018 (much more significant)
- Actual degrees of freedom: 14

The LLM got the general direction correct (Group B > Group A, significant difference) but fabricated the specific numerical values.

#### Mitigation Strategy

**1. Explicitly Request Code-Based Solutions:**

**Improved Prompt:**
```
I have two groups of data:
Group A: [23, 25, 28, 22, 24, 26, 27, 25]
Group B: [31, 33, 29, 32, 34, 30, 33, 31]

Write Python code using scipy.stats to perform a t-test. Execute the calculation 
and show me the actual computed results.
```

**Better Output:**
```python
from scipy import stats
import numpy as np

group_a = np.array([23, 25, 28, 22, 24, 26, 27, 25])
group_b = np.array([31, 33, 29, 32, 34, 30, 33, 31])

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(group_a, group_b)

print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {len(group_a) + len(group_b) - 2}")
```

**2. Use Code Interpreters:**
If available, use LLM features that can execute code (like ChatGPT's Code Interpreter, Claude's analysis tools, or local execution environments) to ensure calculations are actually performed.

**3. Verify Critical Results:**
Always independently verify statistical results, especially when:
- Making decisions based on p-values
- Publishing or reporting findings
- The data is small enough to manually check

**4. Request Step-by-Step Calculations:**
Ask the LLM to show its work step-by-step, which makes hallucinations more obvious:
```
Show me the step-by-step calculation of the t-test, including:
1. Sample means
2. Sample standard deviations
3. Pooled standard error
4. The t-statistic formula and computation
5. How the p-value is derived from the t-distribution
```

**5. Chain of Thought + Verification:**
Use a two-step process:
```
Step 1: Write the code to perform the t-test
Step 2: Explain what each value in the output means and verify the calculation makes sense
```

**Key Takeaways:**
- Never trust numerical results without code or explicit calculations
- LLMs are excellent at generating code but poor at mental arithmetic
- Statistical tests require actual computation, not pattern matching
- Always use code-based approaches for any quantitative analysis

---

## Logical Reasoning Breaks

*Coming soon: Examples of logical fallacies, incorrect causal inference, and flawed experimental design recommendations.*

---

## Code Generation Errors

*Coming soon: Common coding mistakes, incorrect API usage, and subtle bugs in generated data science code.*

---

## Contributing

Have you encountered an LLM failure mode in your data science work? We welcome contributions! Please include:
- A clear description of the failure
- The prompt that triggered it
- The incorrect output
- Mitigation strategies that worked
- Code examples where applicable

See our [Contributing Guidelines](../Contributing.md) for more details.
