# LLM Prompting Benchmarks

## Overview

This directory contains a standardized framework for comparing how different Large Language Models (LLMs) respond to identical data science tasks. The goal is to create reproducible, controlled experiments that evaluate model performance across multiple dimensions, not anecdotal commentary.

## Methodology: Controlled Prompt Experiments

### Core Principle

A **controlled prompt experiment** follows the scientific method by changing **only one variable at a time** while keeping all other factors constant. This allows us to isolate the impact of specific variables on model performance.

### Variables to Control

When designing experiments, consider controlling these variables:

1. **Prompt Strategy**
   - Zero-shot (no examples)
   - Few-shot (with examples)
   - Chain-of-thought
   - Role-based prompting

2. **Model Configuration**
   - Model version (e.g., GPT-4, Claude-3, Llama-2-70b)
   - Temperature setting
   - Max tokens
   - Top-p sampling

3. **Task Characteristics**
   - Task type (classification, generation, reasoning, coding)
   - Task complexity (simple, moderate, complex)
   - Domain (statistics, ML, data engineering)

4. **Input Format**
   - Prompt structure
   - Context length
   - Data format (JSON, CSV, plain text)

### Experimental Design Template

For each experiment:

1. **Define the Hypothesis**
   - What are you testing? (e.g., "Does few-shot prompting improve accuracy on statistical reasoning tasks?")

2. **Identify Control Variables**
   - List all variables you will keep constant
   - Example: Same model version, same temperature, same task type

3. **Identify the Independent Variable**
   - The ONE thing you will change
   - Example: Prompt strategy (zero-shot vs. few-shot)

4. **Define Success Metrics**
   - Quantitative: Success rate, accuracy, F1 score
   - Qualitative: Reasoning depth, hallucination rate, verbosity

5. **Create Test Cases**
   - Minimum 5-10 test cases per condition
   - Ensure tasks are representative and diverse

6. **Document Everything**
   - Exact prompts used
   - Model configurations
   - Raw outputs
   - Evaluation criteria

### Example Experiment

**Hypothesis:** Few-shot prompting improves accuracy on statistical hypothesis testing problems.

**Control Variables:**
- Model: GPT-4-turbo-preview
- Temperature: 0.7
- Max tokens: 1000
- Task domain: Statistical hypothesis testing

**Independent Variable:**
- Prompt strategy: Zero-shot vs. 3-shot

**Test Cases:** 10 hypothesis testing problems of varying difficulty

**Metrics:**
- Correct answer rate (%)
- Average reasoning steps
- Hallucination instances

## Evaluation Dimensions

### 1. Reasoning Depth
- Does the model show its work?
- Are intermediate steps logical?
- Does it identify edge cases?

### 2. Hallucination Rate
- Frequency of factual errors
- Made-up citations or data
- Contradictions in responses

### 3. Numerical Reliability
- Accuracy of calculations
- Proper use of statistical methods
- Correct interpretation of results

### 4. Verbosity
- Conciseness vs. over-explanation
- Signal-to-noise ratio
- Adherence to requested format

### 5. Tool-Use Performance
- When given code execution capabilities
- API call accuracy
- Data manipulation correctness

## Running Experiments

1. **Design your experiment** using the template above
2. **Configure the benchmark script** (`run_benchmark.py`)
3. **Execute the experiment** and collect raw outputs
4. **Analyze results** using the results template
5. **Document findings** with statistical rigor
6. **Share reproducible artifacts** (prompts, configs, raw data)

## Best Practices

- **Randomize test order** to avoid order effects
- **Run multiple trials** for statistical significance
- **Blind evaluation** when possible (evaluate outputs without knowing which model produced them)
- **Version control everything** (prompts, configs, scripts)
- **Report null results** (when variables don't make a difference)
- **Include failure cases** in documentation
- **Share raw data** for community validation

## Reproducibility Checklist

Before publishing results, ensure:

- [ ] Exact model versions documented
- [ ] All hyperparameters recorded
- [ ] Complete prompt texts shared
- [ ] Random seeds noted (if applicable)
- [ ] Evaluation rubric defined
- [ ] Raw outputs preserved
- [ ] Statistical tests applied
- [ ] Confidence intervals calculated

## Contributing

When adding new experiments:

1. Create a subdirectory with a descriptive name (e.g., `statistical-reasoning/`)
2. Include a README with hypothesis and methodology
3. Provide all prompts and configurations
4. Use the results template for documentation
5. Submit raw data in a structured format

## Resources

- See `results-template.md` for the standard result matrix format
- See `run_benchmark.py` for the reference implementation
- Consult the main repository for related data science resources
