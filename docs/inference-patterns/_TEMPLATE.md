# [Model Name] - Inference Design Pattern

> **Contributors:** [Your Name/Team] | **Last Updated:** [YYYY-MM-DD]

## Model Family

**Model Name:** [e.g., GPT-4, Claude 3 Opus, Llama 3.1 70B]  
**Provider:** [OpenAI, Anthropic, Meta, etc.]  
**Version/Release Date:** [Specify version or release date]  
**Context Window:** [e.g., 128K tokens]  
**Pricing (as of date):** [Input: $X/1M tokens, Output: $Y/1M tokens]  
**Modality:** [Text-only, Multimodal (Vision), Audio, etc.]

## Key Behavioral Tendencies

Describe the model's characteristic behaviors observed across multiple use cases:

- **Verbosity:** [Terse/Moderate/Verbose - how the model responds by default]
- **Refusal Triggers:** [Known topics or phrasings that trigger refusals/safety responses]
- **Instruction Following:** [How well the model adheres to system prompts and constraints]
- **Formatting Consistency:** [JSON, XML, Markdown - which formats work best]
- **Temperature Sensitivity:** [How temperature affects output quality/creativity]
- **Other Notable Tendencies:** [Any unique behavioral patterns]

### Example
```python
# Example prompt demonstrating a key behavioral tendency
# [Describe what this demonstrates]
```

## Reasoning Characteristics

Document the model's performance on different reasoning tasks:

- **Chain-of-Thought (CoT) Performance:** [Effectiveness when prompted to "think step by step"]
- **Mathematical Reasoning:** [Strengths/weaknesses in arithmetic, algebra, calculus]
- **Logical Reasoning:** [Performance on deduction, induction, analogy tasks]
- **Code Generation:** [Quality and correctness of generated code]
- **Multimodal Reasoning:** [If applicable - vision, audio, etc.]
- **Planning & Multi-Step Tasks:** [Ability to break down complex problems]

### Example
```python
# Example demonstrating reasoning capability or limitation
# [Describe the reasoning pattern]
```

## Known Failure Modes

Document reproducible failure patterns with examples:

### 1. [Failure Mode Name]
**Description:** [What goes wrong]  
**Trigger Conditions:** [What causes this failure]  
**Frequency:** [Common/Occasional/Rare]  
**Impact:** [High/Medium/Low]

**Example:**
```python
# Prompt that triggers the failure
prompt = """
[Your example prompt]
"""

# Expected behavior:
# [What should happen]

# Actual behavior:
# [What actually happens]
```

**Mitigation:**
- [Strategy 1 to avoid this failure]
- [Strategy 2 to avoid this failure]

### 2. [Additional Failure Modes]
[Repeat structure above for each failure mode]

## Optimization Strategies

### Cost Optimization

| Strategy | Impact | Implementation | Trade-offs |
|----------|--------|----------------|------------|
| [Strategy name] | [Cost reduction %] | [How to implement] | [Quality/latency impacts] |
| Use smaller context | 20-40% savings | Chunk documents, summarize history | May lose relevant context |
| Batch requests | 10-30% savings | Queue non-urgent tasks | Increased latency |

### Performance Optimization

| Strategy | Impact | Implementation | Best Use Cases |
|----------|--------|----------------|----------------|
| [Strategy name] | [Latency/quality improvement] | [How to implement] | [When to use] |
| Streaming responses | Reduced perceived latency | Use streaming API | Interactive applications |
| Few-shot examples | +15-30% accuracy | Include 3-5 examples | Structured outputs |

### Prompt Engineering Best Practices

1. **System Prompts:**
   - [Best practice 1]
   - [Best practice 2]

2. **User Prompts:**
   - [Best practice 1]
   - [Best practice 2]

3. **Output Formatting:**
   - [Best practice 1]
   - [Best practice 2]

### Example Implementation
```python
# Complete example showing optimized implementation
# for a common data science task

import openai  # or relevant library

def optimized_prompt_example():
    """
    [Description of what this example demonstrates]
    """
    system_prompt = """
    [Optimized system prompt]
    """
    
    user_prompt = """
    [Optimized user prompt template]
    """
    
    # [Additional implementation details]
    pass
```

## Real-World Data Science Workflows

### Use Case 1: [Task Name]
**Task Description:** [Brief description]  
**Model Configuration:**
- Temperature: [X]
- Max tokens: [Y]
- System prompt strategy: [Z]

**Prompt Template:**
```python
# [Prompt template for this use case]
```

**Performance Notes:** [Observed quality, latency, cost]

### Use Case 2: [Task Name]
[Repeat structure above]

## Empirical Observations

Document quantitative findings from your testing:

| Metric | Value | Test Conditions | Notes |
|--------|-------|----------------|-------|
| Average latency | [X ms] | [Context size, task type] | [Any relevant notes] |
| Accuracy on [task] | [X%] | [Dataset/benchmark] | [Comparison to other models] |
| Cost per 1K requests | $[X] | [Average tokens/request] | [Date of testing] |

## References & Resources

- **Official Documentation:** [Link]
- **Research Papers:** [Links to relevant papers]
- **Community Discussions:** [Links to discussions, blog posts]
- **Benchmarks:** [Links to relevant benchmark results]

## Contributing

This pattern document was tested on [dates/versions]. If you observe different behaviors or have additional insights:
1. Update this document with empirical evidence
2. Include example prompts/code demonstrating your findings
3. Note the testing conditions (date, model version, context)

---

**Template Version:** 1.0  
**Last Updated:** 2026-02-18
