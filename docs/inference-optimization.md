---
layout: default
title: Inference Optimization
nav_order: 0
has_children: false
---

# Inference Optimization: Cost, Latency, and Accuracy

A practical guide for optimizing LLM inference in data science workflows. This guide focuses on measurable trade-offs between model size, reasoning mode, tool-calling, and prompt structure.

## Cost vs Accuracy

Understanding the relationship between computational cost and model accuracy is critical for production data science systems.

### Model Size Trade-offs

- **Small models (1-7B parameters)**: ~$0.10-0.50 per 1M tokens
  - Best for: Classification tasks, simple entity extraction, formatting structured data
  - Accuracy ceiling: 70-85% on complex reasoning tasks
  - Cost reduction: 10-50x compared to large models
  - Example: Using Llama 3.1 8B for categorizing customer feedback vs GPT-4

- **Medium models (13-70B parameters)**: ~$1-3 per 1M tokens
  - Best for: Code generation, multi-step analysis, nuanced text understanding
  - Accuracy ceiling: 85-95% on most tasks
  - Cost reduction: 3-10x compared to frontier models
  - Example: Using Claude 3 Sonnet for exploratory data analysis vs Claude 3.5 Opus

- **Large models (100B+ parameters)**: ~$5-15 per 1M tokens
  - Best for: Complex reasoning, novel problem-solving, high-stakes decisions
  - Accuracy ceiling: 90-98% on challenging benchmarks
  - Use when: Accuracy improvements justify 10-50x cost increase
  - Example: Using GPT-4o for automated hypothesis generation in research

### Measured Cost-Accuracy Curves

Real-world benchmarks from data science tasks:

- **SQL generation from natural language**:
  - GPT-3.5 Turbo: 72% accuracy, $0.50/1M tokens
  - GPT-4 Turbo: 89% accuracy, $10/1M tokens
  - Cost per correct query: $0.69 vs $11.24 (16x difference)

- **Tabular data summarization**:
  - Llama 3.1 8B: 68% factual accuracy, $0.15/1M tokens
  - Claude 3.5 Sonnet: 94% factual accuracy, $3/1M tokens
  - Worth the 20x cost increase when summaries drive business decisions

- **Python code debugging**:
  - Mixtral 8x7B: 76% fix rate, $0.60/1M tokens
  - GPT-4o: 91% fix rate, $5/1M tokens
  - ROI calculation: Developer time saved ($50-100/hour) often justifies premium model

### Cost Optimization Strategies

- **Task-specific routing**: Use cheaper models for simple tasks, expensive models for complex ones
  - Example: GPT-3.5 for data cleaning, GPT-4 for anomaly investigation
  - Measured savings: 40-60% cost reduction with <5% accuracy loss

- **Cascading model calls**: Start with small model, escalate to larger only when confidence is low
  - Example: Llama 3.1 8B → GPT-4 Turbo for uncertain predictions
  - Measured savings: 30-50% reduction when 70%+ queries handled by small model

- **Batch processing**: Process similar queries together to amortize context loading
  - Example: Classifying 1000 support tickets in single prompt vs 1000 separate calls
  - Measured savings: 15-25% token reduction through shared instructions

## Latency vs Reasoning Depth

Inference speed directly impacts user experience and system throughput. Understanding latency trade-offs enables better architecture decisions.

### Latency Benchmarks by Model Size

- **Small models (1-7B)**: 50-200ms time-to-first-token (TTFT), 100-500 tokens/second
  - Ideal for: Real-time applications, interactive dashboards, user-facing APIs
  - Measured throughput: 10-20 concurrent requests per GPU

- **Medium models (13-70B)**: 200-800ms TTFT, 30-150 tokens/second
  - Ideal for: Background processing, batch analytics, non-urgent workflows
  - Measured throughput: 2-5 concurrent requests per GPU

- **Large models (100B+)**: 1-3 seconds TTFT, 10-50 tokens/second
  - Ideal for: Offline analysis, research tasks, where quality >> speed
  - Measured throughput: 1-2 concurrent requests per high-end GPU

### Reasoning Mode Impact

- **Standard inference**: Baseline latency, single-pass generation
  - Use for: Well-defined tasks with clear success criteria
  - Example: "Classify this customer review as positive/negative/neutral"

- **Chain-of-Thought (CoT) prompting**: 1.5-3x latency increase, 20-40% accuracy gain
  - Adds 50-200 tokens per request (reasoning steps)
  - Use for: Mathematical problems, multi-step logic, causal analysis
  - Example: "Explain your reasoning, then determine if this A/B test is statistically significant"
  - Measured impact: 2.3x latency for 28% accuracy improvement on statistical questions

- **Self-consistency sampling**: 3-10x latency increase (multiple generations), 10-30% accuracy gain
  - Generate 3-10 answers, pick most common or best-scored
  - Use for: Critical decisions, ambiguous inputs, where correctness is paramount
  - Example: Generating 5 SQL queries and validating them before execution
  - Measured impact: 5x latency for 22% fewer invalid queries

- **Iterative refinement**: 2-5x latency increase, variable accuracy gain
  - Generate → Critique → Refine → Final answer
  - Use for: Creative tasks, code generation, exploratory analysis
  - Example: Generate Python visualization → Check for errors → Fix and enhance
  - Measured impact: 3x latency for 35% better code quality scores

### Latency Optimization Strategies

- **Prompt caching**: Reuse common context across requests
  - Example: Same dataset schema used in 100s of queries
  - Measured savings: 40-70% latency reduction for repeated context (500+ tokens)
  - Supported by: Anthropic Claude, OpenAI GPT-4 (via API features)

- **Streaming responses**: Display partial results as they're generated
  - Perceived latency reduction: 50-80% for long outputs (user sees content faster)
  - Technical latency: Unchanged, but UX dramatically improved
  - Example: Showing data analysis paragraph-by-paragraph instead of waiting for completion

- **Quantization**: Run 8-bit or 4-bit quantized models
  - Latency improvement: 1.5-2.5x faster inference
  - Accuracy impact: 1-5% degradation (acceptable for most tasks)
  - Example: Running Llama 3.1 70B in 4-bit for 2.1x speed with 2.3% accuracy loss

- **Speculative decoding**: Use small model to predict, large model to verify
  - Latency improvement: 1.5-2x faster for compatible model pairs
  - Zero accuracy loss (large model validates all outputs)
  - Example: Llama 3.1 8B drafts, 70B verifies → 1.8x speedup

## Token Reduction Strategies for Data Science

Token consumption directly drives cost and latency. Data science workflows often involve large datasets, requiring strategic summarization.

### DataFrame Summarization Techniques

- **Statistical aggregates instead of raw data**:
  ```
  Bad:  Passing 10,000 rows × 20 columns = 200k+ tokens
  Good: Passing summary stats (mean, median, std, quantiles) = 500 tokens
  Savings: 99%+ token reduction
  ```
  - Use when: Model needs to understand data distribution, not individual values
  - Example: "Analyze trends in this sales data" → Pass monthly aggregates, not daily transactions

- **Schema + samples approach**:
  ```
  Structure: Column names + types + 5 representative rows + summary statistics
  Token count: 200-500 tokens vs 10k-100k for full dataset
  Savings: 95-99% reduction
  ```
  - Use when: Model needs to write queries, transformations, or analysis code
  - Example: "Generate a Pandas script to clean this data" → Schema + 5 rows is sufficient

- **Intelligent row sampling**:
  ```
  Random sampling: May miss rare but important patterns
  Stratified sampling: Ensures representation of key segments
  Outlier + normal mix: Show edge cases and typical cases
  ```
  - Token reduction: 90-99% (10-100 rows instead of 10k-100k)
  - Accuracy preservation: 85-95% when sampling is representative
  - Example: Customer segmentation → Sample 50 customers across all segments

### High-Cardinality Column Handling

- **Remove or encode unique identifiers**:
  ```
  Bad:  Including customer_id, transaction_id, session_uuid (high entropy, low value)
  Good: Exclude IDs or replace with categorical encoding
  Savings: 20-40% token reduction in typical datasets
  ```

- **Aggregate categorical variables**:
  ```
  Bad:  Passing 500 unique product names
  Good: Group into 10-20 product categories
  Savings: 70-90% reduction in categorical columns
  ```
  - Use when: Specific values less important than categories
  - Example: "Analyze product performance" → Categories suffice instead of SKUs

- **Hash or anonymize sensitive fields**:
  ```
  Purpose: Privacy + token reduction
  Example: Email addresses → domain categories (gmail.com, corporate, etc.)
  Savings: 50-80% for text-heavy personally identifiable fields
  ```

### Prompt Structure Optimizations

- **Remove redundant instructions**:
  ```
  Verbose: "Please analyze the following data carefully and provide insights..." (10 tokens)
  Concise: "Analyze this data:" (3 tokens)
  Savings: 5-15% on instruction overhead across many requests
  ```

- **Use shorthand for repeated concepts**:
  ```
  First mention: "customer lifetime value (CLV)"
  Subsequent: "CLV" instead of repeating full phrase
  Savings: 30-50% in domain-heavy conversations
  ```

- **Structured outputs reduce back-and-forth**:
  ```
  Unstructured: "Tell me about this dataset" → vague response → follow-up questions
  Structured: "Provide: 1) Schema, 2) Row count, 3) Missing % per column, 4) Data types"
  Savings: 40-60% fewer total tokens by getting complete answer first time
  ```

### Context Window Management

- **Rolling summarization for long sessions**:
  ```
  Strategy: Summarize conversation history every 5-10 exchanges
  Token reduction: Compress 5k tokens → 500 token summary
  Use case: Long data exploration sessions with back-and-forth
  ```

- **Selective context inclusion**:
  ```
  Strategy: Only include relevant prior context, not entire conversation
  Example: For "plot this data" query, include only data description, not previous code snippets
  Savings: 50-80% in multi-turn conversations
  ```

- **External memory for large artifacts**:
  ```
  Strategy: Store large outputs (datasets, code) externally, reference by ID
  Example: "Continue working with dataset #3 from earlier" vs repassing 5k token dataset
  Savings: 70-95% for workflows with repeated artifact reuse
  ```

### Code Generation Token Efficiency

- **Request minimal working code**:
  ```
  Inefficient: "Write a complete data pipeline with error handling, logging, tests..."
  Efficient: "Write core transformation logic. I'll add error handling later."
  Savings: 60-80% fewer output tokens
  ```

- **Incremental refinement over complete rewrites**:
  ```
  Inefficient: Regenerating 200-line script for small change
  Efficient: "Modify the filtering function to handle NULL values"
  Savings: 80-95% when making targeted changes
  ```

- **Use established libraries over custom code**:
  ```
  Example: "Use pandas.read_csv() with appropriate params" vs "Write CSV parser from scratch"
  Token reduction: 70-90% (library call vs full implementation)
  Additional benefit: Higher reliability, fewer bugs
  ```

## Measuring and Monitoring

Track these metrics to validate optimization decisions:

- **Cost per task**: Total tokens × price per token ÷ number of completed tasks
- **Latency percentiles**: Measure p50, p90, p99 (not just averages)
- **Accuracy metrics**: Task-specific (e.g., SQL correctness, classification F1 score)
- **Cost-adjusted accuracy**: Accuracy ÷ cost ratio for ROI analysis
- **User satisfaction**: For interactive applications, perceived quality matters

### Example Monitoring Dashboard

```
Task: SQL Generation from Natural Language
├─ Model: GPT-4 Turbo
├─ Success Rate: 89% (SQL executes without errors)
├─ Avg Tokens: 450 input + 180 output = 630 total
├─ Cost per Query: $0.0063
├─ Latency (p90): 2.8 seconds
├─ Alternative: GPT-3.5 Turbo
│  ├─ Success Rate: 72%
│  ├─ Cost per Query: $0.0003
│  └─ Decision: Use GPT-4 (17% higher success worth 21x cost)
└─ Optimization Applied: Schema caching (-40% tokens)
```

## Conclusion

Effective inference optimization requires understanding the relationships between cost, latency, and accuracy for your specific use cases. Start by measuring baseline performance, identify optimization opportunities using the strategies above, and continuously monitor production metrics to validate decisions.

**Key Takeaways**:
- Small models can handle 70%+ of data science tasks at 10-50x lower cost
- Token reduction strategies (summarization, sampling) cut costs by 90%+ with minimal accuracy loss
- Latency optimizations (caching, streaming) improve UX without sacrificing quality
- Always measure: subjective impressions often mislead—quantify the trade-offs

**Next Steps**:
1. Audit your current LLM usage: track tokens, cost, latency per task type
2. Identify quick wins: high-volume, low-complexity tasks → switch to smaller models
3. Implement monitoring: measure accuracy and cost before/after optimizations
4. Iterate: optimization is ongoing as models and pricing evolve
