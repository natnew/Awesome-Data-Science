# Model-Specific Inference Design Patterns

This directory contains rigorously tested prompting and inference design patterns for major frontier and open-weight models. Each document provides practical guidance for data scientists and AI engineers working with specific LLM models.

## Purpose

Contributors document behavioral tendencies, reasoning characteristics, failure modes, and optimization strategies for specific models. All submissions should include:

- **Empirical observations** from real-world testing
- **Cost-performance considerations** for production use
- **Example task templates** grounded in data science workflows
- **Reproducible code examples** demonstrating patterns

## Available Patterns

- [Anthropic Claude 3 Sonnet](anthropic-claude-3-sonnet.md) - Comprehensive guide to Claude 3 Sonnet's capabilities and best practices

## Contributing a New Pattern

1. **Copy the template:** Use [`_TEMPLATE.md`](_TEMPLATE.md) as your starting point
2. **Fill in all sections:** Ensure you include empirical evidence and code examples
3. **Test your recommendations:** Verify all code examples run correctly
4. **Submit a pull request:** Follow the [Contributing Guidelines](../../Contributing.md)

### Required Sections

Each pattern document must include:

1. **Model Family** - Basic model information (provider, version, pricing, capabilities)
2. **Key Behavioral Tendencies** - Observable patterns in model responses
3. **Reasoning Characteristics** - Performance on different cognitive tasks
4. **Known Failure Modes** - Reproducible failures with examples and mitigations
5. **Optimization Strategies** - Cost and performance optimization techniques
6. **Real-World Workflows** - Example implementations for common tasks
7. **Empirical Observations** - Quantitative metrics from testing
8. **References** - Links to documentation, papers, and resources

### Quality Standards

- **Code examples must be runnable** - Include necessary imports and setup
- **Observations must be empirical** - Back claims with data from testing
- **Prompts should be realistic** - Use actual data science scenarios
- **Costs should be current** - Include dates for pricing information
- **Examples should be diverse** - Cover multiple use cases

## Models to Document

Priority models for documentation:

### Proprietary Models
- OpenAI GPT-4o, GPT-4 Turbo, o1
- Anthropic Claude 3.5 Sonnet, Claude 3 Opus
- Google Gemini 1.5 Pro, Gemini 2.0 Flash
- Alibaba Qwen 2.5
- DeepSeek V3

### Open-Weight Models
- Meta Llama 3.1 (8B, 70B, 405B)
- Mistral Large, Mixtral 8x22B
- Microsoft Phi-4
- Databricks DBRX
- Cohere Command R+

## Document Naming Convention

Use the following format for file names:
- `{provider}-{model-family}-{variant}.md`
- Examples: `openai-gpt-4o.md`, `meta-llama-3-1-70b.md`, `google-gemini-1-5-pro.md`

## Maintenance

Documents should be updated when:
- New model versions are released
- Pricing changes significantly
- New failure modes are discovered
- Better optimization strategies are found

Include version numbers and last-updated dates in all documents.

## Questions?

For questions about contributing inference patterns:
1. Check the [template file](_TEMPLATE.md) for guidance
2. Review the [Claude 3 Sonnet example](anthropic-claude-3-sonnet.md) as a reference
3. Open an issue in the main repository

---

**Last Updated:** 2026-02-18
