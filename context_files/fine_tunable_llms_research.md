# Best Fine-Tunable LLMs: Data Formats and Limits (2025)

## Executive Summary

This research document covers the top 3-5 fine-tunable large language models currently available on the market, including their data format requirements and limitations for fine-tuning jobs.

---

## 1. OpenAI GPT Models

### Available Models
- **GPT-4o** and **GPT-4o mini** (recommended for most use cases)
- **GPT-4.1** and **GPT-4.1 mini**
- **GPT-3.5-turbo** (legacy, still supported)

### Data Format
**Format:** JSONL (JSON Lines) - each line is a separate JSON object

**Structure:**
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]}
```

### Data Limits

#### Token Limits Per Example:
- **GPT-4o mini**: 128,000 tokens (inference), 65,536 tokens (training)
- **GPT-4.1 mini**: 65,536 tokens per training example
- **GPT-3.5-turbo-0125**: 16,385 tokens per example
- **GPT-3.5-turbo-0613**: 4,096 tokens per example

#### Overall Limits:
- **Maximum training tokens**: 50 million tokens per fine-tuning job (approximately $400 at standard pricing)
- **Minimum examples**: 10 examples (but 50-100 recommended for quality results)
- **Daily training token limit**: 2 million tokens for promotional periods (varies)

### Best Practices:
- Start with at least 50-100 high-quality examples
- Quality over quantity - doubling dataset size typically leads to linear quality improvements
- Examples longer than the limit are truncated from the end
- Use the tiktoken library to count tokens before uploading

---

## 2. Anthropic Claude

### Available Models
- **Claude 3 Haiku** (only Claude model available for fine-tuning via Amazon Bedrock)

### Data Format
**Format:** JSONL (JSON Lines)

**Structure:**
```json
{"system": "<system message>","messages":[
  {"role": "user", "content": "<user query>"},
  {"role": "assistant", "content": "<expected generated text>"}
]}
```

### Data Limits

#### Context and Tokens:
- **Maximum context length**: 32,000 tokens (32K)
- Currently supports text-based fine-tuning only (vision capabilities planned for future)

#### Dataset Requirements:
- No specific minimum mentioned, but quality is paramount
- Start with high-quality prompt-completion pairs
- Focus on comprehensive datasets representing full spectrum of use cases

#### Regional Availability:
- **Available regions**: US West (Oregon) AWS Region only
- Accessed exclusively through Amazon Bedrock
- Not available through Anthropic's native API

### Best Practices:
- Prioritize well-crafted examples over large volumes of mediocre data
- Include industry-specific terminology and contextual nuances
- Plan for multiple rounds of fine-tuning based on real-world feedback
- Consider using larger Claude models (3.5 Sonnet, Opus) to generate or refine training data

### Performance Notes:
- Fine-tuned Claude 3 Haiku can outperform base Claude 3.5 Sonnet on specific tasks
- Typical improvements: 24.6% F1 score increase on specialized tasks

---

## 3. Google Gemini

### Available Models (via Vertex AI)
- **Gemini 2.5 Pro**
- **Gemini 2.5 Flash**
- **Gemini 2.5 Flash-lite**

**Note:** Gemini 3 fine-tuning is currently limited to enterprise customers. Public Gemini API deprecated fine-tuning support in May 2025.

### Data Format
**Format:** JSONL (JSON Lines)

**Structure:** Multiple formats supported depending on data type
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### Data Limits

#### Training Data:
- **Recommended starting point**: 100 examples minimum
- Can scale to thousands of examples if needed
- **Maximum training dataset file size**: Varies by model (see Vertex AI documentation)

#### Token Limits:
- Varies by specific model version
- Support for text, image, audio, video, and document data types

#### Validation Dataset:
- Strongly recommended to provide validation dataset
- Helps measure effectiveness of tuning job

### Best Practices:
- Quality far more important than quantity
- Ensure prompts in fine-tuning dataset closely resemble production use
- For models with thinking capabilities, set thinking budget to off or lowest value
- Start with smaller models (Flash) for testing before moving to larger models (Pro)

### Important Notes:
- Supervised fine-tuning is NOT a covered service under SLA
- Quota enforced on number of concurrent tuning jobs (default: 1 per project)
- Global quota shared across all regions and models

---

## 4. Mistral AI

### Available Models
- **Mistral 7B** (Instruct variants)
- **Mistral Large v2** (123B parameters)
- **Mistral Nemo** (12B parameters)
- **Mixtral 8x7B** (MoE model)

### Data Format
**Format:** JSONL (JSON Lines)

**Instruct Format:**
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "system", "content": "..."}
]}
```

**Function Calling Format:**
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "tool_calls": [...]}
]}
```

**Pretrain Format:**
```json
{"text": "Plain text content for pretraining"}
```

### Data Limits

#### General Requirements:
- All data files must be in JSONL format
- Loss computed only for "role" == "assistant"
- Last message in conversation must be from assistant

#### Model-Specific Notes:
- **Mistral Large v2**: Requires significantly more memory (set seq_len to ≤ 8192)
- **Mistral Nemo**: Higher memory requirements due to larger vocabulary (Tekkenizer)
- Recommended learning rate for Large v2: 1e-6

### Best Practices:
- 50-100 examples minimum for favorable results
- Can scale to 1,000+ examples for complex tasks
- Use reformat_data.py utility to correct formatting issues
- Function calling requires specific format with tool definitions

---

## 5. Meta Llama (Open Source)

### Available Models
- **Llama 3.1** (8B, 70B, 405B parameters)
- **Llama 3** (multiple sizes)
- **Llama 2** (legacy)

### Data Format
**Format:** JSONL (JSON Lines)

**Standard Format:**
```json
{"text": "<s>[INST] {instruction} [/INST] {response} </s>"}
```

**Conversational Format:**
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### Data Limits

#### Flexibility:
- Open source allows for flexible implementation
- Limits depend on infrastructure and fine-tuning framework used
- Can be fine-tuned locally or on cloud platforms

#### Framework Recommendations:
- **Axolotl**: Good for those who want to avoid deep technical details
- **Unsloth**: 2-5x faster training with 80% less memory (optimized for consumer GPUs)
- **LLaMA-Factory**: Comprehensive framework supporting 100+ LLMs
- **Hugging Face Transformers**: Full control and customization

### Best Practices:
- Available via Hugging Face, AWS, Google Cloud, NVIDIA
- Supports various functions: inference, fine-tuning, continual pre-training, RAG, function calling
- No hard dataset limits, but quality over quantity principle applies
- 50-1000+ examples depending on task complexity

### Performance Optimization:
- Use QLoRA (4-bit quantization) for memory efficiency
- LoRA attention dimension (lora_r): typically 64
- LoRA alpha: typically 16
- LoRA dropout: typically 0.1

---

## Comparison Summary Table

| Provider | Model | Format | Min Examples | Token Limit/Example | Max Training Tokens | Special Notes |
|----------|-------|--------|--------------|---------------------|---------------------|---------------|
| OpenAI | GPT-4o mini | JSONL | 50-100 | 65,536 | 50M | Most cost-effective |
| OpenAI | GPT-4.1 | JSONL | 50-100 | 65,536 | 50M | Latest, most capable |
| Anthropic | Claude 3 Haiku | JSONL | High-quality | 32,000 | Not specified | Bedrock only |
| Google | Gemini 2.5 Flash | JSONL | 100+ | Varies | Not specified | Vertex AI only |
| Mistral | Mistral 7B | JSONL | 50-100 | Varies | Flexible | Open weights |
| Meta | Llama 3.1 | JSONL | 50-100 | Flexible | Flexible | Fully open source |

---

## General Best Practices Across All Models

### Dataset Quality
1. **Quality over quantity**: 100 high-quality examples beats 1,000 low-quality ones
2. **Consistency**: Maintain consistent formatting throughout
3. **Representative**: Include examples covering edge cases
4. **Alignment**: Training data should match production use cases

### Data Preparation
1. **Validation split**: Always create separate validation dataset (typically 10-20%)
2. **Token counting**: Verify examples fit within limits before uploading
3. **Format checking**: Use provided tools to validate JSONL format
4. **Balance**: Ensure diverse distribution of example types

### Training Strategy
1. **Start small**: Begin with 50-100 examples
2. **Iterate**: Use metrics to refine dataset
3. **Scale up**: Double dataset size for linear quality improvements
4. **Monitor**: Track metrics during training (loss, accuracy)

### Cost Optimization
1. **Fixed vs Variable**: Fine-tuning is fixed cost, inference is variable
2. **Model selection**: Use smallest model that meets requirements
3. **Token efficiency**: Reduce prompt tokens through fine-tuning
4. **Batch processing**: Use batch APIs when possible

---

## Key Takeaways

1. **JSONL is universal**: All major providers use JSONL format with slight variations
2. **Token limits vary widely**: From 4K (legacy) to 128K+ (modern models)
3. **No strict minimum**: But 50-100 quality examples recommended across providers
4. **Maximum limits**: OpenAI has explicit 50M token cap; others more flexible
5. **Quality critical**: All providers emphasize quality over quantity
6. **Validation important**: Separate validation data helps measure success

---

## Resources and Documentation

- **OpenAI**: https://platform.openai.com/docs/guides/fine-tuning
- **Anthropic/Bedrock**: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization.html
- **Google Vertex AI**: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning
- **Mistral**: https://docs.mistral.ai/capabilities/finetuning
- **Meta Llama**: https://llama.meta.com/docs

---

*Research compiled December 2024. Information subject to change as providers update their offerings.*
