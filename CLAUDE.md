# Fine-Tuning Dataset Builder - Claude Code Navigation Guide

> **Purpose**: This file helps Claude Code navigate the codebase, understand the overall architecture, remember the project's goals, and track implementation progress.

---

## 🎯 Project Mission

**Build an intelligent agentic system that generates high-quality fine-tuning datasets for multiple LLM providers while respecting each provider's unique format requirements and token limitations.**

### Core Value Proposition
- **Saves Time**: Automates hours of manual dataset creation
- **Multi-Provider**: Supports OpenAI, Anthropic, Google, Mistral, Meta Llama
- **Smart Splitting**: Intelligent train/validation splits with weighted distribution
- **Quality Focused**: AI-powered generation + extraction from real documents
- **Free-Tier First**: Works entirely on free tiers (Gemini + SQLite)

---

## 📁 Project Structure

```
fine-tune-dataset-builder/
│
├── 🔧 Configuration Files
│   ├── .env                      # API keys (NEVER commit)
│   ├── .env.example              # Template for users
│   ├── config.yaml               # Dataset generation config
│   ├── requirements.txt          # Python dependencies
│   └── Makefile                  # Optional convenience commands
│
├── 📚 Documentation
│   ├── CLAUDE.md                 # This file - navigation guide
│   ├── instructions.txt          # Detailed implementation plan
│   ├── README.md                 # User-facing documentation
│   └── context_files/            # Reference materials
│       ├── recruiter_instructions.pdf
│       └── fine_tunable_llms_research.md
│
├── 📂 Data Directories
│   ├── input_files/              # User uploads source documents
│   ├── output_datasets/          # Generated JSONL + CSV exports
│   └── memory/                   # SQLite database storage
│
├── 🤖 Prompts
│   ├── prompts/system/           # Core agent system prompts
│   │   ├── create_mode.txt
│   │   ├── identify_mode.txt
│   │   └── hybrid_mode.txt
│   └── prompts/custom_mode/      # User-defined prompts
│       └── prompt.txt            # Optional override
│
└── 💻 Source Code
    └── src/
        ├── main.py               # CLI entry point
        ├── agents/               # AI agent implementations
        ├── providers/            # LLM provider integrations
        ├── utils/                # Helper utilities
        ├── persistence/          # SQLite database integration
        └── cli/                  # CLI interface
```

---

## 🧠 Key Concepts

### The Four Modes

1. **CREATE MODE** 🎨
   - Uses source files as "inspiration"
   - AI generates synthetic conversations
   - Most creative, highest variety
   - Best for: Creating diverse training data from limited examples

2. **IDENTIFY MODE** 🔍
   - Extracts existing conversations from documents
   - Preserves original text verbatim (no AI modification)
   - Uses pattern matching + AI to find Q&A exchanges
   - Best for: Mining real conversations from documentation, support logs, chat logs

3. **HYBRID MODE** 🔄
   - Combines both create and identify with dynamic fill
   - Extracts ALL available conversations first, then generates synthetic to reach target
   - Example: target=40, found 16 conversations → extract 16, generate 24 → total 40
   - Best for: Maximizing dataset size while preserving all real conversations

4. **CUSTOM MODE** 🎭
   - User defines custom behavior
   - Can load instructions from `prompts/custom_mode/prompt.txt`
   - Or prompt user interactively
   - Best for: Experimental workflows, specialized use cases

### Provider-Specific Formats

Each provider requires slightly different JSONL formatting:

**OpenAI:**
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Anthropic:**
```json
{"system": "...", "messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Google, Mistral, Llama:** Similar to OpenAI but may have variations

### Token Limits (From Research Doc)

| Provider | Model | Max Tokens/Example | Max Training Tokens |
|----------|-------|-------------------|---------------------|
| OpenAI | GPT-4o mini | 65,536 | 50M |
| OpenAI | GPT-4.1 | 65,536 | 50M |
| Anthropic | Claude 3 Haiku | 32,000 | Not specified |
| Google | Gemini 2.5 Flash | Varies | Not specified |
| Mistral | Mistral 7B | Varies | Flexible |
| Meta | Llama 3.1 | Flexible | Flexible |

---

## 🏗️ Implementation Status

### ✅ Completed Components
- [x] requirements.txt
- [x] instructions.txt
- [x] CLAUDE.md (this file)

### 🚧 In Progress
- [ ] Directory structure setup
- [ ] Base configuration system
- [ ] Provider integrations

### ⏳ Not Started
- [ ] Agent implementations
- [ ] File processing utilities
- [ ] Supabase integration
- [ ] CLI interface
- [ ] Validation system
- [ ] Splitting logic

---

## 🔑 Critical Implementation Details

### Conversation Detection (identify mode)
```python
# Multi-stage detection strategy:
# 1. Pattern matching (fast, deterministic)
PATTERNS = [
    r'Q:\s*.+?\n+A:\s*',           # Q&A
    r'User:\s*.+?\n+Assistant:\s*', # Chat
    r'\*\*Question\*\*:.*\n+\*\*Answer\*\*:' # Markdown
]

# 2. AI-powered (for ambiguous cases)
# Use agentic LLM to identify conversational structure
# Extract verbatim WITHOUT modification
```

### Token Counting (all modes)
```python
# Different providers = different tokenizers
def count_tokens(text, provider, model):
    if provider == "openai":
        return tiktoken.encoding_for_model(model).encode(text)
    elif provider == "anthropic":
        return len(text) // 4  # ~4 chars per token
    elif provider == "google":
        return gemini_api.count_tokens(text)
    # etc.
```

### Weighted Splitting (validation)
```python
# Equal representation per source or topic
def weighted_split(samples, ratio=0.8, weight_by="source"):
    groups = group_by(samples, key=weight_by)
    train, val = [], []
    
    for group in groups:
        split_idx = int(len(group) * ratio)
        train.extend(group[:split_idx])
        val.extend(group[split_idx:])
    
    return train, val
```

---

## 🎯 Development Priorities

### Phase 1: Foundation (Start Here)
1. Set up directory structure
2. Create `.env.example` with all provider templates
3. Create `config.yaml` with sensible defaults
4. Implement file processing (`utils/file_processor.py`)
5. Implement token counting (`utils/token_counter.py`)

### Phase 2: Provider Integration
1. Create `providers/base_provider.py` abstract class
2. Implement `providers/google_provider.py` (FREE TIER - start here!)
3. Implement other providers (OpenAI, Anthropic, Mistral, Llama)
4. Hardcode limits from research doc

### Phase 3: Agent System
1. Create `agents/base_agent.py`
2. Implement `agents/create_agent.py` (simplest)
3. Implement `agents/identify_agent.py` (needs conversation_detector)
4. Implement `agents/hybrid_agent.py`
5. Implement `agents/free_agent.py`

### Phase 4: CLI & Integration
1. Build menu system (`cli/menu.py`)
2. Create main orchestrator (`main.py`)
3. Add validation (`utils/validator.py`)
4. Add splitting (`utils/splitter.py`)

### Phase 5: Persistence
1. Set up SQLite tables
2. Implement `persistence/sqlite_client.py`
3. Add CSV export
4. Track metadata

---

## 🐛 Common Pitfalls to Avoid

### 1. Token Counting Errors
- ❌ Using same tokenizer for all providers
- ✅ Provider-specific tokenization

### 2. File Processing
- ❌ Loading entire file into memory
- ✅ Stream large files, process in chunks

### 3. Conversation Detection
- ❌ Using only regex patterns
- ✅ Regex first, AI fallback for ambiguous cases

### 4. Dataset Validation
- ❌ Generating dataset then discovering it exceeds limits
- ✅ Validate BEFORE generation, cap at provider max

### 5. Error Handling
- ❌ Silent failures
- ✅ Clear, actionable error messages with recovery suggestions

---

## 🔍 How to Navigate This Codebase

### When Adding a New Feature
1. Check `instructions.txt` for detailed implementation plan
2. Find relevant section in this file for context
3. Review `context_files/` for provider requirements
4. Implement with error handling
5. Update this file's "Implementation Status"

### When Debugging
1. Check config.yaml for correct settings
2. Verify .env has required API keys
3. Check logs for specific error messages
4. Review provider documentation in `context_files/`

### When Extending
1. Follow existing patterns (base classes, abstractions)
2. Update config.yaml schema if needed
3. Add tests for new functionality
4. Update README.md with usage examples

---

## 📊 File Metadata Schema

Every generated sample should track:
```python
{
    "content": "The actual conversation/example",
    "token_length": 1234,
    "topic_keywords": ["sales", "negotiation"],
    "source_file": "input_files/sales_guide.pdf",
    "split": "train" | "validation",
    "mode": "create" | "identify" | "hybrid" | "free"
}
```

Exported as CSV alongside JSONL dataset.

---

## 🚀 Quick Reference Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the system
python src/main.py

# Or use Make for convenience
make run

# Run tests (future)
pytest tests/
```

**Tip**: For faster dependency installation, consider using `uv`:
```bash
pip install uv
uv pip install -r requirements.txt  # 10-100x faster!
```

---

## 💡 Design Principles

1. **Free-Tier First**: Default to Gemini (free) unless user opts for paid
2. **Fail Gracefully**: Never crash without clear error message
3. **User Empowerment**: Give users control (config.yaml, .env, prompts)
4. **Quality > Quantity**: 100 great examples > 1000 mediocre ones
5. **Provider Agnostic**: Support all 5 providers equally well
6. **Future-Proof**: Design for web UI expansion

---

## 🧪 Testing

### Test Structure
```
tests/
├── unit/              # Unit tests for all components
│   ├── test_config_loader.py
│   ├── test_token_counter.py
│   ├── test_file_processor.py
│   ├── test_conversation_detector.py
│   ├── test_validator.py
│   ├── test_formatter.py
│   ├── test_splitter.py
│   ├── providers/     # Provider tests
│   │   ├── test_base_provider.py
│   │   ├── test_google_provider.py
│   │   ├── test_openai_provider.py
│   │   └── test_all_providers.py
│   ├── agents/        # Agent tests
│   │   ├── test_base_agent.py
│   │   ├── test_create_agent.py
│   │   ├── test_identify_agent.py
│   │   ├── test_hybrid_agent.py
│   │   └── test_free_agent.py
│   └── persistence/   # Database tests
│       ├── test_sqlite_client.py
│       └── test_models.py
├── integration/       # End-to-end workflow tests
│   ├── test_create_mode_workflow.py
│   ├── test_identify_mode_workflow.py
│   ├── test_hybrid_mode_workflow.py
│   └── test_full_pipeline.py
├── test_data/         # Sample files for testing
└── conftest.py        # Pytest fixtures
```

### Running Tests
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_config_loader.py -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests for specific module
pytest tests/unit/providers/ -v
```

### Test Coverage Goals
- **Config & Utilities**: 90%+ (high business logic)
- **Providers**: 80%+
- **Agents**: 75%+ (heavy LLM interaction, mocked)
- **Persistence**: 85%+
- **CLI**: 60%+ (manual testing preferred)

**Overall target: 80%+ coverage**

### Testing Strategy

**Unit Tests:**
- All LLM API calls are mocked to avoid costs
- In-memory SQLite (`:memory:`) for fast database tests
- Temporary directories (`pytest.tmp_path`) for file tests
- Comprehensive fixtures in `conftest.py`

**Integration Tests:**
- End-to-end workflows with mocked LLM calls
- Test complete pipeline from config to export
- Validate JSONL output format
- Verify database persistence

**Manual Testing:**
- Real API calls with Google Gemini (free tier)
- Test all 4 modes (CREATE, IDENTIFY, HYBRID, FREE)
- Test all 5 providers (with API keys)
- Edge cases (large files, rate limits, errors)

**Estimated Test Coverage: 200+ test cases**

---

## 📞 Need Help?

- **Implementation questions**: Check `instructions.txt`
- **Provider requirements**: Check `context_files/fine_tunable_llms_research.md`
- **Recruiter guidelines**: Check `context_files/recruiter_instructions.pdf`
- **Architecture decisions**: This file (CLAUDE.md)

---

## 🎓 Learning Resources

- [OpenAI Fine-tuning Docs](https://platform.openai.com/docs/guides/fine-tuning)
- [Anthropic Claude on Bedrock](https://docs.aws.amazon.com/bedrock/)
- [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs)
- [Mistral Docs](https://docs.mistral.ai/capabilities/finetuning)
- [Meta Llama Docs](https://llama.meta.com/docs)

---

## ✅ Pre-Implementation Checklist

Before starting implementation, ensure:
- [ ] Read `instructions.txt` fully
- [ ] Reviewed `fine_tunable_llms_research.md`
- [ ] Understood all 4 modes
- [ ] Know which provider to start with (Google Gemini - free!)
- [ ] Clear on directory structure
- [ ] .env.example created with all providers
- [ ] config.yaml schema defined

---

## 🔄 Version History

- **v0.1** (Current): Initial project setup, documentation created
- **v0.2** (Next): Directory structure + file processing
- **v0.3** (Future): First working mode (create with Gemini)
- **v1.0** (MVP): All modes, all providers, CLI working

---

**Remember**: This is an agentic system. The agents should be smart, but the orchestration (coordination) is what makes it truly intelligent. Focus on quality coordination between components.

Good luck! 🚀
