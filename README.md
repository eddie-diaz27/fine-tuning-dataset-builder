# Fine-Tuning Dataset Builder 🤖

> An intelligent agentic system for generating high-quality fine-tuning datasets for multiple LLM providers.

## 🎯 What Does This Do?

This tool helps you create fine-tuning datasets for popular LLM providers (OpenAI, Anthropic, Google, Mistral, Meta Llama) by:

1. **Extracting** real conversations from your documents
2. **Generating** synthetic high-quality training examples
3. **Formatting** data to match each provider's requirements
4. **Validating** against token limits and formatting rules
5. **Splitting** into training and validation sets intelligently

## ✨ Key Features

- 🎨 **4 Generation Modes**: Create, Identify, Hybrid, Custom
- 🔌 **5 LLM Providers**: OpenAI, Anthropic, Google, Mistral, Meta Llama
- 💰 **Zero Setup**: Works with free APIs (Gemini) + local SQLite database
- 🎯 **Smart Validation**: Ensures compliance with provider limits
- 📊 **Metadata Tracking**: CSV export with sample statistics
- 🔀 **Intelligent Splitting**: Weighted train/validation distribution
- 📁 **Multi-Format Support**: PDF, DOCX, TXT, CSV, XLSX, PPTX

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fine-tune-dataset-builder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Use uv for 10-100x faster installation
# pip install uv
# uv pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
# That's all you need for the free tier!
```

### 3. Run

```bash
# Activate venv (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
python src/main.py

# Or use Make for convenience
make run
```

## 📖 Usage Guide

### Generation Modes

#### 🎨 CREATE MODE
Uses your documents as inspiration to generate synthetic conversations.

**Best for:**
- Creating diverse training data from limited examples
- Generating variations on themes
- Building large datasets quickly

**Example:**
```
Input: Technical documentation about Python
Output: 100 Q&A pairs about Python concepts
```

#### 🔍 IDENTIFY MODE
Extracts existing conversations from your documents verbatim.

**Best for:**
- Mining real conversations from support logs
- Preserving authentic dialogue
- Using existing Q&A documentation

**Example:**
```
Input: Customer support chat logs
Output: Real customer-agent conversations
```

#### 🔄 HYBRID MODE
Combines extraction and generation for best results.

**Best for:**
- Maximizing dataset size and quality
- Balancing real and synthetic data
- Most use cases

**Example:**
```
Input: Sales training materials
Output: 50% real examples + 50% generated variations
```

#### 🎭 CUSTOM MODE
Custom behavior defined by you.

**Best for:**
- Experimental workflows
- Specialized use cases
- Advanced users

**Example:**
```
Input: Your custom prompt in prompts/custom_mode/prompt.txt
Output: Whatever you define!
```

### Configuration (`config.yaml`)

```yaml
# Choose your provider and model
provider: "google"
model: "gemini-2.5-flash"

# Select generation mode
mode: "create"

# Set dataset size
dataset:
  size: 100
  output_name: "my_dataset"

# Configure train/validation split
split:
  enabled: true
  ratio: 0.8  # 80% train, 20% validation
  strategy: "weighted"
  weight_by: "source"
```

### Input Files

Place your source documents in `input_files/`:

```
input_files/
├── sales_guide.pdf
├── customer_faqs.docx
├── support_chats.txt
└── training_data.csv
```

Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX, MD, HTML

### Output

The system generates:

1. **JSONL Dataset**: `output_datasets/{name}.jsonl`
   - Formatted for your chosen provider
   - Validated against token limits
   - Ready for fine-tuning

2. **Metadata CSV**: `output_datasets/{name}_metadata.csv`
   - Sample previews
   - Token lengths
   - Topic keywords
   - Source files
   - Train/validation split

3. **Statistics**: `output_datasets/{name}_stats.json`
   - Total samples
   - Average token length
   - Topic distribution
   - Source breakdown

## 🔧 Advanced Features

### Make Commands (Optional)

For convenience, the project includes a Makefile:

```bash
make run      # Run the application (auto-activates venv)
make clean    # Remove venv and cache files
```

### Weighted Splitting

Ensure balanced representation in your training and validation sets:

```yaml
split:
  strategy: "weighted"
  weight_by: "source"  # or "keywords"
```

**Example:**
If you have 100 samples from 4 different source files, weighted splitting ensures each file contributes equally to both train and validation sets.

### Custom Prompts (Custom Mode)

Create `prompts/custom_mode/prompt.txt`:

```
Generate question-answer pairs about advanced Python concepts.
Focus on: decorators, metaclasses, async/await
Format: Technical but accessible
Tone: Educational, friendly
Length: 3-5 exchange turns
```

### Token Validation

The system automatically:
- Counts tokens per provider's tokenizer
- Warns if samples exceed limits
- Caps dataset at provider maximum
- Suggests alternatives if needed

## 📊 Provider Specifications

| Provider | Models | Max Tokens/Example | Max Training Tokens |
|----------|--------|-------------------|---------------------|
| OpenAI | GPT-4o, GPT-4.1 | 65,536 | 50M |
| Anthropic | Claude 3 Haiku | 32,000 | Not specified |
| Google | Gemini 2.5 Flash | Varies | Not specified |
| Mistral | Mistral 7B, Large v2 | Varies | Flexible |
| Meta | Llama 3.1 | Flexible | Flexible |

## 🆓 Free Tier Setup

### Recommended Configuration

1. **Google Gemini** (Agent System + Dataset Generation)
   - Free tier: 15 requests/minute
   - Get API key: https://aistudio.google.com/app/apikey

2. **SQLite** (Database)
   - Built into Python - zero setup!
   - Automatic database creation on first run

3. **Configuration**:
```bash
# .env (only one API key needed!)
AGENT_PROVIDER=google
AGENT_MODEL=gemini-2.0-flash-exp
GOOGLE_API_KEY=your-key-here
```

This gives you unlimited dataset generation at zero cost with zero external dependencies!

## 🐛 Troubleshooting

### "API key not found"
- Check your `.env` file exists (not `.env.example`)
- Ensure no extra spaces around `=` signs
- Verify key starts with correct prefix (e.g., `sk-` for OpenAI)

### "Token limit exceeded"
- Reduce `dataset.size` in `config.yaml`
- Choose a model with higher token limit
- Split your dataset into smaller batches

### "No conversations found"
- Try CREATE mode instead of IDENTIFY mode
- Lower `agent.identify.confidence_threshold` in `config.yaml`
- Ensure your documents actually contain Q&A patterns

### "Database error"
- Check that `memory/` directory exists
- Ensure write permissions in project directory
- Try deleting `memory/dataset_builder.db` to reset database

## 📚 Documentation

- **For Developers**: See `CLAUDE.md` for codebase navigation
- **For Implementation**: See `instructions.txt` for detailed plan
- **For Research**: See `context_files/fine_tunable_llms_research.md`

## 🛣️ Roadmap

### Current (v1.0 - MVP)
- ✅ CLI interface
- ✅ 4 generation modes
- ✅ 5 provider support
- ✅ Free tier operation
- ✅ Smart validation

### Planned (v2.0)
- 🔜 Web UI
- 🔜 Dataset editing tools
- 🔜 Quality scoring
- 🔜 Deduplication
- 🔜 Topic clustering

### Future (v3.0+)
- 💭 Active learning loops
- 💭 Multi-provider ensemble
- 💭 Integration with fine-tuning APIs
- 💭 Cost optimization tools

## 🤝 Contributing

This is a portfolio project for a job application. Feel free to:
- Report issues
- Suggest features
- Submit pull requests

## 📄 License

MIT License - Feel free to use this for your own projects!

## 🙏 Acknowledgments

- Built as part of UpSmith's agentic challenge
- Inspired by the need for better fine-tuning data tools
- Powered by free-tier AI services

## 📧 Contact

Questions? Reach out to the repository owner or open an issue!

---

**Happy Fine-Tuning! 🚀**
