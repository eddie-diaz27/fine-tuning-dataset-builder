# FineDatasets - Quick Start Guide

## Prerequisites

✅ Virtual environment created and activated
✅ Dependencies installed (`pip install -r requirements.txt`)
✅ Google Gemini API key configured in `.env`
✅ Sales training documents in `input_files/`

## Running the Application

### 1. Start the Interactive CLI

```bash
cd /Users/eddiediaz/Desktop/repos/FineDatasets
python src/main.py
```

### 2. Main Menu

You'll see 6 options:
```
=== Fine-Tuning Dataset Builder ===

1. Configure Dataset
2. Generate Dataset
3. View Dataset Statistics
4. Export Dataset
5. Settings
6. Exit
```

### 3. Configure Your Dataset (Option 1)

Follow the prompts to configure:

**Provider Selection:**
- `1` - Google Gemini (FREE - no credit card needed)
- `2` - OpenAI (requires API key + credits)
- `3` - Anthropic (Bedrock only)
- `4` - Mistral (requires API key)
- `5` - Meta Llama (requires API key)

**Recommended for first run:** Google Gemini (option 1)

**Model Selection** (Google Gemini models):
- `1` - gemini-2.5-pro
- `2` - gemini-2.5-flash
- `3` - gemini-2.5-flash-lite
- `4` - gemini-2.0-flash-exp (recommended)

**Generation Mode:**
- `1` - CREATE (generate synthetic conversations)
- `2` - IDENTIFY (extract conversations from documents)
- `3` - HYBRID (combine extraction + generation)
- `4` - FREE (custom prompts)

**Recommended for first run:** CREATE (option 1)

**Dataset Size:**
- Enter target number of samples (e.g., `50`)

**Generation Variety:**
- `1` - Low (temperature 0.3 - more consistent)
- `2` - Medium (temperature 0.7 - balanced)
- `3` - High (temperature 0.9 - more creative)

**Conversation Length:**
- `1` - Short (2-3 turns)
- `2` - Medium (4-6 turns)
- `3` - Long (7+ turns)

**Train/Validation Split:**
- Enable split? `y` or `n`
- If yes, choose ratio:
  - `1` - No split
  - `2` - 80/20 (recommended)
  - `3` - 70/30
  - `4` - 90/10
  - `5` - Custom

**Split Strategy** (if enabled):
- `1` - Random
- `2` - Weighted by Source
- `3` - Weighted by Keywords

**Dataset Name:**
- Enter custom name (e.g., `sales_demo`)
- Or press Enter for auto-generated name: `dataset_YYYYMMDD_HHMMSS`

### 4. Generate Dataset (Option 2)

After configuration, select option `2` to generate.

**What happens:**
```
[1/8] Loading configuration...
[2/8] Processing input files...
[3/8] Initializing components...
[4/8] Generating samples...
[5/8] Validating dataset...
[6/8] Splitting dataset...
[7/8] Saving to database...
[8/8] Exporting files...
```

### 5. Check Your Output

Your generated dataset will be in:
```
output_datasets/
└── {dataset_name}/
    ├── train.jsonl           # Training samples
    ├── validation.jsonl      # Validation samples (if split enabled)
    └── metadata.csv          # Sample metadata
```

**View your dataset:**
```bash
# List generated datasets
ls -la output_datasets/

# View your dataset files
ls -la output_datasets/sales_demo/

# Preview train.jsonl (pretty print first sample)
head -n 1 output_datasets/sales_demo/train.jsonl | python -m json.tool

# Count samples
wc -l output_datasets/sales_demo/train.jsonl
wc -l output_datasets/sales_demo/validation.jsonl
```

---

## Quick Test Scenarios

### Scenario 1: CREATE Mode (Synthetic Generation)
```
Menu: 1 (Configure Dataset)
  Provider: 1 (Google)
  Model: 4 (gemini-2.0-flash-exp)
  Mode: 1 (CREATE)
  Size: 50
  Variety: 2 (Medium)
  Length: 2 (Medium)
  Split: y → 2 (80/20) → 1 (Random)
  Name: create_test

Menu: 2 (Generate Dataset)
```

**Expected:** 50 synthetic sales conversations generated from methodology documents

---

### Scenario 2: IDENTIFY Mode (Extraction)
```
Menu: 1 (Configure Dataset)
  Provider: 1 (Google)
  Model: 4 (gemini-2.0-flash-exp)
  Mode: 2 (IDENTIFY)
  Size: 20 (will extract ~13 available)
  Min turns: 2
  Include context: y
  Confidence: 0.8
  Split: y → 2 (80/20) → 1 (Random)
  Name: identify_test

Menu: 2 (Generate Dataset)
```

**Expected:** ~13 conversations extracted verbatim from sales transcripts

---

### Scenario 3: HYBRID Mode (Combined)
```
Menu: 1 (Configure Dataset)
  Provider: 1 (Google)
  Model: 4 (gemini-2.0-flash-exp)
  Mode: 3 (HYBRID)
  Size: 100
  Generation ratio: 0.5 (50/50)
  Variety: 2 (Medium)
  Length: 2 (Medium)
  Deduplicate: y
  Split: y → 2 (80/20) → 2 (Weighted by Source)
  Name: hybrid_test

Menu: 2 (Generate Dataset)
```

**Expected:** ~50 extracted + ~50 generated = 100 total samples

---

## Troubleshooting

### API Key Error
```
Error: GOOGLE_API_KEY not found
```
**Fix:** Add to `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
AGENT_PROVIDER=google
AGENT_MODEL=gemini-2.0-flash-exp
```

### No Input Files Found
```
Error: No files found in input_files/
```
**Fix:** Ensure sales documents are in `input_files/`:
- sales_identify_conversations.txt
- sales_identify_chatlogs.txt
- sales_create_methodology.txt
- sales_create_playbooks.txt

### Module Import Error
```
ModuleNotFoundError: No module named 'google.generativeai'
```
**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Database & Statistics

### View Dataset Statistics (Menu Option 3)

After generating datasets, select option `3` to view:
- Total datasets created
- Sample counts per dataset
- Token usage statistics
- Source file breakdown

### Inspect Database (Optional)

```bash
sqlite3 memory/dataset_builder.db

.tables                          # Show tables
SELECT * FROM datasets;          # View datasets
SELECT COUNT(*) FROM samples;    # Count all samples
.quit
```

---

## Next Steps

1. **Run your first dataset:** Follow Scenario 1 (CREATE mode)
2. **Inspect output:** Check `output_datasets/{your_name}/`
3. **Validate format:** Ensure JSONL is valid JSON (one object per line)
4. **Try other modes:** Test IDENTIFY and HYBRID
5. **Test different providers:** OpenAI, Anthropic (if you have API keys)
6. **Adjust parameters:** Experiment with variety, length, split strategies

---

## Command Reference

**Start application:**
```bash
python src/main.py
```

**Run unit tests:**
```bash
pytest tests/unit/ -v
```

**Check environment:**
```bash
cat .env | grep GOOGLE_API_KEY
ls input_files/
```

**View outputs:**
```bash
ls -la output_datasets/
```

---

**Ready to generate your first fine-tuning dataset!** 🚀
