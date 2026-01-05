.PHONY: help run clean

# Default target
help:
	@echo "Fine-Tuning Dataset Builder - Available Commands"
	@echo ""
	@echo "  make run      - Run the application"
	@echo "  make clean    - Remove venv and cache files"
	@echo ""

# Run the application
run:
	@if [ ! -d "venv" ]; then \
		echo "Error: Virtual environment not found."; \
		echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"; \
		exit 1; \
	fi
	@echo "Starting Fine-Tuning Dataset Builder..."
	@. venv/bin/activate && python src/main.py

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf venv/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"
