
# Chess AI Engine with GUI

A comprehensive chess AI system combining neural networks, decision trees, and Stockfish engine evaluation with an interactive GUI.

## Features

- **Multi-model AI**: Neural network + decision tree + Stockfish ensemble
- **Interactive GUI**: tkinter-based chess interface
- **Training System**: Process PGN files to train models
- **Adjustable Difficulty**: 10 levels from beginner to expert
- **Cross-Platform**: Windows/macOS/Linux support
- **Hardware Acceleration**: CUDA/MPS/CPU support

## Demo

https://github.com/user-attachments/assets/f2d55404-670a-462a-8c13-e93db73eae97


## Installation

### Prerequisites

```bash
# Python packages
pip install torch scikit-learn python-chess joblib numpy tkinter pillow

# Stockfish installation:
# macOS: brew install stockfish
# Linux: sudo apt-get install stockfish
# Windows: Download from stockfishchess.org
```

### Setup

```bash
git clone https://github.com/nk-gujjar/AI-Chessbot.git
cd AI-Chessbot
```

## Usage

```bash
# Start GUI
python chess_gui.py

# Train models
python model_trainer.py

```

## Project Structure

```
chess-ai-engine/
├── chess_gui.py              # Main GUI application
├── chess_engine.py           # AI engine core
├── model_trainer.py          # Model training
├── config.py                 # Configuration
├── games/                    # Training PGNs
├── models/                   # Saved models
└── README.md
```

## Configuration

Edit `config.py`:

```python
# Stockfish settings
STOCKFISH_PATH = "/usr/bin/stockfish"  # Update for your OS
STOCKFISH_DEPTH = 15

# AI settings
NEURAL_NET_ENABLED = True
DEVICE_PREFERENCE = "auto"  # "cpu", "cuda", or "mps"

# GUI settings
BOARD_SIZE = 600
PIECE_THEME = "default"
```

## Training Models

1. Add PGN files to `games/` directory
2. Configure training in `config.py`:

```python
TRAINING_CONFIG = {
    'max_games_per_file': 500,
    'epochs': 30,
    'batch_size': 128
}
```

3. Run training:

```bash
python model_trainer.py
```

## Troubleshooting

**Stockfish not found:**
- Verify path in `config.py`
- Install Stockfish for your OS

**GUI issues:**
- Install tkinter: `sudo apt-get install python3-tk` (Linux)
- Ensure Pillow is installed

**Performance problems:**
- Reduce Stockfish depth
- Set `DEVICE_PREFERENCE = "cpu"`

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## License

MIT License
