# train_model.py
#!/usr/bin/env python3
"""
Advanced model training script for chess bot
"""

import argparse
import os
import sys
from model_trainer import ChessModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train chess bot model')
    parser.add_argument('--pgn-dir', default='games', 
                       help='Directory containing PGN files')
    parser.add_argument('--output', default='models/chess_model.pkl',
                       help='Output model file path')
    parser.add_argument('--games-limit', type=int, default=None,
                       help='Limit number of games to process')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Find all PGN files
    pgn_files = []
    if os.path.exists(args.pgn_dir):
        for file in os.listdir(args.pgn_dir):
            if file.endswith('.pgn'):
                pgn_files.append(os.path.join(args.pgn_dir, file))
    
    if not pgn_files:
        print(f"No PGN files found in {args.pgn_dir}")
        print("Please add PGN files to train the model.")
        return
    
    print(f"Found {len(pgn_files)} PGN files:")
    for file in pgn_files:
        print(f"  - {file}")
    
    # Train model
    trainer = ChessModelTrainer()
    model_data = trainer.train_model(pgn_files, args.output)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.output}")
    print(f"Training statistics:")
    print(f"  - Games processed: {model_data['training_stats']['total_games']}")
    print(f"  - Positions analyzed: {model_data['training_stats']['positions_analyzed']}")
    print(f"  - Opening moves: {model_data['training_stats']['opening_moves']}")

if __name__ == "__main__":
    main()
