# model_trainer.py
import chess
import chess.pgn
import numpy as np
import pickle
from collections import defaultdict, Counter
import os

class ChessModelTrainer:
    def __init__(self):
        self.move_patterns = defaultdict(list)
        self.position_evaluations = defaultdict(float)
        self.opening_moves = defaultdict(int)
        self.endgame_patterns = defaultdict(list)
        
    def process_pgn_file(self, pgn_file_path):
        """Process PGN file to extract training data"""
        games_processed = 0
        
        with open(pgn_file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                self.extract_game_features(game)
                games_processed += 1
                
                if games_processed % 100 == 0:
                    print(f"Processed {games_processed} games...")
        
        print(f"Total games processed: {games_processed}")
        return games_processed
    
    def extract_game_features(self, game):
        """Extract features from a single game"""
        board = game.board()
        move_count = 0
        
        # Get game result
        result = game.headers.get('Result', '*')
        result_score = self.parse_result(result)
        
        for move in game.mainline_moves():
            move_count += 1
            
            # Extract position features before move
            position_key = self.get_position_key(board)
            
            # Store opening moves (first 10 moves)
            if move_count <= 20:
                opening_key = board.fen().split()[0]  # Just piece positions
                self.opening_moves[f"{opening_key}:{move.uci()}"] += 1
            
            # Store move patterns with context
            self.move_patterns[position_key].append({
                'move': move.uci(),
                'move_number': move_count,
                'result': result_score,
                'evaluation': self.evaluate_move_quality(board, move)
            })
            
            board.push(move)
    
    def get_position_key(self, board):
        """Generate a key for the current position"""
        # Simplified position key based on material and basic structure
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces.append(f"{piece.symbol()}{chess.square_name(square)}")
        
        return f"{''.join(sorted(pieces))}_{board.turn}"
    
    def parse_result(self, result):
        """Parse game result to numerical score"""
        if result == '1-0':
            return 1.0  # White wins
        elif result == '0-1':
            return -1.0  # Black wins
        else:
            return 0.0  # Draw
    
    def evaluate_move_quality(self, board, move):
        """Simple move quality evaluation"""
        board_copy = board.copy()
        
        # Basic evaluation before move
        eval_before = self.simple_evaluate(board_copy)
        
        # Make move and evaluate
        board_copy.push(move)
        eval_after = self.simple_evaluate(board_copy)
        
        # Return evaluation change
        return eval_after - eval_before
    
    def simple_evaluate(self, board):
        """Simple position evaluation"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        return score
    
    def build_decision_tree(self):
        """Build decision tree from processed data"""
        decision_tree = {}
        
        for position_key, moves in self.move_patterns.items():
            if len(moves) < 3:  # Skip positions with too few examples
                continue
            
            # Calculate move probabilities based on results
            move_scores = defaultdict(list)
            for move_data in moves:
                move_scores[move_data['move']].append(
                    move_data['result'] + move_data['evaluation'] * 0.1
                )
            
            # Calculate average scores and probabilities
            move_probs = {}
            for move, scores in move_scores.items():
                avg_score = np.mean(scores)
                move_probs[move] = max(0.0, avg_score + 1.0) / 2.0  # Normalize to 0-1
            
            # Normalize probabilities
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            
            decision_tree[position_key] = move_probs
        
        return decision_tree
    
    def train_model(self, pgn_files, output_path):
        """Train model from multiple PGN files"""
        print("Starting model training...")
        
        total_games = 0
        for pgn_file in pgn_files:
            if os.path.exists(pgn_file):
                print(f"Processing {pgn_file}...")
                games = self.process_pgn_file(pgn_file)
                total_games += games
            else:
                print(f"Warning: PGN file {pgn_file} not found")
        
        print(f"Building decision tree from {total_games} games...")
        decision_tree = self.build_decision_tree()
        
        # Build opening book
        opening_book = self.build_opening_book()
        
        # Save model
        model_data = {
            'decision_tree': decision_tree,
            'opening_book': opening_book,
            'position_scores': dict(self.position_evaluations),
            'training_stats': {
                'total_games': total_games,
                'positions_analyzed': len(self.move_patterns),
                'opening_moves': len(self.opening_moves)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {output_path}")
        print(f"Training complete: {total_games} games, {len(decision_tree)} positions")
        
        return model_data
    
    def build_opening_book(self):
        """Build opening book from processed games"""
        opening_book = {}
        
        # Sort opening moves by frequency
        sorted_openings = sorted(self.opening_moves.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Keep top moves for each position
        for opening_key, count in sorted_openings:
            if ':' not in opening_key:
                continue
                
            position, move = opening_key.split(':')
            if position not in opening_book:
                opening_book[position] = []
            
            if len(opening_book[position]) < 5:  # Keep top 5 moves
                opening_book[position].append({
                    'move': move,
                    'frequency': count
                })
        
        return opening_book

# Training script
def train_chess_model(pgn_files, model_output_path):
    """Main training function"""
    trainer = ChessModelTrainer()
    return trainer.train_model(pgn_files, model_output_path)

if __name__ == "__main__":
    # Example usage
    pgn_files = [
        "games/masters_games.pgn",
        "games/grandmaster_games.pgn"
    ]
    
    model_path = "models/chess_model.pkl"
    train_chess_model(pgn_files, model_path)
