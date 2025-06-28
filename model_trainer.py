# # # model_trainer.py
# # import chess
# # import chess.pgn
# # import numpy as np
# # import pickle
# # from collections import defaultdict, Counter
# # import os

# # class ChessModelTrainer:
# #     def __init__(self):
# #         self.move_patterns = defaultdict(list)
# #         self.position_evaluations = defaultdict(float)
# #         self.opening_moves = defaultdict(int)
# #         self.endgame_patterns = defaultdict(list)
        
# #     def process_pgn_file(self, pgn_file_path):
# #         """Process PGN file to extract training data"""
# #         games_processed = 0
        
# #         with open(pgn_file_path, 'r') as pgn_file:
# #             while True:
# #                 game = chess.pgn.read_game(pgn_file)
# #                 if game is None:
# #                     break
                
# #                 self.extract_game_features(game)
# #                 games_processed += 1
                
# #                 if games_processed % 100 == 0:
# #                     print(f"Processed {games_processed} games...")
        
# #         print(f"Total games processed: {games_processed}")
# #         return games_processed
    
# #     def extract_game_features(self, game):
# #         """Extract features from a single game"""
# #         board = game.board()
# #         move_count = 0
        
# #         # Get game result
# #         result = game.headers.get('Result', '*')
# #         result_score = self.parse_result(result)
        
# #         for move in game.mainline_moves():
# #             move_count += 1
            
# #             # Extract position features before move
# #             position_key = self.get_position_key(board)
            
# #             # Store opening moves (first 10 moves)
# #             if move_count <= 20:
# #                 opening_key = board.fen().split()[0]  # Just piece positions
# #                 self.opening_moves[f"{opening_key}:{move.uci()}"] += 1
            
# #             # Store move patterns with context
# #             self.move_patterns[position_key].append({
# #                 'move': move.uci(),
# #                 'move_number': move_count,
# #                 'result': result_score,
# #                 'evaluation': self.evaluate_move_quality(board, move)
# #             })
            
# #             board.push(move)
    
# #     def get_position_key(self, board):
# #         """Generate a key for the current position"""
# #         # Simplified position key based on material and basic structure
# #         pieces = []
# #         for square in chess.SQUARES:
# #             piece = board.piece_at(square)
# #             if piece:
# #                 pieces.append(f"{piece.symbol()}{chess.square_name(square)}")
        
# #         return f"{''.join(sorted(pieces))}_{board.turn}"
    
# #     def parse_result(self, result):
# #         """Parse game result to numerical score"""
# #         if result == '1-0':
# #             return 1.0  # White wins
# #         elif result == '0-1':
# #             return -1.0  # Black wins
# #         else:
# #             return 0.0  # Draw
    
# #     def evaluate_move_quality(self, board, move):
# #         """Simple move quality evaluation"""
# #         board_copy = board.copy()
        
# #         # Basic evaluation before move
# #         eval_before = self.simple_evaluate(board_copy)
        
# #         # Make move and evaluate
# #         board_copy.push(move)
# #         eval_after = self.simple_evaluate(board_copy)
        
# #         # Return evaluation change
# #         return eval_after - eval_before
    
# #     def simple_evaluate(self, board):
# #         """Simple position evaluation"""
# #         if board.is_checkmate():
# #             return -9999 if board.turn else 9999
        
# #         piece_values = {
# #             chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
# #             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
# #         }
        
# #         score = 0
# #         for square in chess.SQUARES:
# #             piece = board.piece_at(square)
# #             if piece:
# #                 value = piece_values[piece.piece_type]
# #                 score += value if piece.color == chess.WHITE else -value
        
# #         return score
    
# #     def build_decision_tree(self):
# #         """Build decision tree from processed data"""
# #         decision_tree = {}
        
# #         for position_key, moves in self.move_patterns.items():
# #             if len(moves) < 3:  # Skip positions with too few examples
# #                 continue
            
# #             # Calculate move probabilities based on results
# #             move_scores = defaultdict(list)
# #             for move_data in moves:
# #                 move_scores[move_data['move']].append(
# #                     move_data['result'] + move_data['evaluation'] * 0.1
# #                 )
            
# #             # Calculate average scores and probabilities
# #             move_probs = {}
# #             for move, scores in move_scores.items():
# #                 avg_score = np.mean(scores)
# #                 move_probs[move] = max(0.0, avg_score + 1.0) / 2.0  # Normalize to 0-1
            
# #             # Normalize probabilities
# #             total_prob = sum(move_probs.values())
# #             if total_prob > 0:
# #                 for move in move_probs:
# #                     move_probs[move] /= total_prob
            
# #             decision_tree[position_key] = move_probs
        
# #         return decision_tree
    
# #     def train_model(self, pgn_files, output_path):
# #         """Train model from multiple PGN files"""
# #         print("Starting model training...")
        
# #         total_games = 0
# #         for pgn_file in pgn_files:
# #             if os.path.exists(pgn_file):
# #                 print(f"Processing {pgn_file}...")
# #                 games = self.process_pgn_file(pgn_file)
# #                 total_games += games
# #             else:
# #                 print(f"Warning: PGN file {pgn_file} not found")
        
# #         print(f"Building decision tree from {total_games} games...")
# #         decision_tree = self.build_decision_tree()
        
# #         # Build opening book
# #         opening_book = self.build_opening_book()
        
# #         # Save model
# #         model_data = {
# #             'decision_tree': decision_tree,
# #             'opening_book': opening_book,
# #             'position_scores': dict(self.position_evaluations),
# #             'training_stats': {
# #                 'total_games': total_games,
# #                 'positions_analyzed': len(self.move_patterns),
# #                 'opening_moves': len(self.opening_moves)
# #             }
# #         }
        
# #         with open(output_path, 'wb') as f:
# #             pickle.dump(model_data, f)
        
# #         print(f"Model saved to {output_path}")
# #         print(f"Training complete: {total_games} games, {len(decision_tree)} positions")
        
# #         return model_data
    
# #     def build_opening_book(self):
# #         """Build opening book from processed games"""
# #         opening_book = {}
        
# #         # Sort opening moves by frequency
# #         sorted_openings = sorted(self.opening_moves.items(), 
# #                                 key=lambda x: x[1], reverse=True)
        
# #         # Keep top moves for each position
# #         for opening_key, count in sorted_openings:
# #             if ':' not in opening_key:
# #                 continue
                
# #             position, move = opening_key.split(':')
# #             if position not in opening_book:
# #                 opening_book[position] = []
            
# #             if len(opening_book[position]) < 5:  # Keep top 5 moves
# #                 opening_book[position].append({
# #                     'move': move,
# #                     'frequency': count
# #                 })
        
# #         return opening_book

# # # Training script
# # def train_chess_model(pgn_files, model_output_path):
# #     """Main training function"""
# #     trainer = ChessModelTrainer()
# #     return trainer.train_model(pgn_files, model_output_path)

# # if __name__ == "__main__":
# #     # Example usage
# #     pgn_files = [
# #         "games/masters_games.pgn",
# #         "games/grandmaster_games.pgn"
# #     ]
    
# #     model_path = "models/chess_model.pkl"
# #     train_chess_model(pgn_files, model_path)

# # enhanced_chess_trainer.py
# import chess
# import chess.pgn
# import chess.engine
# import numpy as np
# import pickle
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import defaultdict, Counter
# import os
# import random
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.tree import DecisionTreeClassifier
# import joblib
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ChessNeuralNetwork(nn.Module):
#     """Simple neural network for chess position evaluation"""
#     def __init__(self, input_size=773, hidden_size=512, output_size=4096):
#         super(ChessNeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc4 = nn.Linear(hidden_size // 2, output_size)
#         self.dropout = nn.Dropout(0.3)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.fc4(x)
#         return self.softmax(x)

# class EnhancedChessModelTrainer:
#     def __init__(self, stockfish_path=None, use_neural_net=True):
#         self.move_patterns = defaultdict(list)
#         self.position_evaluations = defaultdict(float)
#         self.opening_moves = defaultdict(int)
#         self.endgame_patterns = defaultdict(list)
#         self.training_data = []
#         self.validation_data = []
        
#         # Initialize Stockfish engine if path provided
#         self.engine = None
#         if stockfish_path and os.path.exists(stockfish_path):
#             try:
#                 self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
#                 logger.info("Stockfish engine initialized successfully")
#             except Exception as e:
#                 logger.warning(f"Could not initialize Stockfish: {e}")
        
#         # Initialize neural network
#         self.use_neural_net = use_neural_net
#         if use_neural_net:
#             # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             if torch.backends.mps.is_available():
#                 device = torch.device("mps")  # Apple Metal (M1/M2 GPU)
#             elif torch.cuda.is_available():
#                 device = torch.device("cuda")
#             else:
#                 device = torch.device("cpu")
#             self.device = device
#             self.neural_net = ChessNeuralNetwork().to(self.device)
#             self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
#             self.criterion = nn.CrossEntropyLoss()
#             logger.info(f"Neural network initialized on {self.device}")
        
#         # Decision tree for move selection
#         self.decision_tree = DecisionTreeClassifier(
#             max_depth=15,
#             min_samples_split=10,
#             min_samples_leaf=5,
#             random_state=42
#         )
        
#         # Accuracy tracking
#         self.accuracy_history = []
#         self.validation_accuracy = []
        
#     def board_to_features(self, board):
#         """Convert chess board to feature vector"""
#         features = []
        
#         # Piece positions (64 squares * 12 piece types = 768 features)
#         piece_map = {
#             chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
#             chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
#         }
        
#         for square in chess.SQUARES:
#             piece_features = [0] * 12  # 6 white pieces + 6 black pieces
#             piece = board.piece_at(square)
#             if piece:
#                 piece_idx = piece_map[piece.piece_type] - 1
#                 if piece.color == chess.WHITE:
#                     piece_features[piece_idx] = 1
#                 else:
#                     piece_features[piece_idx + 6] = 1
#             features.extend(piece_features)
        
#         # Additional features
#         features.append(1 if board.turn == chess.WHITE else 0)  # Turn
#         features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
#         features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
#         features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
#         features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
#         return np.array(features, dtype=np.float32)
    
#     def get_stockfish_evaluation(self, board, depth=15):
#         """Get position evaluation from Stockfish"""
#         if not self.engine:
#             return 0.0
        
#         try:
#             info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
#             score = info["score"].relative
            
#             if score.is_mate():
#                 return 10000 if score.mate() > 0 else -10000
#             else:
#                 return float(score.score()) / 100.0
#         except Exception as e:
#             logger.warning(f"Stockfish evaluation failed: {e}")
#             return 0.0
    
#     def get_stockfish_best_move(self, board, time_limit=0.1):
#         """Get best move from Stockfish"""
#         if not self.engine:
#             return None
        
#         try:
#             result = self.engine.play(board, chess.engine.Limit(time=time_limit))
#             return result.move
#         except Exception as e:
#             logger.warning(f"Stockfish move generation failed: {e}")
#             return None
    
#     def move_to_index(self, move):
#         """Convert move to index for neural network output"""
#         from_square = move.from_square
#         to_square = move.to_square
#         return from_square * 64 + to_square
    
#     def index_to_move(self, index, board):
#         """Convert index back to move"""
#         from_square = index // 64
#         to_square = index % 64
#         move = chess.Move(from_square, to_square)
        
#         # Handle promotions
#         if move in board.legal_moves:
#             return move
        
#         # Try with promotion
#         for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
#             promoted_move = chess.Move(from_square, to_square, promotion)
#             if promoted_move in board.legal_moves:
#                 return promoted_move
        
#         return None
    
#     def process_pgn_file(self, pgn_file_path, max_games=None):
#         """Process PGN file to extract training data"""
#         games_processed = 0
        
#         with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
#             while True:
#                 if max_games and games_processed >= max_games:
#                     break
                
#                 game = chess.pgn.read_game(pgn_file)
#                 if game is None:
#                     break
                
#                 try:
#                     self.extract_game_features(game)
#                     games_processed += 1
                    
#                     if games_processed % 100 == 0:
#                         logger.info(f"Processed {games_processed} games...")
#                 except Exception as e:
#                     logger.warning(f"Error processing game {games_processed}: {e}")
#                     continue
        
#         logger.info(f"Total games processed: {games_processed}")
#         return games_processed
    
#     def extract_game_features(self, game):
#         """Extract features from a single game"""
#         board = game.board()
#         move_count = 0
        
#         # Get game result and player ratings
#         result = game.headers.get('Result', '*')
#         white_elo = int(game.headers.get('WhiteElo', '1500'))
#         black_elo = int(game.headers.get('BlackElo', '1500'))
        
#         result_score = self.parse_result(result)
        
#         for move in game.mainline_moves():
#             if move_count > 100:  # Limit moves per game for training efficiency
#                 break
                
#             move_count += 1
            
#             # Extract features before move
#             features = self.board_to_features(board)
#             position_key = self.get_position_key(board)
            
#             # Get Stockfish evaluation and best move
#             stockfish_eval = self.get_stockfish_evaluation(board)
#             stockfish_best = self.get_stockfish_best_move(board)
            
#             # Store training data
#             move_index = self.move_to_index(move)
#             is_best_move = 1 if stockfish_best and move == stockfish_best else 0
            
#             training_sample = {
#                 'features': features,
#                 'move': move.uci(),
#                 'move_index': move_index,
#                 'position_key': position_key,
#                 'evaluation': stockfish_eval,
#                 'is_best_move': is_best_move,
#                 'move_number': move_count,
#                 'result': result_score,
#                 'white_elo': white_elo,
#                 'black_elo': black_elo,
#                 'player_to_move': 1 if board.turn == chess.WHITE else 0
#             }
            
#             self.training_data.append(training_sample)
            
#             # Store in move patterns for decision tree
#             self.move_patterns[position_key].append(training_sample)
            
#             # Store opening moves
#             if move_count <= 20:
#                 opening_key = board.fen().split()[0]
#                 self.opening_moves[f"{opening_key}:{move.uci()}"] += 1
            
#             board.push(move)
    
#     def get_position_key(self, board):
#         """Generate a simplified position key"""
#         # Use material signature and basic position info
#         material = []
#         for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
#             white_count = len(board.pieces(piece_type, chess.WHITE))
#             black_count = len(board.pieces(piece_type, chess.BLACK))
#             material.append(f"{piece_type}{white_count}{black_count}")
        
#         castling = ""
#         if board.has_kingside_castling_rights(chess.WHITE):
#             castling += "K"
#         if board.has_queenside_castling_rights(chess.WHITE):
#             castling += "Q"
#         if board.has_kingside_castling_rights(chess.BLACK):
#             castling += "k"
#         if board.has_queenside_castling_rights(chess.BLACK):
#             castling += "q"
        
#         return f"{''.join(material)}_{castling}_{board.turn}"
    
#     def parse_result(self, result):
#         """Parse game result to numerical score"""
#         if result == '1-0':
#             return 1.0
#         elif result == '0-1':
#             return -1.0
#         else:
#             return 0.0
    
#     def prepare_training_data(self, validation_split=0.2):
#         """Prepare data for training"""
#         logger.info(f"Preparing {len(self.training_data)} training samples...")
        
#         # Shuffle data
#         random.shuffle(self.training_data)
        
#         # Split into training and validation
#         split_idx = int(len(self.training_data) * (1 - validation_split))
#         train_data = self.training_data[:split_idx]
#         val_data = self.training_data[split_idx:]
        
#         # Prepare neural network data
#         if self.use_neural_net:
#             train_features = np.array([sample['features'] for sample in train_data])
#             train_targets = np.array([sample['move_index'] for sample in train_data])
            
#             val_features = np.array([sample['features'] for sample in val_data])
#             val_targets = np.array([sample['move_index'] for sample in val_data])
            
#             self.train_features = torch.FloatTensor(train_features).to(self.device)
#             self.train_targets = torch.LongTensor(train_targets).to(self.device)
#             self.val_features = torch.FloatTensor(val_features).to(self.device)
#             self.val_targets = torch.LongTensor(val_targets).to(self.device)
        
#         # Prepare decision tree data
#         dt_features = []
#         dt_targets = []
        
#         for sample in train_data:
#             # Create features for decision tree
#             dt_feature = [
#                 sample['evaluation'],
#                 sample['move_number'],
#                 sample['white_elo'],
#                 sample['black_elo'],
#                 sample['player_to_move'],
#                 sample['result']
#             ]
#             dt_features.append(dt_feature)
#             dt_targets.append(sample['is_best_move'])
        
#         self.dt_train_features = np.array(dt_features)
#         self.dt_train_targets = np.array(dt_targets)
        
#         # Validation data for decision tree
#         dt_val_features = []
#         dt_val_targets = []
        
#         for sample in val_data:
#             dt_feature = [
#                 sample['evaluation'],
#                 sample['move_number'],
#                 sample['white_elo'],
#                 sample['black_elo'],
#                 sample['player_to_move'],
#                 sample['result']
#             ]
#             dt_val_features.append(dt_feature)
#             dt_val_targets.append(sample['is_best_move'])
        
#         self.dt_val_features = np.array(dt_val_features)
#         self.dt_val_targets = np.array(dt_val_targets)
        
#         logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
#     def train_neural_network(self, epochs=50, batch_size=64):
#         """Train the neural network"""
#         if not self.use_neural_net:
#             return
        
#         logger.info("Training neural network...")
        
#         for epoch in range(epochs):
#             self.neural_net.train()
#             total_loss = 0
#             correct_predictions = 0
#             total_predictions = 0
            
#             # Training loop
#             for i in range(0, len(self.train_features), batch_size):
#                 batch_features = self.train_features[i:i+batch_size]
#                 batch_targets = self.train_targets[i:i+batch_size]
                
#                 self.optimizer.zero_grad()
#                 outputs = self.neural_net(batch_features)
#                 loss = self.criterion(outputs, batch_targets)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
                
#                 # Calculate accuracy
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_predictions += batch_targets.size(0)
#                 correct_predictions += (predicted == batch_targets).sum().item()
            
#             # Validation
#             self.neural_net.eval()
#             with torch.no_grad():
#                 val_outputs = self.neural_net(self.val_features)
#                 val_loss = self.criterion(val_outputs, self.val_targets)
#                 _, val_predicted = torch.max(val_outputs.data, 1)
#                 val_accuracy = (val_predicted == self.val_targets).float().mean().item()
            
#             train_accuracy = correct_predictions / total_predictions
#             self.accuracy_history.append(train_accuracy)
#             self.validation_accuracy.append(val_accuracy)
            
#             if epoch % 10 == 0:
#                 logger.info(f"Epoch {epoch}: Train Loss: {total_loss:.4f}, "
#                           f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    
#     def train_decision_tree(self):
#         """Train the decision tree"""
#         logger.info("Training decision tree...")
        
#         self.decision_tree.fit(self.dt_train_features, self.dt_train_targets)
        
#         # Calculate accuracy
#         train_pred = self.decision_tree.predict(self.dt_train_features)
#         val_pred = self.decision_tree.predict(self.dt_val_features)
        
#         train_accuracy = accuracy_score(self.dt_train_targets, train_pred)
#         val_accuracy = accuracy_score(self.dt_val_targets, val_pred)
        
#         logger.info(f"Decision Tree - Train Accuracy: {train_accuracy:.4f}, "
#                    f"Val Accuracy: {val_accuracy:.4f}")
        
#         return train_accuracy, val_accuracy
    
#     def predict_move(self, board):
#         """Predict best move using combined model"""
#         legal_moves = list(board.legal_moves)
#         if not legal_moves:
#             return None
        
#         move_scores = {}
#         features = self.board_to_features(board)
        
#         # Neural network prediction
#         if self.use_neural_net:
#             self.neural_net.eval()
#             with torch.no_grad():
#                 nn_input = torch.FloatTensor(features).unsqueeze(0).to(self.device)
#                 nn_output = self.neural_net(nn_input)
#                 nn_probs = nn_output.cpu().numpy()[0]
        
#         # Score each legal move
#         for move in legal_moves:
#             score = 0.0
            
#             # Neural network score
#             if self.use_neural_net:
#                 move_idx = self.move_to_index(move)
#                 if move_idx < len(nn_probs):
#                     score += nn_probs[move_idx] * 0.6
            
#             # Stockfish score
#             if self.engine:
#                 board_copy = board.copy()
#                 board_copy.push(move)
#                 stockfish_eval = self.get_stockfish_evaluation(board_copy)
#                 score += (stockfish_eval + 10) / 20 * 0.3  # Normalize to 0-1
            
#             # Decision tree score
#             dt_features = np.array([[
#                 self.get_stockfish_evaluation(board),
#                 len(board.move_stack),
#                 1500,  # Default ELO
#                 1500,  # Default ELO
#                 1 if board.turn == chess.WHITE else 0,
#                 0.0    # Unknown result
#             ]])
            
#             dt_prob = self.decision_tree.predict_proba(dt_features)[0]
#             if len(dt_prob) > 1:
#                 score += dt_prob[1] * 0.1  # Probability of being best move
            
#             move_scores[move] = score
        
#         # Return move with highest score
#         best_move = max(move_scores, key=move_scores.get)
#         return best_move
    
#     def evaluate_model(self, test_pgn_path, max_games=100):
#         """Evaluate model accuracy on test games"""
#         logger.info(f"Evaluating model on {test_pgn_path}...")
        
#         correct_predictions = 0
#         total_predictions = 0
#         games_evaluated = 0
        
#         with open(test_pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
#             while games_evaluated < max_games:
#                 game = chess.pgn.read_game(pgn_file)
#                 if game is None:
#                     break
                
#                 board = game.board()
#                 moves_in_game = 0
                
#                 for move in game.mainline_moves():
#                     if moves_in_game > 30:  # Evaluate first 30 moves
#                         break
                    
#                     # Get model prediction
#                     predicted_move = self.predict_move(board)
                    
#                     if predicted_move:
#                         total_predictions += 1
#                         if predicted_move == move:
#                             correct_predictions += 1
                    
#                     board.push(move)
#                     moves_in_game += 1
                
#                 games_evaluated += 1
                
#                 if games_evaluated % 10 == 0:
#                     current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#                     logger.info(f"Evaluated {games_evaluated} games, "
#                               f"Current accuracy: {current_accuracy:.4f}")
        
#         final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#         logger.info(f"Final evaluation accuracy: {final_accuracy:.4f} "
#                    f"({correct_predictions}/{total_predictions})")
        
#         return final_accuracy
    
#     def save_model(self, output_path):
#         """Save the trained model"""
#         model_data = {
#             'neural_net_state': self.neural_net.state_dict() if self.use_neural_net else None,
#             'decision_tree': self.decision_tree,
#             'move_patterns': dict(self.move_patterns),
#             'opening_moves': dict(self.opening_moves),
#             'accuracy_history': self.accuracy_history,
#             'validation_accuracy': self.validation_accuracy,
#             'training_stats': {
#                 'total_samples': len(self.training_data),
#                 'positions_analyzed': len(self.move_patterns),
#                 'opening_moves': len(self.opening_moves)
#             }
#         }
        
#         # Save with pickle
#         with open(output_path, 'wb') as f:
#             pickle.dump(model_data, f)
        
#         # Save decision tree separately
#         dt_path = output_path.replace('.pkl', '_decision_tree.joblib')
#         joblib.dump(self.decision_tree, dt_path)
        
#         # Save neural network separately
#         if self.use_neural_net:
#             nn_path = output_path.replace('.pkl', '_neural_net.pth')
#             torch.save(self.neural_net.state_dict(), nn_path)
        
#         logger.info(f"Model saved to {output_path}")
    
#     def load_model(self, model_path):
#         """Load a trained model"""
#         with open(model_path, 'rb') as f:
#             model_data = pickle.load(f)
        
#         # Load decision tree
#         dt_path = model_path.replace('.pkl', '_decision_tree.joblib')
#         if os.path.exists(dt_path):
#             self.decision_tree = joblib.load(dt_path)
        
#         # Load neural network
#         if self.use_neural_net:
#             nn_path = model_path.replace('.pkl', '_neural_net.pth')
#             if os.path.exists(nn_path):
#                 self.neural_net.load_state_dict(torch.load(nn_path, map_location=self.device))
        
#         self.move_patterns = model_data.get('move_patterns', {})
#         self.opening_moves = model_data.get('opening_moves', {})
#         self.accuracy_history = model_data.get('accuracy_history', [])
        
#         logger.info("Model loaded successfully")
    
#     def train_model(self, pgn_files, output_path, max_games_per_file=1000):
#         """Main training function"""
#         logger.info("Starting enhanced model training...")
        
#         # Process PGN files
#         total_games = 0
#         for pgn_file in pgn_files:
#             if os.path.exists(pgn_file):
#                 logger.info(f"Processing {pgn_file}...")
#                 games = self.process_pgn_file(pgn_file, max_games_per_file)
#                 total_games += games
#             else:
#                 logger.warning(f"PGN file {pgn_file} not found")
        
#         if not self.training_data:
#             logger.error("No training data collected!")
#             return None
        
#         # Prepare training data
#         self.prepare_training_data()
        
#         # Train neural network
#         if self.use_neural_net:
#             self.train_neural_network(epochs=100)
        
#         # Train decision tree
#         dt_train_acc, dt_val_acc = self.train_decision_tree()
        
#         # Save model
#         self.save_model(output_path)
        
#         logger.info(f"Training complete: {total_games} games processed")
#         logger.info(f"Final decision tree accuracy: {dt_val_acc:.4f}")
        
#         return {
#             'total_games': total_games,
#             'decision_tree_accuracy': dt_val_acc,
#             'neural_net_accuracy': self.validation_accuracy[-1] if self.validation_accuracy else 0
#         }
    
#     def __del__(self):
#         """Cleanup"""
#         if self.engine:
#             self.engine.quit()


# # Main training script
# def train_enhanced_chess_model(pgn_files, model_output_path, stockfish_path=None, 
#                               test_pgn_path=None):
#     """Main training function with evaluation"""
    
#     # Initialize trainer
#     trainer = EnhancedChessModelTrainer(
#         stockfish_path=stockfish_path,
#         use_neural_net=True
#     )
    
#     # Train model
#     training_results = trainer.train_model(pgn_files, model_output_path)
    
#     # Evaluate on test set if provided
#     if test_pgn_path and os.path.exists(test_pgn_path):
#         test_accuracy = trainer.evaluate_model(test_pgn_path)
#         training_results['test_accuracy'] = test_accuracy
    
#     return training_results, trainer


# if __name__ == "__main__":
#     # Configuration
#     pgn_files = [
#         "games/masters_games.pgn",
#         "games/your_training_games.pgn"
#     ]
    
#     model_path = "models/enhanced_chess_model.pkl"
#     stockfish_path = "/opt/homebrew/bin/stockfish"  # Update this path
#     test_pgn_path = "games/test_games.pgn"
    
#     # Create directories
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("games", exist_ok=True)
    
#     # Train model
#     results, trainer = train_enhanced_chess_model(
#         pgn_files=pgn_files,
#         model_output_path=model_path,
#         stockfish_path=stockfish_path,
#         test_pgn_path=test_pgn_path
#     )
    
#     print("Training Results:")
#     for key, value in results.items():
#         print(f"{key}: {value}")
    
#     # Example of using the trained model
#     print("\nTesting model prediction...")
#     test_board = chess.Board()
#     predicted_move = trainer.predict_move(test_board)
#     print(f"Predicted opening move: {predicted_move}")




# enhanced_chess_trainer_fixed.py
import chess
import chess.pgn
import chess.engine
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import os
import random
import time
import threading
import signal
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout operations"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Set the signal handler and a alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class ChessNeuralNetwork(nn.Module):
    """Optimized neural network for chess position evaluation"""
    def __init__(self, input_size=773, hidden_size=256, output_size=4096):
        super(ChessNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class EnhancedChessModelTrainer:
    def __init__(self, stockfish_path=None, use_neural_net=True, stockfish_timeout=2):
        self.move_patterns = defaultdict(list)
        self.position_evaluations = defaultdict(float)
        self.opening_moves = defaultdict(int)
        self.endgame_patterns = defaultdict(list)
        self.training_data = []
        self.validation_data = []
        self.stockfish_timeout = stockfish_timeout
        self.stockfish_calls = 0
        self.stockfish_timeouts = 0
        
        # Initialize Stockfish engine with timeout protection
        self.engine = None
        if stockfish_path and os.path.exists(stockfish_path):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                # Test engine with timeout
                test_board = chess.Board()
                with timeout(5):
                    self.engine.analyse(test_board, chess.engine.Limit(depth=1))
                logger.info("Stockfish engine initialized and tested successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Stockfish: {e}")
                if self.engine:
                    try:
                        self.engine.quit()
                    except:
                        pass
                    self.engine = None
        
        # Initialize neural network with device optimization
        self.use_neural_net = use_neural_net
        if use_neural_net:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self.device = device
            self.neural_net = ChessNeuralNetwork().to(self.device)
            self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            logger.info(f"Neural network initialized on {self.device}")
        
        # Decision tree for move selection
        self.decision_tree = DecisionTreeClassifier(
            max_depth=10,  # Reduced for faster training
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Accuracy tracking
        self.accuracy_history = []
        self.validation_accuracy = []
        
        # Progress tracking
        self.start_time = None
        self.last_progress_time = None
        
    def board_to_features(self, board):
        """Convert chess board to feature vector with caching"""
        features = []
        
        # Piece positions (64 squares * 12 piece types = 768 features)
        piece_map = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece_features = [0] * 12
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece_map[piece.piece_type] - 1
                if piece.color == chess.WHITE:
                    piece_features[piece_idx] = 1
                else:
                    piece_features[piece_idx + 6] = 1
            features.extend(piece_features)
        
        # Additional features
        features.append(1 if board.turn == chess.WHITE else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        return np.array(features, dtype=np.float32)
    
    def get_stockfish_evaluation_safe(self, board, depth=10):
        """Get position evaluation from Stockfish with timeout protection"""
        if not self.engine:
            return 0.0
        
        self.stockfish_calls += 1
        
        try:
            with timeout(self.stockfish_timeout):
                info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].relative
                
                if score.is_mate():
                    return 10000 if score.mate() > 0 else -10000
                else:
                    return float(score.score()) / 100.0
        except TimeoutException:
            self.stockfish_timeouts += 1
            logger.warning(f"Stockfish evaluation timeout (#{self.stockfish_timeouts})")
            return 0.0
        except Exception as e:
            logger.warning(f"Stockfish evaluation failed: {e}")
            return 0.0
    
    def get_stockfish_best_move_safe(self, board, time_limit=0.5):
        """Get best move from Stockfish with timeout protection"""
        if not self.engine:
            return None
        
        try:
            with timeout(max(1, int(time_limit * 2))):
                result = self.engine.play(board, chess.engine.Limit(time=time_limit))
                return result.move
        except TimeoutException:
            logger.warning("Stockfish move generation timeout")
            return None
        except Exception as e:
            logger.warning(f"Stockfish move generation failed: {e}")
            return None
    
    def move_to_index(self, move):
        """Convert move to index for neural network output"""
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square
    
    def simple_position_evaluation(self, board):
        """Fast position evaluation without engine - FIXED VERSION"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
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
        
        # Add positional bonuses - FIXED: Convert generator to list for counting
        legal_moves_count = len(list(board.legal_moves))
        score += legal_moves_count * 0.1 if board.turn == chess.WHITE else -0.1
        
        return score
    
    def process_pgn_file(self, pgn_file_path, max_games=None):
        """Process PGN file with progress tracking and timeout protection"""
        games_processed = 0
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        
        logger.info(f"Starting to process {pgn_file_path}")
        
        try:
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    if max_games and games_processed >= max_games:
                        break
                    
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        
                        # Process game with timeout
                        with timeout(30):  # 30 second timeout per game
                            self.extract_game_features_fast(game)
                        
                        games_processed += 1
                        
                        # Progress reporting
                        current_time = time.time()
                        if current_time - self.last_progress_time > 10:  # Every 10 seconds
                            elapsed = current_time - self.start_time
                            rate = games_processed / elapsed if elapsed > 0 else 0
                            logger.info(f"Processed {games_processed} games in {elapsed:.1f}s "
                                      f"(Rate: {rate:.2f} games/s, Stockfish calls: {self.stockfish_calls}, "
                                      f"Timeouts: {self.stockfish_timeouts})")
                            self.last_progress_time = current_time
                        
                    except TimeoutException:
                        logger.warning(f"Game {games_processed} processing timeout, skipping...")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing game {games_processed}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading PGN file: {e}")
            return games_processed
        
        total_time = time.time() - self.start_time
        logger.info(f"Completed processing {games_processed} games in {total_time:.1f}s")
        logger.info(f"Stockfish statistics: {self.stockfish_calls} calls, {self.stockfish_timeouts} timeouts")
        
        return games_processed
    
    def extract_game_features_fast(self, game):
        """Extract features from a single game with optimizations"""
        board = game.board()
        move_count = 0
        
        # Get game metadata
        result = game.headers.get('Result', '*')
        try:
            white_elo = int(game.headers.get('WhiteElo', '1500'))
            black_elo = int(game.headers.get('BlackElo', '1500'))
        except (ValueError, TypeError):
            white_elo = black_elo = 1500
        
        result_score = self.parse_result(result)
        
        # Limit Stockfish calls for efficiency
        use_stockfish_for_this_game = (
            self.engine is not None and 
            len(self.training_data) % 5 == 0 and  # Only every 5th game
            self.stockfish_timeouts < 10  # Stop if too many timeouts
        )
        
        for move in game.mainline_moves():
            if move_count > 50:  # Reduced from 100 for faster processing
                break
                
            move_count += 1
            
            # Extract features
            try:
                features = self.board_to_features(board)
                position_key = self.get_position_key_fast(board)
            except Exception as e:
                logger.warning(f"Error extracting features for move {move_count}: {e}")
                board.push(move)
                continue
            
            # Get evaluation (use simple evaluation most of the time)
            if use_stockfish_for_this_game and move_count <= 20:
                stockfish_eval = self.get_stockfish_evaluation_safe(board, depth=8)
                stockfish_best = self.get_stockfish_best_move_safe(board, time_limit=0.3)
            else:
                stockfish_eval = self.simple_position_evaluation(board)
                stockfish_best = None
            
            # Store training data
            move_index = self.move_to_index(move)
            is_best_move = 1 if stockfish_best and move == stockfish_best else 0
            
            training_sample = {
                'features': features,
                'move': move.uci(),
                'move_index': move_index,
                'position_key': position_key,
                'evaluation': stockfish_eval,
                'is_best_move': is_best_move,
                'move_number': move_count,
                'result': result_score,
                'white_elo': white_elo,
                'black_elo': black_elo,
                'player_to_move': 1 if board.turn == chess.WHITE else 0
            }
            
            self.training_data.append(training_sample)
            self.move_patterns[position_key].append(training_sample)
            
            # Store opening moves
            if move_count <= 15:
                opening_key = board.fen().split()[0]
                self.opening_moves[f"{opening_key}:{move.uci()}"] += 1
            
            board.push(move)
    
    def get_position_key_fast(self, board):
        """Generate a faster position key"""
        # Simplified key generation
        piece_count = len(board.piece_map())
        material_balance = self.simple_position_evaluation(board)
        
        return f"{piece_count}_{int(material_balance)}_{board.turn}_{len(board.move_stack)}"
    
    def parse_result(self, result):
        """Parse game result to numerical score"""
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
        else:
            return 0.0
    
    def prepare_training_data(self, validation_split=0.2):
        """Prepare data for training with memory optimization"""
        logger.info(f"Preparing {len(self.training_data)} training samples...")
        
        if len(self.training_data) == 0:
            logger.error("No training data available!")
            return False
        
        # Shuffle data
        random.shuffle(self.training_data)
        
        # Limit data size if too large
        max_samples = 50000  # Limit for memory efficiency
        if len(self.training_data) > max_samples:
            logger.info(f"Limiting training data to {max_samples} samples")
            self.training_data = self.training_data[:max_samples]
        
        # Split into training and validation
        split_idx = int(len(self.training_data) * (1 - validation_split))
        train_data = self.training_data[:split_idx]
        val_data = self.training_data[split_idx:]
        
        # Prepare neural network data
        if self.use_neural_net and len(train_data) > 0:
            try:
                train_features = np.array([sample['features'] for sample in train_data])
                train_targets = np.array([sample['move_index'] for sample in train_data])
                
                val_features = np.array([sample['features'] for sample in val_data])
                val_targets = np.array([sample['move_index'] for sample in val_data])
                
                self.train_features = torch.FloatTensor(train_features).to(self.device)
                self.train_targets = torch.LongTensor(train_targets).to(self.device)
                self.val_features = torch.FloatTensor(val_features).to(self.device)
                self.val_targets = torch.LongTensor(val_targets).to(self.device)
                
                logger.info(f"Neural network data prepared: {len(train_features)} training, {len(val_features)} validation")
            except Exception as e:
                logger.error(f"Error preparing neural network data: {e}")
                self.use_neural_net = False
        
        # Prepare decision tree data
        try:
            dt_features = []
            dt_targets = []
            
            for sample in train_data:
                dt_feature = [
                    sample['evaluation'],
                    sample['move_number'],
                    sample['white_elo'],
                    sample['black_elo'],
                    sample['player_to_move'],
                    sample['result']
                ]
                dt_features.append(dt_feature)
                dt_targets.append(sample['is_best_move'])
            
            self.dt_train_features = np.array(dt_features)
            self.dt_train_targets = np.array(dt_targets)
            
            # Validation data for decision tree
            dt_val_features = []
            dt_val_targets = []
            
            for sample in val_data:
                dt_feature = [
                    sample['evaluation'],
                    sample['move_number'],
                    sample['white_elo'],
                    sample['black_elo'],
                    sample['player_to_move'],
                    sample['result']
                ]
                dt_val_features.append(dt_feature)
                dt_val_targets.append(sample['is_best_move'])
            
            self.dt_val_features = np.array(dt_val_features)
            self.dt_val_targets = np.array(dt_val_targets)
            
            logger.info(f"Decision tree data prepared: {len(dt_features)} training, {len(dt_val_features)} validation")
            
        except Exception as e:
            logger.error(f"Error preparing decision tree data: {e}")
            return False
        
        return True
    
    def train_neural_network(self, epochs=30, batch_size=128):
        """Train the neural network with progress tracking"""
        if not self.use_neural_net:
            logger.info("Neural network training skipped")
            return
        
        logger.info("Starting neural network training...")
        
        for epoch in range(epochs):
            self.neural_net.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Training loop with progress tracking
            num_batches = len(self.train_features) // batch_size + 1
            for i in range(0, len(self.train_features), batch_size):
                batch_features = self.train_features[i:i+batch_size]
                batch_targets = self.train_targets[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.neural_net(batch_features)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += batch_targets.size(0)
                correct_predictions += (predicted == batch_targets).sum().item()
            
            # Validation
            self.neural_net.eval()
            with torch.no_grad():
                val_outputs = self.neural_net(self.val_features)
                val_loss = self.criterion(val_outputs, self.val_targets)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == self.val_targets).float().mean().item()
            
            train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.accuracy_history.append(train_accuracy)
            self.validation_accuracy.append(val_accuracy)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {total_loss:.4f}, "
                          f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    
    def train_decision_tree(self):
        """Train the decision tree"""
        logger.info("Training decision tree...")
        
        try:
            self.decision_tree.fit(self.dt_train_features, self.dt_train_targets)
            
            # Calculate accuracy
            train_pred = self.decision_tree.predict(self.dt_train_features)
            val_pred = self.decision_tree.predict(self.dt_val_features)
            
            train_accuracy = accuracy_score(self.dt_train_targets, train_pred)
            val_accuracy = accuracy_score(self.dt_val_targets, val_pred)
            
            logger.info(f"Decision Tree - Train Accuracy: {train_accuracy:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
            
            return train_accuracy, val_accuracy
        except Exception as e:
            logger.error(f"Decision tree training failed: {e}")
            return 0.0, 0.0
    
    def predict_move(self, board):
        """Predict best move using combined model"""
        legal_moves = list(board.legal_moves)  # Convert generator to list
        if not legal_moves:
            return None
        
        move_scores = {}
        features = self.board_to_features(board)
        
        # Neural network prediction
        if self.use_neural_net:
            self.neural_net.eval()
            with torch.no_grad():
                nn_input = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                nn_output = self.neural_net(nn_input)
                nn_probs = nn_output.cpu().numpy()[0]
        
        # Score each legal move
        for move in legal_moves:
            score = 0.0
            
            # Neural network score
            if self.use_neural_net:
                move_idx = self.move_to_index(move)
                if move_idx < len(nn_probs):
                    score += nn_probs[move_idx] * 0.6
            
            # Stockfish score
            if self.engine:
                board_copy = board.copy()
                board_copy.push(move)
                stockfish_eval = self.get_stockfish_evaluation_safe(board_copy)
                score += (stockfish_eval + 10) / 20 * 0.3  # Normalize to 0-1
            
            # Decision tree score
            dt_features = np.array([[
                self.get_stockfish_evaluation_safe(board),
                len(board.move_stack),
                1500,  # Default ELO
                1500,  # Default ELO
                1 if board.turn == chess.WHITE else 0,
                0.0    # Unknown result
            ]])
            
            try:
                dt_prob = self.decision_tree.predict_proba(dt_features)[0]
                if len(dt_prob) > 1:
                    score += dt_prob[1] * 0.1  # Probability of being best move
            except:
                pass  # Skip if decision tree not trained yet
            
            move_scores[move] = score
        
        # Return move with highest score
        if move_scores:
            best_move = max(move_scores, key=move_scores.get)
            return best_move
        else:
            return random.choice(legal_moves)
    
    def save_model(self, output_path):
        """Save the trained model"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            model_data = {
                'move_patterns': dict(self.move_patterns),
                'opening_moves': dict(self.opening_moves),
                'accuracy_history': self.accuracy_history,
                'validation_accuracy': self.validation_accuracy,
                'training_stats': {
                    'total_samples': len(self.training_data),
                    'positions_analyzed': len(self.move_patterns),
                    'opening_moves': len(self.opening_moves),
                    'stockfish_calls': self.stockfish_calls,
                    'stockfish_timeouts': self.stockfish_timeouts
                }
            }
            
            # Save main model data
            with open(output_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save decision tree separately
            dt_path = output_path.replace('.pkl', '_decision_tree.joblib')
            joblib.dump(self.decision_tree, dt_path)
            
            # Save neural network separately
            if self.use_neural_net:
                nn_path = output_path.replace('.pkl', '_neural_net.pth')
                torch.save(self.neural_net.state_dict(), nn_path)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def train_model(self, pgn_files, output_path, max_games_per_file=500):
        """Main training function with comprehensive error handling"""
        logger.info("Starting enhanced model training...")
        
        # Process PGN files
        total_games = 0
        for pgn_file in pgn_files:
            if os.path.exists(pgn_file):
                logger.info(f"Processing {pgn_file}...")
                try:
                    games = self.process_pgn_file(pgn_file, max_games_per_file)
                    total_games += games
                except Exception as e:
                    logger.error(f"Error processing {pgn_file}: {e}")
                    continue
            else:
                logger.warning(f"PGN file {pgn_file} not found")
        
        if not self.training_data:
            logger.error("No training data collected!")
            return None
        
        # Prepare training data
        if not self.prepare_training_data():
            logger.error("Failed to prepare training data!")
            return None
        
        # Train neural network
        if self.use_neural_net:
            try:
                self.train_neural_network(epochs=20)
            except Exception as e:
                logger.error(f"Neural network training failed: {e}")
        
        # Train decision tree
        try:
            dt_train_acc, dt_val_acc = self.train_decision_tree()
        except Exception as e:
            logger.error(f"Decision tree training failed: {e}")
            dt_train_acc = dt_val_acc = 0.0
        
        # Save model
        try:
            self.save_model(output_path)
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
        
        logger.info(f"Training complete: {total_games} games processed")
        logger.info(f"Final decision tree accuracy: {dt_val_acc:.4f}")
        
        return {
            'total_games': total_games,
            'decision_tree_accuracy': dt_val_acc,
            'neural_net_accuracy': self.validation_accuracy[-1] if self.validation_accuracy else 0,
            'stockfish_calls': self.stockfish_calls,
            'stockfish_timeouts': self.stockfish_timeouts
        }
    
    def __del__(self):
        """Cleanup resources"""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass


# Main training script
def train_enhanced_chess_model(pgn_files, model_output_path, stockfish_path=None):
    """Main training function with timeout protection"""
    
    logger.info("Initializing enhanced chess model trainer...")
    
    # Initialize trainer with timeout protection
    trainer = EnhancedChessModelTrainer(
        stockfish_path=stockfish_path,
        use_neural_net=True,
        stockfish_timeout=2  # 2 second timeout for Stockfish calls
    )
    
    # Train model
    try:
        training_results = trainer.train_model(pgn_files, model_output_path, max_games_per_file=200)
        return training_results, trainer
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, trainer


if __name__ == "__main__":
    # Configuration
    pgn_files = [
        "games/masters_games.pgn",
        # Add more PGN files here
    ]
    
    model_path = "models/enhanced_chess_model.pkl"
    stockfish_path = "/opt/homebrew/bin/stockfish"  # Update this path
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("games", exist_ok=True)
    
    logger.info("Starting chess model training...")
    
    # Train model
    results, trainer = train_enhanced_chess_model(
        pgn_files=pgn_files,
        model_output_path=model_path,
        stockfish_path=stockfish_path
    )
    
    if results:
        print("\nTraining Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Test model prediction
        print("\nTesting model prediction...")
        try:
            test_board = chess.Board()
            predicted_move = trainer.predict_move(test_board)
            print(f"Sample move: {predicted_move}")
        except Exception as e:
            print(f"Error in prediction test: {e}")
    else:
        print("Training failed!")
