# chess_engine.py
import chess
import chess.pgn
import numpy as np
import pickle
import json
from collections import defaultdict
import random

class ChessBot:
    def __init__(self, model_path=None):
        self.board = chess.Board()
        self.move_history = []
        self.position_scores = defaultdict(float)
        self.opening_book = {}
        self.endgame_table = {}
        
        if model_path:
            self.load_model(model_path)
        else:
            self.initialize_default_weights()
    
    def initialize_default_weights(self):
        """Initialize default piece values and position weights"""
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Simplified position evaluation weights
        self.position_weights = {
            'center_control': 0.3,
            'piece_development': 0.2,
            'king_safety': 0.4,
            'material_balance': 0.5
        }
    
    def evaluate_position(self, board):
        """Evaluate current board position"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        # Position evaluation
        score += self.evaluate_center_control(board)
        score += self.evaluate_king_safety(board)
        score += self.evaluate_piece_development(board)
        
        return score if board.turn == chess.WHITE else -score
    
    def evaluate_center_control(self, board):
        """Evaluate control of center squares"""
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        score = 0
        
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                score += 0.5
            if board.is_attacked_by(chess.BLACK, square):
                score -= 0.5
                
        return score
    
    def evaluate_king_safety(self, board):
        """Evaluate king safety"""
        score = 0
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king:
            # Penalty for exposed king
            if board.is_attacked_by(chess.BLACK, white_king):
                score -= 2
        
        if black_king:
            if board.is_attacked_by(chess.WHITE, black_king):
                score += 2
                
        return score
    
    def evaluate_piece_development(self, board):
        """Evaluate piece development"""
        score = 0
        
        # Encourage knight and bishop development
        for square in [chess.B1, chess.G1, chess.C1, chess.F1]:
            if not board.piece_at(square):
                score += 0.3
                
        for square in [chess.B8, chess.G8, chess.C8, chess.F8]:
            if not board.piece_at(square):
                score -= 0.3
                
        return score
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_best_move(self, depth=3):
        """Get the best move using minimax"""
        if not self.board.legal_moves:
            return None
            
        best_move = None
        best_value = float('-inf') if self.board.turn else float('inf')
        
        for move in self.board.legal_moves:
            self.board.push(move)
            
            if self.board.turn:  # If it's white's turn after the move
                value = self.minimax(self.board, depth - 1, float('-inf'), float('inf'), False)
            else:
                value = self.minimax(self.board, depth - 1, float('-inf'), float('inf'), True)
            
            self.board.pop()
            
            if self.board.turn:  # Maximizing for white
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # Minimizing for black
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move
    
    def make_move(self, move):
        """Make a move on the board"""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
    
    def get_board_state(self):
        """Get current board state"""
        return {
            'fen': self.board.fen(),
            'turn': 'white' if self.board.turn else 'black',
            'legal_moves': [str(move) for move in self.board.legal_moves],
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'game_over': self.board.is_game_over()
        }
    
    def reset_game(self):
        """Reset the game to starting position"""
        self.board = chess.Board()
        self.move_history = []
    
    def load_model(self, path):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                self.position_scores = model_data.get('position_scores', {})
                self.opening_book = model_data.get('opening_book', {})
        except FileNotFoundError:
            print(f"Model file {path} not found. Using default weights.")
            self.initialize_default_weights()
    
    def save_model(self, path):
        """Save current model"""
        model_data = {
            'position_scores': dict(self.position_scores),
            'opening_book': self.opening_book,
            'piece_values': self.piece_values,
            'position_weights': self.position_weights
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
