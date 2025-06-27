# chess_gui.py
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import chess
import chess.svg
from PIL import Image, ImageTk
import os
from chess_engine import ChessBot

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Bot - AI Chess Game")
        self.root.geometry("800x700")
        self.root.resizable(False, False)
        
        # Initialize chess bot
        self.chess_bot = ChessBot()
        self.board = self.chess_bot.board
        
        # GUI variables
        self.selected_square = None
        self.square_size = 64
        self.board_squares = {}
        self.piece_images = {}
        self.highlighted_squares = []
        self.last_move = None
        
        # Game state
        self.human_color = chess.WHITE
        self.ai_thinking = False
        
        self.setup_gui()
        self.load_piece_images()
        self.draw_board()
        self.update_status()
    
    def setup_gui(self):
        """Setup the main GUI components"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Board frame
        board_frame = tk.Frame(main_frame, bg='#2b2b2b')
        board_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Chess board canvas
        self.canvas = tk.Canvas(
            board_frame, 
            width=self.square_size * 8, 
            height=self.square_size * 8,
            highlightthickness=2,
            highlightbackground='#8B4513'
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#2b2b2b', width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            control_frame, 
            text="Chess Bot", 
            font=("Arial", 18, "bold"),
            fg='white', 
            bg='#2b2b2b'
        )
        title_label.pack(pady=(0, 20))
        
        # Game info
        self.info_frame = tk.LabelFrame(
            control_frame, 
            text="Game Info", 
            fg='white', 
            bg='#2b2b2b',
            font=("Arial", 10, "bold")
        )
        self.info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_label = tk.Label(
            self.info_frame, 
            text="Turn: White", 
            fg='white', 
            bg='#2b2b2b'
        )
        self.turn_label.pack(anchor=tk.W)
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="Status: Playing", 
            fg='white', 
            bg='#2b2b2b'
        )
        self.status_label.pack(anchor=tk.W)
        
        self.material_label = tk.Label(
            self.info_frame, 
            text="Material: Even", 
            fg='white', 
            bg='#2b2b2b'
        )
        self.material_label.pack(anchor=tk.W)
        
        # Controls
        controls_frame = tk.LabelFrame(
            control_frame, 
            text="Controls", 
            fg="#514c4c", 
            bg='#2b2b2b',
            font=("Arial", 10, "bold")
        )
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(
            controls_frame, 
            text="New Game", 
            command=self.new_game,
            bg='#4CAF50', 
            fg="#514c4c",  
            font=("Arial", 10, "bold")
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            controls_frame, 
            text="Undo Move", 
            command=self.undo_move,
            bg='#FF9800', 
            fg="#514c4c",  
            font=("Arial", 10, "bold")
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            controls_frame, 
            text="Flip Board", 
            command=self.flip_board,
            bg='#2196F3', 
            fg="#514c4c", 
            font=("Arial", 10, "bold")
        ).pack(fill=tk.X, pady=2)
        
        # AI Settings
        ai_frame = tk.LabelFrame(
            control_frame, 
            text="AI Settings", 
            fg="#514c4c",  
            bg='#2b2b2b',
            font=("Arial", 10, "bold")
        )
        ai_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(ai_frame, text="Difficulty:", fg='white', bg='#2b2b2b').pack(anchor=tk.W)
        self.difficulty_var = tk.StringVar(value="Medium")
        difficulty_combo = ttk.Combobox(
            ai_frame, 
            textvariable=self.difficulty_var,
            values=["Easy", "Medium", "Hard"], 
            state="readonly"
        )
        difficulty_combo.pack(fill=tk.X, pady=2)
        
        self.auto_play_var = tk.BooleanVar()
        tk.Checkbutton(
            ai_frame, 
            text="AI vs AI", 
            variable=self.auto_play_var,
            fg='white', 
            bg='#2b2b2b', 
            selectcolor='#2b2b2b'
        ).pack(anchor=tk.W)
        
        # Move history
        history_frame = tk.LabelFrame(
            control_frame, 
            text="Move History", 
            fg='white', 
            bg='#2b2b2b',
            font=("Arial", 10, "bold")
        )
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = tk.Text(
            history_frame, 
            width=20, 
            height=10, 
            bg='#1e1e1e', 
            fg='white',
            font=("Courier", 9)
        )
        scrollbar = tk.Scrollbar(history_frame, command=self.history_text.yview)
        self.history_text.config(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_piece_images(self):
        """Load piece images"""
        # Create simple colored squares as piece representations
        # In a full implementation, you would load actual piece images
        pieces = ['p', 'r', 'n', 'b', 'q', 'k', 'P', 'R', 'N', 'B', 'Q', 'K']
        colors = {
            'p': '#8B4513', 'r': '#8B4513', 'n': '#8B4513', 'b': '#8B4513', 'q': '#8B4513', 'k': '#8B4513',
            'P': '#DEB887', 'R': '#DEB887', 'N': '#DEB887', 'B': '#DEB887', 'Q': '#DEB887', 'K': '#DEB887'
        }
        
        for piece in pieces:
            # Create a simple colored rectangle as piece image
            img = Image.new('RGB', (self.square_size-4, self.square_size-4), colors[piece])
            self.piece_images[piece] = ImageTk.PhotoImage(img)
    
    def draw_board(self):
        """Draw the chess board"""
        self.canvas.delete("all")
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = '#F0D9B5'  # Light squares
                else:
                    color = '#B58863'  # Dark squares
                
                # Highlight selected square
                square = chess.square(col, 7-row)
                if square == self.selected_square:
                    color = '#FFFF00'  # Yellow for selected
                elif square in self.highlighted_squares:
                    color = '#90EE90'  # Light green for possible moves
                elif self.last_move and square in [self.last_move.from_square, self.last_move.to_square]:
                    color = '#FFB6C1'  # Light pink for last move
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
                
                # Store square mapping
                self.board_squares[(row, col)] = square
        
        # Draw pieces
        self.draw_pieces()
        
        # Draw coordinates
        self.draw_coordinates()
    
    def draw_pieces(self):
        """Draw pieces on the board"""
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7-row)
                piece = self.board.piece_at(square)
                
                if piece:
                    x = col * self.square_size + self.square_size // 2
                    y = row * self.square_size + self.square_size // 2
                    
                    # Use simple text representation for pieces
                    piece_symbol = piece.unicode_symbol()
                    self.canvas.create_text(
                        x, y, 
                        text=piece_symbol, 
                        font=("Arial", 32), 
                        fill='black'
                    )
    
    def draw_coordinates(self):
        """Draw board coordinates"""
        files = 'abcdefgh'
        ranks = '87654321'
        
        for i, file in enumerate(files):
            x = i * self.square_size + self.square_size // 2
            self.canvas.create_text(x, 8 * self.square_size + 10, text=file, font=("Arial", 10))
        
        for i, rank in enumerate(ranks):
            y = i * self.square_size + self.square_size // 2
            self.canvas.create_text(-10, y, text=rank, font=("Arial", 10))
    
    def on_square_click(self, event):
        """Handle square clicks"""
        if self.ai_thinking or self.board.is_game_over():
            return
        
        col = event.x // self.square_size
        row = event.y // self.square_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            clicked_square = chess.square(col, 7-row)
            
            if self.selected_square is None:
                # Select piece
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.highlight_legal_moves(clicked_square)
            else:
                # Make move
                move = chess.Move(self.selected_square, clicked_square)
                
                # Check for promotion
                if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                    chess.square_rank(clicked_square) in [0, 7]):
                    move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self.make_move(move)
                
                self.clear_highlights()
            
            self.draw_board()
    
    def highlight_legal_moves(self, square):
        """Highlight legal moves for selected piece"""
        self.highlighted_squares = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                self.highlighted_squares.append(move.to_square)
    
    def clear_highlights(self):
        """Clear all highlights"""
        self.selected_square = None
        self.highlighted_squares = []
    
    def make_move(self, move):
        """Make a move and update the game"""
        if move in self.board.legal_moves:
            # Record move in history
            move_san = self.board.san(move)
            move_number = self.board.fullmove_number
            
            self.board.push(move)
            self.last_move = move
            
            # Update move history
            if self.board.turn == chess.WHITE:  # Just made a black move
                self.history_text.insert(tk.END, f"{move_number}...{move_san}\n")
            else:  # Just made a white move
                self.history_text.insert(tk.END, f"{move_number}.{move_san} ")
            
            self.history_text.see(tk.END)
            
            self.update_status()
            self.clear_highlights()
            
            # Check if game is over
            if self.board.is_game_over():
                self.handle_game_over()
            else:
                if self.auto_play_var.get():
                    # AI vs AI mode
                    self.root.after(1000, self.make_ai_move)
                elif self.board.turn != self.human_color:
                    # AI's turn
                    self.root.after(500, self.make_ai_move)


            # elif not self.auto_play_var.get() and self.board.turn != self.human_color:
            #     # AI's turn
            #     self.root.after(500, self.make_ai_move)
            # elif self.auto_play_var.get():
            #     # AI vs AI mode
            #     self.root.after(1000, self.make_ai_move)
    
    def make_ai_move(self):
        """Make AI move"""
        if self.board.is_game_over():
            return
        
        self.ai_thinking = True
        self.status_label.config(text="Status: AI Thinking...")
        self.root.update()
        
        # Get difficulty depth
        depth_map = {"Easy": 2, "Medium": 3, "Hard": 4}
        depth = depth_map[self.difficulty_var.get()]
        
        # Get AI move
        ai_move = self.chess_bot.get_best_move(depth)
        
        if ai_move:
            self.make_move(ai_move)
        
        self.ai_thinking = False
        self.draw_board()
        self.update_status()
    
    def update_status(self):
        """Update game status display"""
        # Update turn
        turn_text = "White" if self.board.turn == chess.WHITE else "Black"
        self.turn_label.config(text=f"Turn: {turn_text}")
        
        # Update status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.config(text=f"Status: {winner} Wins!")
        elif self.board.is_stalemate():
            self.status_label.config(text="Status: Stalemate")
        elif self.board.is_check():
            self.status_label.config(text="Status: Check")
        elif self.ai_thinking:
            self.status_label.config(text="Status: AI Thinking...")
        else:
            self.status_label.config(text="Status: Playing")
        
        # Update material count
        material_balance = self.calculate_material_balance()
        if material_balance > 0:
            self.material_label.config(text=f"Material: White +{material_balance}")
        elif material_balance < 0:
            self.material_label.config(text=f"Material: Black +{abs(material_balance)}")
        else:
            self.material_label.config(text="Material: Even")
    
    def calculate_material_balance(self):
        """Calculate material balance"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9}
        
        white_material = black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type in piece_values:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material
    
    def handle_game_over(self):
        """Handle game over scenarios"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            messagebox.showinfo("Game Over", f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            messagebox.showinfo("Game Over", "Stalemate! It's a draw!")
        elif self.board.is_insufficient_material():
            messagebox.showinfo("Game Over", "Draw by insufficient material!")
        elif self.board.can_claim_threefold_repetition():
            messagebox.showinfo("Game Over", "Draw by repetition!")
    
    def new_game(self):
        """Start a new game"""
        self.chess_bot.reset_game()
        self.board = self.chess_bot.board
        self.clear_highlights()
        self.last_move = None
        self.history_text.delete(1.0, tk.END)
        self.draw_board()
        self.update_status()
    
    def undo_move(self):
        """Undo the last move"""
        if len(self.board.move_stack) > 0:
            self.board.pop()
            # If playing against AI, undo AI move too
            if len(self.board.move_stack) > 0 and not self.auto_play_var.get():
                self.board.pop()
            
            self.clear_highlights()
            self.draw_board()
            self.update_status()
            
            # Update history
            self.history_text.delete(1.0, tk.END)
            move_number = 1
            for i, move in enumerate(self.board.move_stack):
                move_san = self.board.san(move)
                if i % 2 == 0:  # White move
                    self.history_text.insert(tk.END, f"{move_number}.{move_san} ")
                else:  # Black move
                    self.history_text.insert(tk.END, f"{move_san}\n")
                    move_number += 1
    
    def flip_board(self):
        """Flip the board view"""
        self.human_color = not self.human_color
        self.draw_board()

def main():
    root = tk.Tk()
    game = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
