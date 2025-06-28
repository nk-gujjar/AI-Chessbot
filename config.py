# config.py
import os
import sys

# Configuration settings for the chess bot
class Config:
    # Paths
    if getattr(sys, 'frozen', False):
        # Running as compiled app
        BASE_DIR = os.path.dirname(sys.executable)
        MODELS_DIR = os.path.join(BASE_DIR, '..', 'Resources', 'models')
    else:
        # Running as script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Model settings
    # DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'chess_model.pkl')
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'enhanced_chess_model.pkl')
    
    # Game settings
    DEFAULT_DIFFICULTY = "Medium"
    DEFAULT_DEPTH = 3
    
    # GUI settings
    SQUARE_SIZE = 64
    BOARD_SIZE = SQUARE_SIZE * 8
    
    # Colors
    LIGHT_SQUARE_COLOR = '#F0D9B5'
    DARK_SQUARE_COLOR = '#B58863'
    HIGHLIGHT_COLOR = '#FFFF00'
    MOVE_HIGHLIGHT_COLOR = '#90EE90'
    LAST_MOVE_COLOR = '#FFB6C1'

# Utility functions
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)
