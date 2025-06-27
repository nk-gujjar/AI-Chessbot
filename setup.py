# setup.py
from setuptools import setup
import os

APP = ['chess_gui.py']
DATA_FILES = [
    ('models', ['models/chess_model.pkl']),
    ('images', []),  # Add piece images if you have them
]

OPTIONS = {
    'argv_emulation': True,
    'packages': ['chess', 'numpy', 'PIL', 'tkinter'],
    'includes': ['chess.engine', 'chess.pgn', 'chess.svg'],
    'excludes': ['matplotlib', 'scipy'],
    'plist': {
        'CFBundleName': 'ChessBot',
        'CFBundleDisplayName': 'Chess Bot AI',
        'CFBundleGetInfoString': 'Chess Bot with AI - Play chess against computer',
        'CFBundleIdentifier': 'com.chessbot.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2025 ChessBot. All rights reserved.',
        'NSHighResolutionCapable': True,
    },
    'iconfile': 'icon.icns',  # Add an icon file if you have one
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    install_requires=[
        'python-chess>=1.999',
        'numpy>=1.20.0',
        'Pillow>=8.0.0',
    ]
)
