
# ============================================================================
# FILE 7: utils.py (Helper utilities)
# ============================================================================
"""
Utility functions for the wake word detection pipeline
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """Save dictionary to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
