"""
Logging Utilities
==================
Setup and configuration for pipeline logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    log_file: str | Path = None,
    level: int = logging.INFO,
    name: str = "ring3d"
) -> logging.Logger:
    """
    Setup a logger for the pipeline.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressLogger:
    """
    A simple progress logger for pipeline steps.
    """
    
    def __init__(self, total_steps: int, logger: logging.Logger = None):
        """
        Initialize progress logger.
        
        Args:
            total_steps: Total number of steps in pipeline
            logger: Logger instance
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logger or logging.getLogger("ring3d")
        self.start_time = datetime.now()
    
    def step(self, message: str):
        """Log a pipeline step."""
        self.current_step += 1
        progress = f"[{self.current_step}/{self.total_steps}]"
        self.logger.info(f"{progress} {message}")
    
    def done(self):
        """Log completion."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Pipeline completed in {elapsed.total_seconds():.1f} seconds")
