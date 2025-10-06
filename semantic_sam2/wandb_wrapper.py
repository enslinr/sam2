"""
Weights & Biases Safe Wrapper
============================

A robust wrapper for W&B operations with retry logic, error handling,
and graceful degradation to prevent training crashes.
"""
"""
Weights & Biases Safe Wrapper
============================

A robust wrapper for W&B operations with retry logic, error handling,
and graceful degradation to prevent training crashes.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Union
from functools import wraps
import matplotlib.pyplot as plt



try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. W&B logging will be disabled.")

def retry_with_backoff(operation_name: str):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.enabled:
                return None
            
            last_exception = None
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self.logger.warning(
                        f"W&B {operation_name} failed (attempt {attempt+1}/{self.max_retries}): {e}"
                    )
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        time.sleep(delay)
            
            self.logger.error(
                f"W&B {operation_name} failed after {self.max_retries} attempts. "
                f"Last error: {last_exception}"
            )
            self.enabled = False
            self.logger.error("Disabling W&B logging due to persistent errors.")
            return None
        return wrapper
    return decorator


class WandBWrapper:
    """
    Safe wrapper for Weights & Biases operations with comprehensive error handling.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        base_delay: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logger or self._setup_default_logger()
        self.initialized = False
        self._login_attempted = False
        self._run = None
        
        if not WANDB_AVAILABLE and enabled:
            self.logger.warning("W&B requested but not available. Disabling W&B logging.")
            self.enabled = False

    def _setup_default_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        


    def flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary for W&B logging.
        
        Args:
            config: Configuration dictionary (possibly nested)
            prefix: Prefix for keys (used in recursion)
        
        Returns:
            Flattened dictionary with dot-separated keys
        """
        flattened = {}
        
        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self.flatten_config(value, new_key))
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to strings for W&B compatibility
                flattened[new_key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                # Keep primitive types as-is
                flattened[new_key] = value
            else:
                # Convert other types to string
                flattened[new_key] = str(value)
        
        return flattened

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Update W&B run.config with (flattened) configuration values.
        Does NOT write anything to run.summary.
        """
        if not self.enabled or not self.initialized:
            return
        flattened = self.flatten_config(config)
        # allow_val_change lets you tweak values if you resume
        wandb.config.update(flattened, allow_val_change=True)
        self.logger.info("Configuration logged to W&B config (not summary)")

    @retry_with_backoff("login")
    def login(self, api_key: Optional[str] = None) -> bool:
        if not self.enabled or self._login_attempted: return self.enabled
        key = api_key or os.getenv('WANDB_API_KEY')
        if not key:
            self.logger.warning("No W&B API key found. Disabling W&B logging.")
            self.enabled = False
            return False
        wandb.login(key=key)
        self._login_attempted = True
        return True

    @retry_with_backoff("init")
    def init(self, project: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        if not self.enabled: return False
        if not self._login_attempted and not self.login(): return False

        flattened_config = None
        if config:
            flattened_config = self.flatten_config(config)

        self._run = wandb.init(project=project, config=flattened_config, **kwargs)
        self.initialized = True
        self.logger.info(f"W&B initialized for project: {project}")
        return True
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.enabled and self._run is not None:
            return self._run.id
        return None

    def get_run_name(self) -> Optional[str]:
        """Get the current run name."""
        if self.enabled and self._run is not None:
            return self._run.name
        return None



    @retry_with_backoff("log")
    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None) -> None:
        if not self.enabled or not self.initialized:
            return
        if not data:
            self.logger.warning("W&B log was called with an empty dictionary. Skipping.")
            return
        
        wandb.log(data, step=step, commit=commit)
        time.sleep(1)

    def log_metrics(self, epoch: int, train_losses: Dict[str, float], val_losses: Dict[str, float]) -> None:
            if not self.enabled or not self.initialized:
                self.logger.warning("W&B logging is disabled or not initialized. Skipping metrics logging.")
                return
            
            # Validate inputs
            if not isinstance(epoch, int) or epoch < 0:
                self.logger.error(f"Invalid epoch value: {epoch}. Must be a non-negative integer.")
                return
            if not train_losses and not val_losses:
                self.logger.warning("Both train_losses and val_losses are empty. Skipping logging.")
                return

            log_dict = {}
            for key, value in train_losses.items():
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"Invalid train metric {key}: {value}. Must be numeric. Skipping.")
                    continue
                log_dict[f'{key}'] = value
            
            for key, value in val_losses.items():
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"Invalid val metric {key}: {value}. Must be numeric. Skipping.")
                    continue
                log_dict[f'{key}'] = value
            
            if log_dict:
                self.log(log_dict, step=epoch)
            else:
                self.logger.warning("No valid metrics to log after validation.")

    def log_figure_with_metrics(self, prefix: str, source_idx: int, source_name: str, epoch: int, fig: plt.Figure, iou_scores: Dict[str, float], close_figure: bool = True) -> None:
            if not self.enabled or not self.initialized:
                if close_figure:
                    plt.close(fig)
                return

            caption = f"Source: {source_idx} ({source_name})"
            figure_key = f"{prefix}_{source_name}"
            
            log_data = {figure_key: wandb.Image(fig, caption=caption)}
            log_data.update({f"{figure_key}_{class_name}": iou for class_name, iou in iou_scores.items()})
            
            if iou_scores:
                mean_iou = sum(iou_scores.values()) / len(iou_scores)
                log_data[f"{figure_key}_mean_iou"] = mean_iou
            
            self.log(log_data, step=epoch)
            if close_figure:
                plt.close(fig)

    @retry_with_backoff("finish")
    def finish(self) -> bool:
        if not self.enabled or not self.initialized: return False
        self.logger.info("W&B run finished successfully")
        wandb.finish()
        return True
    def is_available(self) -> bool:
        """Check if W&B is available and enabled."""
        return self.enabled and WANDB_AVAILABLE
    


# Convenience function for quick setup
def create_wandb_wrapper(
    enabled: bool = True,
    project: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    logger: Optional[logging.Logger] = None
) -> WandBWrapper:
    """
    Create and optionally initialize a W&B wrapper.
    
    Args:
        enabled: Whether W&B logging is enabled
        project: Project name (if provided, will auto-initialize)
        config: Configuration for initialization
        max_retries: Maximum retry attempts
        logger: Optional logger instance
        
    Returns:
        WandBWrapper instance
    """
    wrapper = WandBWrapper(
        enabled=enabled,
        max_retries=max_retries,
        logger=logger
    )
    
    if project and enabled:
        wrapper.init(project=project, config=config)
    
    return wrapper