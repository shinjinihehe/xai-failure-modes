"""
Configuration loader for IPTA 2026 XAI Benchmarking.
Supports YAML configs with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._resolve_paths()
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute paths based on project root."""
        project_root = Path(__file__).parent.parent.parent
        paths = self._config.get('paths', {})
        
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = str(project_root / value)
        
        # Also resolve data roots
        for dataset in ['busi', 'kvasir_seg']:
            if dataset in self._config.get('data', {}):
                data_root = self._config['data'][dataset].get('root', '')
                if data_root and not Path(data_root).is_absolute():
                    self._config['data'][dataset]['root'] = str(project_root / data_root)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, name: str, default: Any = None) -> Any:
        value = self._config.get(name, default)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        # config.py -> utils -> src -> project root
        config_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    return Config.from_yaml(config_path)


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance (singleton)."""
    global _config
    if _config is None or config_path is not None:
        _config = load_config(config_path)
    return _config
