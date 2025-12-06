#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Loader
Unified loading of environment variables and configuration files
"""

import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Configuration Loader"""
    
    def __init__(self, project_root: Path = None):
        """
        Initialize configuration loader
        
        Args:
            project_root: Project root directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "config.yaml"
        
        # Load environment variables
        self._load_env_file()
    
    def _load_env_file(self):
        """Load .env file"""
        if not self.env_file.exists():
            print(f"Warning: .env file not found at {self.env_file}")
            print(f"   Some API keys may not be available.")
            return
        
        with open(self.env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Set environment variable (if not already set)
                    if key not in os.environ:
                        os.environ[key] = value
        
        print(f"Loaded environment variables from .env")
    
    def _substitute_env_vars(self, value: Any) -> Any:
        """
        Recursively substitute environment variable references in configuration
        
        Args:
            value: Configuration value
            
        Returns:
            Substituted value
        """
        if isinstance(value, str):
            # Replace references in ${VAR_NAME} format
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            
            return re.sub(pattern, replace_var, value)
        
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        
        else:
            return value
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load complete configuration
        
        Returns:
            Configuration dictionary with environment variables substituted
        """
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        print(f"Loaded configuration from config.yaml")
        
        return config
    
    def get_api_key(self, service: str, key_name: str = 'key') -> str:
        """
        Get API key
        
        Args:
            service: Service name (e.g. 'openai', 'google', 'reddit')
            key_name: Key field name
            
        Returns:
            API key
        """
        config = self.load_config()
        
        try:
            api_keys = config.get('api_keys', {})
            service_config = api_keys.get(service, {})
            
            if isinstance(service_config, dict):
                return service_config.get(key_name, '')
            else:
                return service_config
        
        except Exception as e:
            print(f"Warning: Could not get API key for {service}: {e}")
            return ''
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate availability of all API keys
        
        Returns:
            Key availability for each service
        """
        config = self.load_config()
        api_keys = config.get('api_keys', {})
        
        validation = {}
        
        # OpenAI
        openai_key = api_keys.get('openai', {}).get('key', '')
        validation['openai'] = bool(openai_key and not openai_key.startswith('${'))
        
        # Google
        google_key = api_keys.get('google', {}).get('api_key', '')
        validation['google'] = bool(google_key and not google_key.startswith('${'))
        
        # Reddit
        reddit_id = api_keys.get('reddit', {}).get('client_id', '')
        validation['reddit'] = bool(reddit_id and not reddit_id.startswith('${'))
        
        # Unpaywall
        unpaywall_email = api_keys.get('unpaywall', {}).get('email', '')
        validation['unpaywall'] = bool(unpaywall_email and not unpaywall_email.startswith('${'))
        
        # OpenReview
        openreview_mailto = api_keys.get('openreview', {}).get('mailto', '')
        validation['openreview'] = bool(openreview_mailto and not openreview_mailto.startswith('${'))
        
        return validation
    
    def print_validation_report(self):
        """Print API key validation report"""
        print("\n" + "="*60)
        print("API Keys Validation Report")
        print("="*60)
        
        validation = self.validate_api_keys()
        
        for service, is_valid in validation.items():
            status = "Available" if is_valid else "Missing"
            print(f"  {service:15s}: {status}")
        
        print("="*60 + "\n")
        
        # Warning
        if not all(validation.values()):
            print("Warning: Some API keys are missing!")
            print("   Please check your .env file and ensure all keys are set.\n")


# Global instance
_config_loader = None

def get_config_loader(project_root: Path = None) -> ConfigLoader:
    """Get global configuration loader instance"""
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(project_root)
    
    return _config_loader


def load_config(project_root: Path = None) -> Dict[str, Any]:
    """Shortcut function: Load configuration"""
    loader = get_config_loader(project_root)
    return loader.load_config()


def get_api_key(service: str, key_name: str = 'key', project_root: Path = None) -> str:
    """Shortcut function: Get API key"""
    loader = get_config_loader(project_root)
    return loader.get_api_key(service, key_name)


if __name__ == "__main__":
    """Test configuration loading"""
    loader = ConfigLoader()
    
    # Validate API keys
    loader.print_validation_report()
    
    # Load configuration
    config = loader.load_config()
    
    print("Configuration loaded successfully!")
    print(f"Research topic: {config.get('topic')}")
    
    # Test API key retrieval
    print("\nAPI Keys (masked):")
    print(f"  OpenAI: {get_api_key('openai')[:20]}...")
    print(f"  Google: {get_api_key('google', 'api_key')[:20]}...")
    print(f"  Reddit ID: {get_api_key('reddit', 'client_id')[:10]}...")
