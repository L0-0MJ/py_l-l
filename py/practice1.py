import pandas as pd 
import json 
import time 
import psutil
import os 
from typing import Dict, Any, List, Optional
from functools import wraps 
from contextlib import contextmanager 

class DataValidationError(Exception):
    pass 

class ConfigurationError(Exception):
    pass 

class FileProcessingError(Exception):
    pass 

class DataValidator:

    def __init__(self, config_path: str):

        self.config_path = config_path 
        self._validation_rules = None 
        self._load_config() 
    
    def _load_config(self):
        try:
            with safe_file_handler(self.config_path, 'r') as f:
                self._validation_rules = json.load(f)
        
        except (FileProcessingError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"설정파일 로드 실패: {str(e)}")
        
        def validation_rules(self) -> Dict[str, Any]:
            return self._validation_rules.copy() if self._validation_rules else {}
        
        @property 
        def validation_rules(self) -> Dict[str, Any]:
            return self._validation_rules.copy() if self._validation_rules else {}