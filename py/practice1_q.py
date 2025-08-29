import csv 
import os

class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path 

        if not os.path.exists(file_path):
            raise FileNotFoundError("file not found")
        
        self._data = None
        
    @property
    def data(self):
        if self._data is None:
             with open(self.file_path, 'r', encoding = 'utf-8') as f:
                 reader = csv.reader(f)
                 self._data = list(reader)
                return self._data 