from datetime import datetime 
from typing import List, Dict, Optional, Any

class Memory:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.messages: List[Dict[str, Any]] = []
        self._log(f"Memory initialized with max_size: {max_size}")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Memory: {message}")

    def _create_message(self, role:str, content: str) -> Dict[str, Any]:

        return{
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.message)

        }
    
    def add_message(self, role: str, content: str) -> None:

        message = self._create_message(role, content)
        self.messages.append(message)

        if len(self.message) > self.max_size:
            removed_message = self.message.pop(0)
            self._log(f"Max size exceeded. Removed oldest message: {removed_message}")
        self._log(f"Message added - [{role}]: {content[:50]}{'...' if len(content) > 50 else''}")


    def get_messages(self) -> List[Dict[str, Any]]:

        return self.messages.copy() #복사본 반환
    
    def search(self, keyword: str) -> List[Dict[str, Any]]:
        keyword_lower = keyword.lower()
        results = [
            message for message in self.messages
            if keyword_lower in message['content'].lower()

        ]

        self._log(f"Search for '{keyword}' returned {len(results)} results")

    def delete_message(self, index: int) -> bool:
        