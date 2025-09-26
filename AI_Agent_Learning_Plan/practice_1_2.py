from datetime import datetime 
import time 
from typing import List, Dict 

class Agent:
    def __init__(self, name:str, role: str):
        self.name = name 
        self.role = role
        self.status = "idle"
        self._log_action(f"Agent {self.name} initialized with role: {self.role}")

    def _log_action(self, action: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%ML%S")
        print(f"[{timestamp}] {self.name} ({self.role}): {action}")

    def _set_status(self, new_status: str) -> None:
        old_status = self.status 
        self.status = new_status 
        if old_status != new_status:
            self._log_action(f"Status changed: {old_status} -> {new_status}")

    def think(self, query: str) -> str:
        self._set_status("thinking")
        self._log_action(f"Think about: {query}")

        if "날씨" in query:
            thought = "날씨 정보를 확인하기 위해 외부 API를 호출"
        elif "안녕" in query:
            thought = "인사에 대한 응답 준비"
        else:
            thought = f"'{query}'에 대한 답변 생각"
        
        self._log_action(f"Thought result : {thought}")
        self._set_status("idle")
        return thought 
    
    def act(self, action:str) -> str:
        self._set_status("acting")
        self._log_action(f"Performing action: {action}")
        time.sleep(0.3)

        if "검색" in action:
            result = f"'{action}' 완료 - 관련 정보를 찾았습니다"
        elif "분석" in action:
            result = f"'{action}' 완료 - 분석 결과를 생성했습니다"
        else:
            result = f"'{action}' 수행 완료"
        
        self._log_action(f"Action result: {result}")
        self._set_status("idle")
        return result
    
    def respond(self, message: str) -> str:
        self._set_status("responding")
        self._log_action(f"Responding to : {message}")
        time.sleep(0.2)

        if "안녕" in message:
            response = "안녕하세요 저는 AI Agent입니다. 무엇을 도와드릴까요"
        elif "고맙" in message or "감사" in message:
            response = "언제든 도움이 필요하면 말씀해주세요"
        elif "?" in message:
            response = f"'{message}' 에 대한 답변을 드리겠습니다. 조금만 기다려주세요"
        else:
            response = f"'{message}' 메시지를 잘 받았습니다. 적절한 응답을 준비하겠습니다"
        
        self._log_action(f"Response generated: {response}")
        self._set_status("idle")
        return response
    
class ChatAgent(Agent):
    def __init__(self, name: str, role: str):

        super().__init__(name, role)

        self.conversation_history: List[dict] = [] 
        self._log_action("ChatAgent capabilites initialized")
    
    def chat(self, message: str) -> str:

        self._log_action(f"Starting chat with message: {message}")

        user_entry = {
            "type": "user",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.conversation_history.append(user_entry)

        response = self.respond(message)

        agent_entry = {
            "type": "agent",
            "message": response,
            "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.conversation_history.append(agent_entry)

        self._log_action(f"Chat completed. History length: {len(self.conversation_history)}")

        return response
    
    def respond(self, message: str) -> str:

        self._set_status("responding")
        self._log_action(f"ChatAgent respoding to: {message}")

        conversation_count = len([entry for entry in self.conversation_history])

    
    
class TaskAgent(Agent):

    def __init__(self, name: str, role: str):

        super().__init__(name, role)

        self.task_queue: List[dict] = []
        self.completed_tasks: List[dict] = []
        self._log_action("TaskAgent capabities initialized")

    def add_task(self, task: str, priority: str = "normal") -> None:

        task_entry = {
            "task" : task, 
            "priority" : priority,
            "added_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status" : "pending"

        }

        if priority == "high":
            self.task_queue.insert(0, task_entry)
        else:
            self.task_queue.append(task_entry)

        self._log_action(f"Task added: '{task}' (priority: {priority})")

    def execute_next_task(self) -> str:
        
        if not self.task_queue:
            result = "작업 큐가 비어있음"
            self._log_action(result)
            return result
        
        current_task = self.task_queue.pop(0)
        task_name = current_task["task"]

        self._log_action(f"Executing task: {task_name}")

        result = self.act(task_name)

        current_task["status"] = "completed"
        current_task["completed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_task["result"] = result
        self.completed_tasks.append(current_task)

        return result
    
    def act(self, action: str) -> str:

        self._set_status("acting")
        self._log_action(f"TaskAgent performing: {action}")

        




                         
                         

    

