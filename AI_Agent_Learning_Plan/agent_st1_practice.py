from datetime import datetime
import time  
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Agent:

    def __init__(self, name : str, role : str):
        
        self.name = name
        self.role = role 
        self.status = "idle"
        self._log_action(f"Agent {self.name} initialized with role: {self.role}")
    
    def _log_action(self, action:str) -> None:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.name} ({self.role}):{action}")

    def _set_status(self, new_status: str) -> None:
        
        old_status = self.status 
        self.status = new_status 
        if old_status != new_status:
            self._log_action(f"Status changed: {old_status} -> {new_status}")
    
    def think(self, query:str) -> str:

        self._set_status("thinking")
        self._log_action(f"Thinking about: {query}")

        time.sleep(0.5)

        if "날씨" in query:
            thought = "날씨 정보를 확인하기 위해 외부 API 호출해야겠다"

        elif "안녕" in query:
            thought = "인사에 대한 적절한 응답을 준비해야겠다"
        
        else:
            thought = f"'{query}'에 대한 적절한 답변을 생각해보겠다"
        
        self._log_action(f"Thought result: {thought}")
        self._set_status("idle")

        return thought
    
    def act(self, action: str) -> str:
        self._set_status("acting")
        self._log_action(f"Performing action: {action}")

        time.sleep(0.3)

        if "검색" in action:
            result = f"'{action}' 완료 - 관련 정보를 찾음"
        elif "분석" in action:
            result = f"'{action}'완료 - 분석 결과를 생성"
        else:
            result = f"'{action}' 수행 완료"

        self._log_action(f"Action result: {result}")
        self._set_status("idle")

        return result 
    
    def respond(self, message: str) -> str:

        self._set_status("responding")
        self._log_action(f"Responding to: {message}")

        time.sleep(0.2)

        if "안녕" in message:
            response = "안녕하세요 저는 ai agent 입니다. 무엇을 도와드릴까요"
        self._log_action(f"Response generate: {response}")
        self._set_status("idle")

        return response
    
    def __str__(self) -> str:
        return f"Agent(name = '{self.name}, role='{self.role}',status'{self.status}')"
     
    def __repr__(self) -> str:
        return self.__str__() 
    
if __name__ == "__name__"   :

    agent = Agent("Alice", "assistance")
    print(f"Agent 정보: {agent}")
    print(f"초기 상태:{agent.status}")

    thought = agent.think("오늘 날씨가 어때")
    print(f"생각 결과:{thought}")

    action_result = agent.act("날씨정보 검색")
    print(f"행동 결과: {action_result}")

    response = agent.respond("안녕하세요")
    print(f"응답결과: {response}")

    print(f"최종상태: {agent.status}")
    print(f"최종 Agent 정보: {agent}")

    test_queries = [
        "내일 회의 일정이 어떻게 되나요",
        "감사합니다",
        "데이터 분석 좀 해주세요"
    ]
    for i , query in enumerate(test_queries, 1):
        thought=agent.think(query)
        action = agent.act(f"{query}관련작업")
        response=agent.respond(query)





