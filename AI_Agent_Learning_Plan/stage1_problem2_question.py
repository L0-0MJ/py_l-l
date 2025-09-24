# 1단계 문제 2: Agent 상속
# 기본 Agent 클래스를 상속받아 특화된 Agent들을 만들어보세요.

"""
문제 요구사항:
1. 이전에 만든 Agent 클래스를 기반으로 상속을 구현하세요
2. ChatAgent 클래스를 만드세요:
   - 대화에 특화된 Agent
   - conversation_history 속성 추가 (리스트)
   - chat(message) 메서드 추가
   - respond() 메서드를 오버라이딩하여 대화 기록을 저장

3. TaskAgent 클래스를 만드세요:
   - 작업 수행에 특화된 Agent
   - task_queue 속성 추가 (리스트)
   - add_task(task) 메서드 추가
   - execute_next_task() 메서드 추가
   - act() 메서드를 오버라이딩하여 작업 큐에서 작업을 처리

4. 각 클래스는 부모 클래스의 기능을 유지하면서 추가 기능을 제공해야 합니다

힌트:
- super() 키워드를 사용하여 부모 클래스의 메서드를 호출하세요
- 자식 클래스에서 __init__ 메서드를 오버라이딩할 때 super().__init__()을 호출하세요
- 메서드 오버라이딩 시 기본 동작은 유지하면서 추가 기능을 구현하세요
"""

from datetime import datetime
import time

# 기본 Agent 클래스 (문제 1에서 구현한 클래스)
class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = "idle"
        self._log_action(f"Agent {self.name} initialized with role: {self.role}")

    def _log_action(self, action: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.name} ({self.role}): {action}")

    def _set_status(self, new_status: str) -> None:
        old_status = self.status
        self.status = new_status
        if old_status != new_status:
            self._log_action(f"Status changed: {old_status} -> {new_status}")

    def think(self, query: str) -> str:
        self._set_status("thinking")
        self._log_action(f"Thinking about: {query}")
        time.sleep(0.5)

        if "날씨" in query:
            thought = "날씨 정보를 확인하기 위해 외부 API를 호출해야겠다"
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
        self._log_action(f"Responding to: {message}")
        time.sleep(0.2)

        if "안녕" in message:
            response = "안녕하세요! 저는 AI Agent입니다. 무엇을 도와드릴까요?"
        elif "고맙" in message or "감사" in message:
            response = "천만에요! 언제든지 도움이 필요하시면 말씀해주세요."
        elif "?" in message:
            response = f"'{message}'에 대한 답변을 드리겠습니다. 조금만 기다려주세요."
        else:
            response = f"'{message}' 메시지를 잘 받았습니다. 적절한 응답을 준비하겠습니다."

        self._log_action(f"Response generated: {response}")
        self._set_status("idle")
        return response

# 여기에 ChatAgent 클래스를 구현하세요
class ChatAgent:
    """
    대화에 특화된 Agent 클래스
    여기에 코드를 작성하세요.
    """
    pass

# 여기에 TaskAgent 클래스를 구현하세요
class TaskAgent:
    """
    작업 수행에 특화된 Agent 클래스
    여기에 코드를 작성하세요.
    """
    pass

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 2: Agent 상속 테스트 ===\n")

    # ChatAgent 테스트
    print("--- ChatAgent 테스트 ---")
    chat_agent = ChatAgent("Bob", "chatbot")

    # 대화 테스트
    chat_agent.chat("안녕하세요!")
    chat_agent.chat("오늘 날씨는 어때요?")
    chat_agent.chat("감사합니다!")

    print(f"대화 기록: {chat_agent.conversation_history}")
    print()

    # TaskAgent 테스트
    print("--- TaskAgent 테스트 ---")
    task_agent = TaskAgent("Charlie", "task_executor")

    # 작업 추가 테스트
    task_agent.add_task("데이터 분석")
    task_agent.add_task("보고서 작성")
    task_agent.add_task("이메일 발송")

    print(f"작업 큐: {task_agent.task_queue}")

    # 작업 실행 테스트
    task_agent.execute_next_task()
    task_agent.execute_next_task()

    print(f"남은 작업: {task_agent.task_queue}")
    print()

    # 상속된 메서드들도 정상 동작하는지 확인
    print("--- 상속된 메서드 테스트 ---")
    chat_agent.think("사용자가 무엇을 원하는지 파악해보자")
    task_agent.respond("작업이 완료되었습니다")