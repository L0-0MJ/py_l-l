# 1단계 문제 2: Agent 상속 - 해답

from datetime import datetime
import time
from typing import List

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

class ChatAgent(Agent):
    """
    대화에 특화된 Agent 클래스

    기본 Agent 클래스를 상속받아 대화 기능을 추가한 클래스입니다.
    대화 기록을 저장하고 관리하는 기능을 제공합니다.
    """

    def __init__(self, name: str, role: str):
        """
        ChatAgent 초기화

        Args:
            name (str): Agent의 이름
            role (str): Agent의 역할
        """
        # 부모 클래스의 초기화 메서드 호출
        super().__init__(name, role)

        # ChatAgent만의 속성 추가
        self.conversation_history: List[dict] = []
        self._log_action("ChatAgent capabilities initialized")

    def chat(self, message: str) -> str:
        """
        대화를 위한 새로운 메서드

        Args:
            message (str): 사용자의 메시지

        Returns:
            str: 생성된 응답
        """
        self._log_action(f"Starting chat with message: {message}")

        # 사용자 메시지를 대화 기록에 저장
        user_entry = {
            "type": "user",
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.conversation_history.append(user_entry)

        # 응답 생성 (부모 클래스의 respond 메서드 활용)
        response = self.respond(message)

        # Agent 응답을 대화 기록에 저장
        agent_entry = {
            "type": "agent",
            "message": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.conversation_history.append(agent_entry)

        self._log_action(f"Chat completed. History length: {len(self.conversation_history)}")
        return response

    def respond(self, message: str) -> str:
        """
        부모 클래스의 respond 메서드를 오버라이딩
        대화 기록을 고려한 더 나은 응답을 생성합니다.

        Args:
            message (str): 응답할 메시지

        Returns:
            str: 생성된 응답
        """
        self._set_status("responding")
        self._log_action(f"ChatAgent responding to: {message}")

        # 대화 기록을 고려한 응답 생성
        conversation_count = len([entry for entry in self.conversation_history if entry["type"] == "user"])

        if conversation_count == 0:
            # 첫 대화
            if "안녕" in message:
                response = "안녕하세요! 저는 대화 전문 AI Agent입니다. 편하게 대화해요!"
            else:
                response = "처음 뵙겠습니다! 무엇을 도와드릴까요?"
        else:
            # 기존 대화가 있는 경우
            if "감사" in message or "고맙" in message:
                response = "도움이 되어서 기뻐요! 또 다른 질문이 있으시면 언제든 말씀해주세요."
            elif "?" in message:
                response = f"좋은 질문이네요! '{message}'에 대해 자세히 설명드리겠습니다."
            else:
                # 부모 클래스의 기본 응답 로직 사용
                response = super().respond(message)

        self._log_action(f"ChatAgent response: {response}")
        self._set_status("idle")
        return response

    def get_conversation_summary(self) -> str:
        """
        대화 요약을 반환하는 메서드

        Returns:
            str: 대화 요약
        """
        total_messages = len(self.conversation_history)
        user_messages = len([entry for entry in self.conversation_history if entry["type"] == "user"])
        agent_messages = len([entry for entry in self.conversation_history if entry["type"] == "agent"])

        return f"총 대화: {total_messages}개 (사용자: {user_messages}, Agent: {agent_messages})"

    def clear_history(self) -> None:
        """
        대화 기록을 초기화하는 메서드
        """
        old_count = len(self.conversation_history)
        self.conversation_history.clear()
        self._log_action(f"Conversation history cleared. {old_count} messages removed.")

class TaskAgent(Agent):
    """
    작업 수행에 특화된 Agent 클래스

    기본 Agent 클래스를 상속받아 작업 관리 기능을 추가한 클래스입니다.
    작업 큐를 관리하고 순차적으로 작업을 실행하는 기능을 제공합니다.
    """

    def __init__(self, name: str, role: str):
        """
        TaskAgent 초기화

        Args:
            name (str): Agent의 이름
            role (str): Agent의 역할
        """
        # 부모 클래스의 초기화 메서드 호출
        super().__init__(name, role)

        # TaskAgent만의 속성 추가
        self.task_queue: List[dict] = []
        self.completed_tasks: List[dict] = []
        self._log_action("TaskAgent capabilities initialized")

    def add_task(self, task: str, priority: str = "normal") -> None:
        """
        작업 큐에 새 작업을 추가하는 메서드

        Args:
            task (str): 추가할 작업
            priority (str): 작업 우선순위 (high, normal, low)
        """
        task_entry = {
            "task": task,
            "priority": priority,
            "added_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending"
        }

        # 우선순위에 따라 작업 큐에 삽입
        if priority == "high":
            self.task_queue.insert(0, task_entry)  # 맨 앞에 삽입
        else:
            self.task_queue.append(task_entry)  # 맨 뒤에 추가

        self._log_action(f"Task added: '{task}' (priority: {priority})")

    def execute_next_task(self) -> str:
        """
        작업 큐에서 다음 작업을 실행하는 메서드

        Returns:
            str: 작업 실행 결과
        """
        if not self.task_queue:
            result = "작업 큐가 비어있습니다."
            self._log_action(result)
            return result

        # 큐에서 첫 번째 작업 가져오기
        current_task = self.task_queue.pop(0)
        task_name = current_task["task"]

        self._log_action(f"Executing task: {task_name}")

        # act 메서드를 사용하여 작업 실행
        result = self.act(task_name)

        # 완료된 작업을 기록
        current_task["status"] = "completed"
        current_task["completed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_task["result"] = result
        self.completed_tasks.append(current_task)

        return result

    def act(self, action: str) -> str:
        """
        부모 클래스의 act 메서드를 오버라이딩
        작업 실행에 특화된 로직을 추가합니다.

        Args:
            action (str): 수행할 액션

        Returns:
            str: 액션 수행 결과
        """
        self._set_status("acting")
        self._log_action(f"TaskAgent performing: {action}")

        # 작업 유형에 따른 특화된 처리
        time.sleep(0.5)  # 작업 수행 시간 시뮬레이션

        if "분석" in action:
            result = f"데이터 분석 완료: {action}에 대한 상세 분석 결과를 생성했습니다."
        elif "보고서" in action:
            result = f"보고서 작성 완료: {action}에 대한 상세 보고서를 작성했습니다."
        elif "이메일" in action:
            result = f"이메일 발송 완료: {action} 관련 이메일을 성공적으로 발송했습니다."
        elif "검색" in action:
            result = f"정보 검색 완료: {action}에 대한 관련 정보를 수집했습니다."
        else:
            # 부모 클래스의 기본 act 로직 사용
            result = super().act(action)

        self._log_action(f"TaskAgent result: {result}")
        self._set_status("idle")
        return result

    def get_task_summary(self) -> str:
        """
        작업 상태 요약을 반환하는 메서드

        Returns:
            str: 작업 상태 요약
        """
        pending_tasks = len(self.task_queue)
        completed_tasks = len(self.completed_tasks)
        high_priority = len([task for task in self.task_queue if task["priority"] == "high"])

        return f"대기 작업: {pending_tasks}개 (고우선순위: {high_priority}개), 완료: {completed_tasks}개"

    def clear_completed_tasks(self) -> None:
        """
        완료된 작업 기록을 초기화하는 메서드
        """
        old_count = len(self.completed_tasks)
        self.completed_tasks.clear()
        self._log_action(f"Completed tasks cleared. {old_count} tasks removed.")

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 2: Agent 상속 테스트 ===\n")

    # ChatAgent 테스트
    print("--- ChatAgent 테스트 ---")
    chat_agent = ChatAgent("Bob", "chatbot")
    print(f"초기 상태: {chat_agent}")

    # 대화 테스트
    print("\n대화 시작:")
    chat_agent.chat("안녕하세요!")
    chat_agent.chat("오늘 날씨는 어때요?")
    chat_agent.chat("감사합니다!")

    print(f"\n대화 요약: {chat_agent.get_conversation_summary()}")
    print("대화 기록:")
    for i, entry in enumerate(chat_agent.conversation_history, 1):
        print(f"  {i}. [{entry['type']}] {entry['message']}")

    print("\n" + "="*50 + "\n")

    # TaskAgent 테스트
    print("--- TaskAgent 테스트 ---")
    task_agent = TaskAgent("Charlie", "task_executor")
    print(f"초기 상태: {task_agent}")

    # 작업 추가 테스트
    print("\n작업 추가:")
    task_agent.add_task("데이터 분석", "high")
    task_agent.add_task("보고서 작성", "normal")
    task_agent.add_task("이메일 발송", "normal")
    task_agent.add_task("긴급 검토", "high")

    print(f"작업 요약: {task_agent.get_task_summary()}")
    print("작업 큐:")
    for i, task in enumerate(task_agent.task_queue, 1):
        print(f"  {i}. [{task['priority']}] {task['task']}")

    # 작업 실행 테스트
    print("\n작업 실행:")
    task_agent.execute_next_task()
    task_agent.execute_next_task()

    print(f"\n작업 요약: {task_agent.get_task_summary()}")
    print("완료된 작업:")
    for task in task_agent.completed_tasks:
        print(f"  - {task['task']}: {task['result'][:50]}...")

    print("\n" + "="*50 + "\n")

    # 상속된 메서드들도 정상 동작하는지 확인
    print("--- 상속된 메서드 테스트 ---")
    print("ChatAgent의 think 메서드:")
    chat_agent.think("사용자가 무엇을 원하는지 파악해보자")

    print("\nTaskAgent의 respond 메서드:")
    task_agent.respond("작업이 완료되었습니다")

"""
학습 포인트:
1. 클래스 상속의 기본 개념과 super() 키워드 사용법
2. 메서드 오버라이딩과 부모 메서드 호출
3. 자식 클래스에서 새로운 속성과 메서드 추가
4. 부모 클래스의 기능을 유지하면서 확장하는 방법
5. 타입 힌팅을 사용한 리스트와 딕셔너리 선언
6. 다형성의 개념 (같은 메서드 이름으로 다른 동작)
7. 클래스별 특화된 로직 구현

실제 AI Agent 개발에서 상속은 매우 중요한 개념입니다.
기본 Agent 클래스를 만들고, 용도에 따라 특화된 Agent들을
상속으로 구현하는 것이 일반적인 패턴입니다.
"""