# 1단계 문제 1: 기본 Agent 클래스 - 해답

from datetime import datetime
import time

class Agent:
    """
    기본 AI Agent 클래스

    AI Agent의 기본적인 구조를 구현한 클래스입니다.
    name, role, status 속성과 think, act, respond 메서드를 포함합니다.
    """

    def __init__(self, name: str, role: str):
        """
        Agent 초기화

        Args:
            name (str): Agent의 이름
            role (str): Agent의 역할
        """
        self.name = name
        self.role = role
        self.status = "idle"  # 초기 상태는 idle
        self._log_action(f"Agent {self.name} initialized with role: {self.role}")

    def _log_action(self, action: str) -> None:
        """
        로그를 출력하는 내부 메서드

        Args:
            action (str): 수행한 동작 내용
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.name} ({self.role}): {action}")

    def _set_status(self, new_status: str) -> None:
        """
        Agent의 상태를 변경하고 로그를 남기는 내부 메서드

        Args:
            new_status (str): 새로운 상태
        """
        old_status = self.status
        self.status = new_status
        if old_status != new_status:
            self._log_action(f"Status changed: {old_status} -> {new_status}")

    def think(self, query: str) -> str:
        """
        주어진 쿼리에 대해 생각하는 메서드

        Args:
            query (str): 생각할 쿼리

        Returns:
            str: 생각한 결과
        """
        self._set_status("thinking")
        self._log_action(f"Thinking about: {query}")

        # 실제 AI Agent에서는 여기서 복잡한 추론 과정이 일어남
        time.sleep(0.5)  # 생각하는 시간 시뮬레이션

        # 간단한 응답 생성 로직
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
        """
        주어진 액션을 수행하는 메서드

        Args:
            action (str): 수행할 액션

        Returns:
            str: 액션 수행 결과
        """
        self._set_status("acting")
        self._log_action(f"Performing action: {action}")

        # 액션 수행 시뮬레이션
        time.sleep(0.3)

        # 간단한 액션 처리 로직
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
        """
        메시지에 대한 응답을 생성하는 메서드

        Args:
            message (str): 응답할 메시지

        Returns:
            str: 생성된 응답
        """
        self._set_status("responding")
        self._log_action(f"Responding to: {message}")

        # 응답 생성 시뮬레이션
        time.sleep(0.2)

        # 간단한 응답 생성 로직
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

    def __str__(self) -> str:
        """
        Agent 객체의 문자열 표현

        Returns:
            str: Agent 정보
        """
        return f"Agent(name='{self.name}', role='{self.role}', status='{self.status}')"

    def __repr__(self) -> str:
        """
        Agent 객체의 개발자용 문자열 표현

        Returns:
            str: Agent 정보
        """
        return self.__str__()

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 1: 기본 Agent 클래스 테스트 ===\n")

    # Agent 인스턴스 생성
    agent = Agent("Alice", "assistant")
    print(f"Agent 정보: {agent}")
    print(f"초기 상태: {agent.status}\n")

    # think 메서드 테스트
    print("--- Think 메서드 테스트 ---")
    thought = agent.think("오늘 날씨가 어때?")
    print(f"생각 결과: {thought}\n")

    # act 메서드 테스트
    print("--- Act 메서드 테스트 ---")
    action_result = agent.act("날씨 정보 검색")
    print(f"행동 결과: {action_result}\n")

    # respond 메서드 테스트
    print("--- Respond 메서드 테스트 ---")
    response = agent.respond("안녕하세요!")
    print(f"응답 결과: {response}\n")

    print(f"최종 상태: {agent.status}")
    print(f"최종 Agent 정보: {agent}")

    print("\n=== 추가 테스트 ===")

    # 다양한 메시지로 테스트
    test_queries = [
        "내일 회의 일정이 어떻게 되나요?",
        "감사합니다!",
        "데이터 분석 좀 해주세요"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 테스트 {i} ---")
        thought = agent.think(query)
        action = agent.act(f"{query} 관련 작업")
        response = agent.respond(query)
        print(f"현재 상태: {agent.status}")

"""
학습 포인트:
1. 클래스 정의와 __init__ 메서드 사용법
2. 인스턴스 속성과 메서드 구현
3. 내부 메서드(_로 시작)를 사용한 코드 구조화
4. 상태 관리와 로깅 시스템
5. 문자열 포매팅과 시간 처리
6. __str__과 __repr__ 매직 메서드 활용
7. 타입 힌팅 (Type Hints) 사용법

이 클래스는 실제 AI Agent의 기본 구조를 보여주며,
향후 더 복잡한 기능들을 추가할 수 있는 기반을 제공합니다.
"""