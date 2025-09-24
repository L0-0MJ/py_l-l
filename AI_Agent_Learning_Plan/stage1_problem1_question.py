# 1단계 문제 1: 기본 Agent 클래스
# AI Agent 개발의 기초가 되는 Agent 클래스를 구현해보세요.

"""
문제 요구사항:
1. Agent 클래스를 정의하세요
2. 다음 속성들을 포함해야 합니다:
   - name: Agent의 이름
   - role: Agent의 역할 (예: "assistant", "chatbot", "analyzer")
   - status: Agent의 현재 상태 (예: "idle", "thinking", "acting")

3. 다음 메서드들을 구현해야 합니다:
   - think(query): 주어진 쿼리에 대해 생각하는 메서드
   - act(action): 주어진 액션을 수행하는 메서드
   - respond(message): 메시지에 대한 응답을 생성하는 메서드

4. 각 메서드 실행 시 로그를 출력해야 합니다:
   - 로그 형식: "[시간] Agent명 (역할): 동작 내용"
   - 예: "[2024-01-01 10:30:00] Alice (assistant): Thinking about: 날씨가 어때?"

5. Agent의 상태를 적절히 변경해야 합니다:
   - think() 실행 중: status = "thinking"
   - act() 실행 중: status = "acting"
   - respond() 실행 중: status = "responding"
   - 메서드 완료 후: status = "idle"

힌트:
- datetime 모듈을 사용하여 현재 시간을 가져올 수 있습니다
- __init__ 메서드에서 초기값을 설정하세요
- 각 메서드에서 적절한 반환값을 만들어보세요
"""

from datetime import datetime

class Agent:
    """
    기본 AI Agent 클래스
    여기에 코드를 작성하세요.
    """
    pass

# 테스트 코드
if __name__ == "__main__":
    # Agent 인스턴스 생성
    agent = Agent("Alice", "assistant")

    # 메서드 테스트
    print("=== Agent 테스트 ===")
    print(f"Agent 정보: {agent.name} ({agent.role})")
    print(f"초기 상태: {agent.status}")
    print()

    # think 메서드 테스트
    thought = agent.think("오늘 날씨가 어때?")
    print(f"생각 결과: {thought}")
    print()

    # act 메서드 테스트
    action_result = agent.act("날씨 정보 검색")
    print(f"행동 결과: {action_result}")
    print()

    # respond 메서드 테스트
    response = agent.respond("안녕하세요!")
    print(f"응답 결과: {response}")
    print()

    print(f"최종 상태: {agent.status}")