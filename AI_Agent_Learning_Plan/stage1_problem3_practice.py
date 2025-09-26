# 1단계 문제 3 연습: Memory 시스템 클래스
# 대화 기록을 저장하는 Memory 클래스를 구현해보세요.

"""
학습 목표:
- 클래스의 생성자(__init__) 구현
- 인스턴스 변수(attributes) 관리
- 매직 메서드들(__len__, __getitem__, __setitem__, __str__) 구현
- 리스트 조작 및 검색 기능 구현
- 타임스탬프 관리

구현할 기능:
1. Memory 클래스:
   - messages: 메시지들을 저장하는 리스트
   - max_size: 최대 저장 개수 제한
   - current_size: 현재 저장된 메시지 개수 (프로퍼티로 구현)

2. 메서드들:
   - add_message(role, content): 메시지 추가
   - get_messages(): 모든 메시지 반환
   - search(keyword): 키워드로 메시지 검색 (대소문자 구분 안함)
   - delete_message(index): 특정 인덱스의 메시지 삭제
   - clear(): 모든 메시지 삭제

3. 매직 메서드들:
   - __len__(): 저장된 메시지 개수 반환
   - __getitem__(index): 인덱스로 메시지 접근
   - __setitem__(index, value): 인덱스로 메시지 수정
   - __str__(): Memory 정보를 문자열로 반환

힌트:
- 메시지는 {"role": "user/assistant", "content": "메시지", "timestamp": "ISO 시간"}으로 저장
- datetime.now().isoformat()로 현재 시간을 ISO 형식으로 변환
- 최대 크기 초과 시 messages.pop(0)으로 가장 오래된 메시지 삭제
- search에서는 content.lower()와 keyword.lower()로 대소문자 무시
- 로깅은 print()로 간단히 구현
"""

from datetime import datetime
from typing import List, Dict, Optional, Any

class Memory:
    """대화 기록을 저장하고 관리하는 Memory 클래스"""

    def __init__(self, max_size: int = 100):
        """
        Memory 초기화

        Args:
            max_size: 최대 저장할 수 있는 메시지 개수
        """
        # TODO: 여기에 초기화 코드를 작성하세요
        # self.messages = ?
        # self.max_size = ?
        pass

    @property
    def current_size(self) -> int:
        """현재 저장된 메시지 개수를 반환하는 프로퍼티"""
        # TODO: 현재 메시지 개수 반환
        pass

    def add_message(self, role: str, content: str) -> None:
        """
        새로운 메시지를 추가

        Args:
            role: 메시지 역할 ("user" 또는 "assistant")
            content: 메시지 내용
        """
        # TODO: 메시지 딕셔너리 생성 (role, content, timestamp 포함)
        # TODO: 최대 크기 확인 후 필요시 가장 오래된 메시지 삭제
        # TODO: 새 메시지 추가
        # TODO: 로그 출력
        pass

    def get_messages(self) -> List[Dict[str, Any]]:
        """모든 메시지를 반환"""
        # TODO: self.messages 반환
        pass

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """
        키워드로 메시지 검색 (대소문자 구분 안함)

        Args:
            keyword: 검색할 키워드

        Returns:
            키워드가 포함된 메시지들의 리스트
        """
        # TODO: 리스트 컴프리헨션을 사용해서 keyword가 content에 포함된 메시지들 찾기
        # 힌트: keyword.lower() in message['content'].lower()
        pass

    def delete_message(self, index: int) -> None:
        """
        특정 인덱스의 메시지 삭제

        Args:
            index: 삭제할 메시지의 인덱스
        """
        # TODO: 인덱스가 유효한지 확인
        # TODO: 해당 인덱스의 메시지 삭제
        # TODO: 로그 출력
        pass

    def clear(self) -> None:
        """모든 메시지 삭제"""
        # TODO: messages 리스트 초기화
        # TODO: 로그 출력
        pass

    # 매직 메서드들
    def __len__(self) -> int:
        """len() 함수가 호출될 때 실행"""
        # TODO: 현재 메시지 개수 반환
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """memory[index] 형태로 접근할 때 실행"""
        # TODO: messages[index] 반환
        pass

    def __setitem__(self, index: int, value: Dict[str, Any]) -> None:
        """memory[index] = value 형태로 할당할 때 실행"""
        # TODO: messages[index] = value 설정
        pass

    def __str__(self) -> str:
        """str() 함수나 print()가 호출될 때 실행"""
        # TODO: Memory 상태 정보를 문자열로 반환
        # 예: "Memory(2/5 messages)"
        pass

# 테스트 코드 - 완성한 후에 실행해보세요
if __name__ == "__main__":
    print("=== Memory 클래스 연습 ===\n")

    # 1. Memory 생성 테스트
    print("1. Memory 생성")
    memory = Memory(max_size=3)
    print(f"생성된 Memory: {memory}")
    print(f"초기 크기: {len(memory)}")
    print()

    # 2. 메시지 추가 테스트
    print("2. 메시지 추가")
    memory.add_message("user", "안녕하세요!")
    memory.add_message("assistant", "안녕하세요! 도와드릴까요?")
    print(f"메시지 추가 후: {memory}")
    print()

    # 3. 메시지 조회 테스트
    print("3. 메시지 조회")
    messages = memory.get_messages()
    for i, msg in enumerate(messages):
        print(f"  {i}: [{msg['role']}] {msg['content']}")
    print()

    # 4. 인덱스 접근 테스트
    print("4. 인덱스 접근")
    print(f"첫 번째 메시지: {memory[0]['content']}")
    print(f"마지막 메시지: {memory[-1]['content']}")
    print()

    # 5. 검색 테스트
    print("5. 검색 테스트")
    results = memory.search("안녕")
    print(f"'안녕' 검색 결과: {len(results)}개")
    for result in results:
        print(f"  - [{result['role']}] {result['content']}")
    print()

    # 6. 최대 크기 테스트
    print("6. 최대 크기 초과 테스트")
    memory.add_message("user", "질문이 있어요")
    print(f"3개 추가 후: {memory}")
    memory.add_message("assistant", "네, 말씀하세요")  # 최대 크기 초과
    print(f"4개 추가 후(최대 3개): {memory}")
    print("남은 메시지들:")
    for i, msg in enumerate(memory.get_messages()):
        print(f"  {i}: [{msg['role']}] {msg['content']}")
    print()

    # 7. 메시지 수정 테스트
    print("7. 메시지 수정")
    print(f"수정 전: {memory[0]['content']}")
    memory[0] = {"role": "user", "content": "수정된 메시지", "timestamp": datetime.now().isoformat()}
    print(f"수정 후: {memory[0]['content']}")
    print()

    # 8. 메시지 삭제 테스트
    print("8. 메시지 삭제")
    print(f"삭제 전 크기: {len(memory)}")
    memory.delete_message(0)
    print(f"삭제 후 크기: {len(memory)}")
    print()

    # 9. 전체 삭제 테스트
    print("9. 전체 삭제")
    memory.clear()
    print(f"전체 삭제 후: {memory}")

    print("\n=== 연습 완료 ===")
    print("구현이 완료되면 모든 테스트가 정상 작동해야 합니다!")