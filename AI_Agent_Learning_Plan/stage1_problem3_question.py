# 1단계 문제 3: Memory 시스템 클래스
# 대화 기록을 저장하는 Memory 클래스를 만들어보세요.

"""
문제 요구사항:
1. Memory 클래스를 만드세요:
   - messages 속성: 메시지들을 저장하는 리스트
   - max_size 속성: 최대 저장 개수 제한
   - current_size 속성: 현재 저장된 메시지 개수

2. 다음 메서드들을 구현하세요:
   - add_message(role, content): 메시지 추가
   - get_messages(): 모든 메시지 반환
   - search(keyword): 키워드로 메시지 검색
   - delete_message(index): 특정 인덱스의 메시지 삭제
   - clear(): 모든 메시지 삭제

3. 다음 매직 메서드들을 구현하세요:
   - __len__(): 저장된 메시지 개수 반환
   - __getitem__(index): 인덱스로 메시지 접근
   - __setitem__(index, value): 인덱스로 메시지 수정
   - __str__(): Memory 정보를 문자열로 반환

4. 추가 기능:
   - 최대 크기를 초과하면 가장 오래된 메시지 자동 삭제
   - 메시지 추가/삭제 시 로깅
   - 메시지 검색 시 대소문자 구분하지 않음

힌트:
- 메시지는 딕셔너리 형태로 저장하세요: {"role": "user", "content": "메시지", "timestamp": "시간"}
- datetime 모듈을 사용하여 타임스탬프를 추가하세요
- 리스트의 슬라이싱을 활용하여 최대 크기를 관리하세요
- search 메서드에서는 리스트 컴프리헨션을 사용해보세요
"""

from datetime import datetime
from typing import List, Dict, Optional, Any

class Memory:
    """
    대화 기록을 저장하고 관리하는 Memory 클래스
    여기에 코드를 작성하세요.
    """
    pass

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 3: Memory 시스템 클래스 테스트 ===\n")

    # Memory 인스턴스 생성
    memory = Memory(max_size=5)
    print(f"Memory 생성: {memory}")
    print(f"초기 크기: {len(memory)}")
    print()

    # 메시지 추가 테스트
    print("--- 메시지 추가 테스트 ---")
    memory.add_message("user", "안녕하세요!")
    memory.add_message("assistant", "안녕하세요! 무엇을 도와드릴까요?")
    memory.add_message("user", "오늘 날씨가 어때요?")
    memory.add_message("assistant", "오늘은 맑고 따뜻한 날씨입니다.")
    print(f"메시지 추가 후 크기: {len(memory)}")
    print()

    # 메시지 조회 테스트
    print("--- 메시지 조회 테스트 ---")
    messages = memory.get_messages()
    for i, msg in enumerate(messages):
        print(f"{i}: [{msg['role']}] {msg['content']}")
    print()

    # 인덱스 접근 테스트
    print("--- 인덱스 접근 테스트 ---")
    print(f"첫 번째 메시지: {memory[0]}")
    print(f"마지막 메시지: {memory[-1]}")
    print()

    # 검색 테스트
    print("--- 검색 테스트 ---")
    search_results = memory.search("날씨")
    print(f"'날씨' 검색 결과: {len(search_results)}개")
    for result in search_results:
        print(f"  - [{result['role']}] {result['content']}")
    print()

    # 최대 크기 초과 테스트
    print("--- 최대 크기 초과 테스트 ---")
    memory.add_message("user", "감사합니다!")
    memory.add_message("assistant", "천만에요!")
    memory.add_message("user", "또 질문이 있어요")  # 최대 크기 초과
    print(f"크기 초과 후 메시지 개수: {len(memory)}")

    # 메시지 수정 테스트
    print("--- 메시지 수정 테스트 ---")
    print(f"수정 전: {memory[0]}")
    memory[0] = {"role": "user", "content": "수정된 메시지", "timestamp": datetime.now().isoformat()}
    print(f"수정 후: {memory[0]}")
    print()

    # 메시지 삭제 테스트
    print("--- 메시지 삭제 테스트 ---")
    memory.delete_message(0)
    print(f"삭제 후 크기: {len(memory)}")

    # 전체 삭제 테스트
    print("--- 전체 삭제 테스트 ---")
    memory.clear()
    print(f"전체 삭제 후 크기: {len(memory)}")
    print(f"Memory 상태: {memory}")