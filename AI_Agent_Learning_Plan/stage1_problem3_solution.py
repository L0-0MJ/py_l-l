# 1단계 문제 3: Memory 시스템 클래스 - 해답

from datetime import datetime
from typing import List, Dict, Optional, Any

class Memory:
    """
    대화 기록을 저장하고 관리하는 Memory 클래스

    AI Agent의 메모리 시스템을 구현한 클래스로,
    대화 기록을 효율적으로 저장, 검색, 관리하는 기능을 제공합니다.
    """

    def __init__(self, max_size: int = 100):
        """
        Memory 초기화

        Args:
            max_size (int): 최대 저장 가능한 메시지 수
        """
        self.max_size = max_size
        self.messages: List[Dict[str, Any]] = []
        self._log(f"Memory initialized with max_size: {max_size}")

    def _log(self, message: str) -> None:
        """
        Memory 시스템 내부 로깅

        Args:
            message (str): 로그 메시지
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Memory: {message}")

    def _create_message(self, role: str, content: str) -> Dict[str, Any]:
        """
        메시지 객체를 생성하는 내부 메서드

        Args:
            role (str): 메시지 역할 (user, assistant, system)
            content (str): 메시지 내용

        Returns:
            Dict[str, Any]: 메시지 딕셔너리
        """
        return {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.messages)  # 간단한 ID 생성
        }

    def add_message(self, role: str, content: str) -> None:
        """
        새 메시지를 추가합니다

        Args:
            role (str): 메시지 역할
            content (str): 메시지 내용
        """
        message = self._create_message(role, content)
        self.messages.append(message)

        # 최대 크기 초과 시 가장 오래된 메시지 삭제
        if len(self.messages) > self.max_size:
            removed_message = self.messages.pop(0)
            self._log(f"Max size exceeded. Removed oldest message: {removed_message['content'][:30]}...")

        self._log(f"Message added - [{role}]: {content[:50]}{'...' if len(content) > 50 else ''}")

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        모든 메시지를 반환합니다

        Returns:
            List[Dict[str, Any]]: 모든 메시지 리스트
        """
        return self.messages.copy()  # 원본 보호를 위해 복사본 반환

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """
        키워드로 메시지를 검색합니다

        Args:
            keyword (str): 검색할 키워드

        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
        """
        keyword_lower = keyword.lower()
        results = [
            message for message in self.messages
            if keyword_lower in message['content'].lower()
        ]

        self._log(f"Search for '{keyword}' returned {len(results)} results")
        return results

    def delete_message(self, index: int) -> bool:
        """
        특정 인덱스의 메시지를 삭제합니다

        Args:
            index (int): 삭제할 메시지의 인덱스

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if -len(self.messages) <= index < len(self.messages):
                # 음수 인덱스 지원
                actual_index = index if index >= 0 else len(self.messages) + index
                deleted_message = self.messages.pop(actual_index)
                self._log(f"Message deleted at index {index}: {deleted_message['content'][:30]}...")
                return True
            else:
                self._log(f"Delete failed: Invalid index {index}")
                return False
        except Exception as e:
            self._log(f"Delete failed: {str(e)}")
            return False

    def clear(self) -> None:
        """
        모든 메시지를 삭제합니다
        """
        message_count = len(self.messages)
        self.messages.clear()
        self._log(f"All messages cleared. {message_count} messages removed.")

    @property
    def current_size(self) -> int:
        """
        현재 저장된 메시지 개수를 반환합니다

        Returns:
            int: 현재 메시지 개수
        """
        return len(self.messages)

    # 매직 메서드들
    def __len__(self) -> int:
        """
        len() 함수 지원 - 저장된 메시지 개수 반환

        Returns:
            int: 메시지 개수
        """
        return len(self.messages)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        인덱스를 사용한 메시지 접근 지원

        Args:
            index (int): 접근할 인덱스

        Returns:
            Dict[str, Any]: 메시지

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
        """
        return self.messages[index]

    def __setitem__(self, index: int, value: Dict[str, Any]) -> None:
        """
        인덱스를 사용한 메시지 수정 지원

        Args:
            index (int): 수정할 인덱스
            value (Dict[str, Any]): 새 메시지 값

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            TypeError: value가 딕셔너리가 아닌 경우
        """
        if not isinstance(value, dict):
            raise TypeError("Message must be a dictionary")

        if 'role' not in value or 'content' not in value:
            raise ValueError("Message must have 'role' and 'content' keys")

        # timestamp가 없으면 현재 시간으로 설정
        if 'timestamp' not in value:
            value['timestamp'] = datetime.now().isoformat()

        old_content = self.messages[index]['content']
        self.messages[index] = value
        self._log(f"Message updated at index {index}: '{old_content[:30]}...' -> '{value['content'][:30]}...'")

    def __str__(self) -> str:
        """
        Memory 정보를 문자열로 반환

        Returns:
            str: Memory 상태 문자열
        """
        return f"Memory(size={len(self.messages)}/{self.max_size}, messages={len(self.messages)})"

    def __repr__(self) -> str:
        """
        개발자용 문자열 표현

        Returns:
            str: Memory 상태 문자열
        """
        return f"Memory(max_size={self.max_size}, current_size={len(self.messages)})"

    def __iter__(self):
        """
        이터레이터 지원 - for문에서 사용 가능

        Returns:
            iterator: 메시지 이터레이터
        """
        return iter(self.messages)

    def __contains__(self, keyword: str) -> bool:
        """
        'in' 연산자 지원 - 키워드가 포함된 메시지가 있는지 확인

        Args:
            keyword (str): 찾을 키워드

        Returns:
            bool: 키워드가 포함된 메시지가 있는지 여부
        """
        keyword_lower = keyword.lower()
        return any(keyword_lower in message['content'].lower() for message in self.messages)

    # 추가 유틸리티 메서드들
    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        특정 역할의 메시지들만 반환

        Args:
            role (str): 찾을 역할

        Returns:
            List[Dict[str, Any]]: 해당 역할의 메시지들
        """
        return [message for message in self.messages if message['role'] == role]

    def get_recent_messages(self, count: int) -> List[Dict[str, Any]]:
        """
        최근 n개의 메시지를 반환

        Args:
            count (int): 반환할 메시지 개수

        Returns:
            List[Dict[str, Any]]: 최근 메시지들
        """
        return self.messages[-count:] if count > 0 else []

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        대화 통계 정보를 반환

        Returns:
            Dict[str, Any]: 통계 정보
        """
        roles = {}
        total_chars = 0

        for message in self.messages:
            role = message['role']
            content_length = len(message['content'])

            roles[role] = roles.get(role, 0) + 1
            total_chars += content_length

        return {
            'total_messages': len(self.messages),
            'messages_by_role': roles,
            'total_characters': total_chars,
            'average_message_length': total_chars / len(self.messages) if self.messages else 0,
            'memory_usage': f"{len(self.messages)}/{self.max_size} ({len(self.messages)/self.max_size*100:.1f}%)"
        }

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 3: Memory 시스템 클래스 테스트 ===\n")

    # Memory 인스턴스 생성
    memory = Memory(max_size=5)
    print(f"Memory 생성: {memory}")
    print(f"Memory 상세: {repr(memory)}")
    print(f"초기 크기: {len(memory)}\n")

    # 메시지 추가 테스트
    print("--- 메시지 추가 테스트 ---")
    memory.add_message("user", "안녕하세요!")
    memory.add_message("assistant", "안녕하세요! 무엇을 도와드릴까요?")
    memory.add_message("user", "오늘 날씨가 어때요?")
    memory.add_message("assistant", "오늘은 맑고 따뜻한 날씨입니다.")
    print(f"메시지 추가 후 크기: {len(memory)}\n")

    # 메시지 조회 테스트
    print("--- 메시지 조회 테스트 ---")
    messages = memory.get_messages()
    for i, msg in enumerate(messages):
        print(f"{i}: [{msg['role']}] {msg['content']} (ID: {msg.get('id', 'N/A')})")
    print()

    # 인덱스 접근 테스트
    print("--- 인덱스 접근 테스트 ---")
    print(f"첫 번째 메시지: [{memory[0]['role']}] {memory[0]['content']}")
    print(f"마지막 메시지: [{memory[-1]['role']}] {memory[-1]['content']}\n")

    # 검색 테스트
    print("--- 검색 테스트 ---")
    search_results = memory.search("날씨")
    print(f"'날씨' 검색 결과: {len(search_results)}개")
    for result in search_results:
        print(f"  - [{result['role']}] {result['content']}")

    # 'in' 연산자 테스트
    print(f"'안녕' 키워드 포함 여부: {'안녕' in memory}")
    print(f"'컴퓨터' 키워드 포함 여부: {'컴퓨터' in memory}\n")

    # 최대 크기 초과 테스트
    print("--- 최대 크기 초과 테스트 ---")
    memory.add_message("user", "감사합니다!")
    memory.add_message("assistant", "천만에요!")
    memory.add_message("user", "또 질문이 있어요")  # 최대 크기 초과
    print(f"크기 초과 후 메시지 개수: {len(memory)}\n")

    # 메시지 수정 테스트
    print("--- 메시지 수정 테스트 ---")
    print(f"수정 전: [{memory[0]['role']}] {memory[0]['content']}")
    memory[0] = {
        "role": "user",
        "content": "수정된 첫 번째 메시지",
        "timestamp": datetime.now().isoformat()
    }
    print(f"수정 후: [{memory[0]['role']}] {memory[0]['content']}\n")

    # 역할별 메시지 조회 테스트
    print("--- 역할별 메시지 조회 ---")
    user_messages = memory.get_messages_by_role("user")
    print(f"사용자 메시지: {len(user_messages)}개")
    for msg in user_messages:
        print(f"  - {msg['content']}")
    print()

    # 최근 메시지 조회 테스트
    print("--- 최근 메시지 조회 ---")
    recent = memory.get_recent_messages(3)
    print(f"최근 3개 메시지:")
    for i, msg in enumerate(recent):
        print(f"  {i+1}. [{msg['role']}] {msg['content']}")
    print()

    # 통계 정보 테스트
    print("--- 대화 통계 ---")
    stats = memory.get_conversation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # 이터레이터 테스트
    print("--- 이터레이터 테스트 (for문) ---")
    for i, message in enumerate(memory):
        print(f"  {i}: [{message['role']}] {message['content'][:30]}...")
    print()

    # 메시지 삭제 테스트
    print("--- 메시지 삭제 테스트 ---")
    print(f"삭제 전 크기: {len(memory)}")
    success = memory.delete_message(0)
    print(f"삭제 성공: {success}, 삭제 후 크기: {len(memory)}")

    # 잘못된 인덱스 삭제 테스트
    success = memory.delete_message(100)
    print(f"잘못된 인덱스 삭제 시도 결과: {success}\n")

    # 전체 삭제 테스트
    print("--- 전체 삭제 테스트 ---")
    memory.clear()
    print(f"전체 삭제 후 크기: {len(memory)}")
    print(f"Memory 상태: {memory}")

"""
학습 포인트:
1. 매직 메서드(__len__, __getitem__, __setitem__, __str__ 등) 구현
2. 프로퍼티(@property) 사용법
3. 타입 힌팅을 사용한 복잡한 데이터 타입 선언
4. 리스트 컴프리헨션을 활용한 효율적인 데이터 필터링
5. 예외 처리와 에러 검증
6. 이터레이터와 컨테이너 프로토콜 구현
7. 클래스 내부 메서드(_로 시작)를 사용한 코드 구조화
8. 메모리 관리와 자동 정리 로직
9. 다양한 유틸리티 메서드 구현

이 Memory 클래스는 실제 AI Agent에서 사용되는 메모리 시스템의
기본 구조를 보여주며, 실무에서 필요한 다양한 기능들을 포함합니다.
"""