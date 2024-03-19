// 전체 숫자 업데이트 함수
function updateTotalCount() {
    // 전체 li 요소를 가져와서 갯수를 세어서 업데이트
    var totalCount = document.querySelectorAll(".main-message-list").length;
    document.querySelector(".main-message-total-number").innerText = totalCount;
}

// 페이지 로드 시 초기 숫자 설정
updateTotalCount();
