function deleteUserListItem(id) {
    // 해당 id를 가진 li 요소를 삭제
    var element = document.querySelector('[data-id="' + id + '"]');
    if (element) {
        element.remove();
        updateTotalCount();
    }
}

// 전체 숫자 업데이트 함수
function updateTotalCount() {
    // 전체 li 요소를 가져와서 갯수를 세어서 업데이트
    var totalCount = document.querySelectorAll(".main-user-list").length;
    document.querySelector(".main-user-total-number").innerText = totalCount;
}

// 삭제 버튼에 이벤트 추가
var deleteButtons = document.querySelectorAll(".main-user-list-button");
deleteButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        var id = this.getAttribute("data-target");
        deleteUserListItem(id);
    });
});

// 페이지 로드 시 초기 숫자 설정
updateTotalCount();
