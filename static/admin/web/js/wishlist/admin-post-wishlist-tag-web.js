// 위시리스트 게시글 태그 수정 이벤트
const tagInput = document.querySelector(".admin-post-modal-place-input");
const tagWrap = document.querySelector(".update-tags-wrap");
const tag = document.querySelector(".create-tag-list");

tagInput.addEventListener("keyup", (e) => {
    if (e.keyCode === 13) {
        console.log("fsda");
        const tags = document.querySelectorAll(
            ".create-tag-list .tag-list-add"
        );

        if (e.target.value.length <= 10 && tags.length <= 4) {
            if (e.target.value) {
                const tagDiv = document.createElement("div");
                tagDiv.classList.add("tag-list-add");
                tagDiv.innerHTML = `<span>${e.target.value} ×</span>`;
                tag.appendChild(tagDiv);
                e.target.value = "";

                tagDiv.addEventListener("click", (e) => {
                    // 취소
                    e.stopPropagation();
                    tag.removeChild(tagDiv);
                });
            }
        } else {
        }
    }
});
