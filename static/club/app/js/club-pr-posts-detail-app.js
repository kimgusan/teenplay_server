// 위시리스트 댓글 메뉴 열고 닫기 이벤트
const wishlistCommentMenuButton = document.querySelector(".comment-menu");
const wishlistCommentMenu = document.querySelector(".comment-menu-open-wrap");
const commentModalWrap = document.querySelector(".comment-modal-wrap");
const commentModalBg = document.querySelector(".comment-modal-bg");
const commentModalContainer = document.querySelector(".comment-modal-container");
const commentMenuOpenContainer = document.querySelector(".comment-menu-open-container");

wishlistCommentMenuButton.addEventListener("click", () => {
    commentModalWrap.classList.toggle("hidden");
});

// 위시리스트 댓글 메뉴 닫기 이벤트
document.addEventListener("click", (e) => {
    if (!wishlistCommentMenuButton.contains(e.target) && !commentMenuOpenContainer.contains(e.target)) {
        commentModalWrap.classList.add("hidden");
    }
});

// 댓글 수정 이벤트
const commentMenuOpenUpdate = document.getElementById("comment-menu-open-update");
const commentInputUpdate = document.querySelector(".comment-update-box-all-wrap");
const commentComment = document.querySelector(".comment-list-all-wrap");

commentMenuOpenUpdate.addEventListener("click", () => {
    commentInputUpdate.classList.remove("hidden");
    commentComment.classList.add("hidden");
    commentModalWrap.classList.add("hidden");
});

const commentUploadFinish = document.getElementById("comment-update-upload");

commentUploadFinish.addEventListener("click", () => {
    commentInputUpdate.classList.add("hidden");
    commentComment.classList.remove("hidden");
});

// 댓글 삭제 이벤트
const commentMenuOpenDelete = document.getElementById("comment-menu-open-delete");
commentMenuOpenDelete.addEventListener("click", () => {
    commentModalWrap.classList.add("hidden");
});
