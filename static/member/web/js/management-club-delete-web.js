const clubDeleteBtn = document.querySelector(".club-delete-btn");
const deleteClubForm = document.getElementById("delete-club-form");
clubDeleteBtn.addEventListener("click", async () => {
    let returnValue = confirm("정말 삭제 하시겠습니까?");
    if (returnValue) {
        deleteClubForm.submit();
    }
});