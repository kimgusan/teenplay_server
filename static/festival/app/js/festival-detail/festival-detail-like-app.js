const likeButtons = document.querySelectorAll(
    ".main-festival-banner-ad-like-button"
);

likeButtons.forEach(function (likeButton, index) {
    const likeStatus = document.querySelectorAll(".unlike-status")[index];
    const unlikeStatus = document.querySelectorAll(".like-status")[index];

    likeButton.addEventListener("click", function () {
        if (likeStatus.style.display === "none") {
            likeStatus.style.display = "inline-block";
            unlikeStatus.style.display = "none";
        } else {
            likeStatus.style.display = "none";
            unlikeStatus.style.display = "inline-block";
        }
    });
});
