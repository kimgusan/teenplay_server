const imageInput = document.getElementById("imageInput");
const mainPhotoImg = document.querySelector("#main-festival-photo-img");
const mainPhotoSvg = document.querySelector("#main-festival-photo-svg");
const mainPhotoPath = document.querySelector("#main-festival-photo-path");

imageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            mainPhotoImg.src = e.target.result;
            mainPhotoImg.style.display = "block";
            mainPhotoImg.classList.remove("hidden");
            mainPhotoSvg.classList.add("hidden");
            mainPhotoPath.classList.add("hidden");
        };

        reader.readAsDataURL(file);
    }
});
