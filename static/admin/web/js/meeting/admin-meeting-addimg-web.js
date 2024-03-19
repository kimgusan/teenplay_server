const imageInput = document.getElementById("imageInput");
const mainPhotoImg = document.querySelector("#main-post-photo-img");

imageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            mainPhotoImg.src = e.target.result;
            mainPhotoImg.style.display = "block";
        };

        reader.readAsDataURL(file);
    }
});

const backimageInput = document.getElementById("backimageInput");
const backPhotoImg = document.querySelector("#main-post-photo-backimg");

backimageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            backPhotoImg.src = e.target.result;
            backPhotoImg.style.display = "block";
        };

        reader.readAsDataURL(file);
    }
});
