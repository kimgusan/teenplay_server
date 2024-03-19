// 기존 선택한 카테고리를 표시
const options = document.querySelectorAll("option")
options.forEach((option) => {
    if (clubPostCategory === option.value){
        option.selected = true
    }
})

//
fileSizeInfo.innerText = getFileSizeWithExtension(getFileSize);
fileNameInfo.innerText = getImagePath.split("/").at(-1)
uploadedImageInfo.style.display = "block";

checkValuesAndValidateButton()