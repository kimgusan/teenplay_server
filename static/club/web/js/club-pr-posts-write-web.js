// 이미지 업로드 옆 아이콘에 마우스 올릴 시 모달창 출력
const infoIcon = document.querySelector(".pr-write-image-modal-btn");
const writeInfoModal = document.querySelector(".pr-write-image-modal-container");

infoIcon.addEventListener("mouseover", () => {
    writeInfoModal.style.display = "block";
});
infoIcon.addEventListener("mouseout", () => {
    writeInfoModal.style.display = "none";
});

// 이미지 없이 작성완료 클릭 시 막아주기
const writeBtn = document.querySelector(".pr-write-button");
const imgInput = document.querySelector(".pr-write-image-hidden");
const errorMsg = document.querySelector(".pr-write-error");

writeBtn.addEventListener("click", (e) => {
    if (!fileNameInfo.innerText) {
        errorMsg.style.display = "block";
        e.preventDefault();
    }
});

// 카테고리, 제목, 내용, 이미지 다 입력했을 때에만 작성완료 버튼 활성화
// 이미지 부분은 아래에 있습니다.
const categorySelect = document.querySelector(".pr-write-fields");
const titleInput = document.querySelector(".pr-write-title");
const contentInput = document.querySelector(".pr-write-content");
let tempDraggedFile = false;

function checkValuesAndValidateButton() {
    if (!categorySelect.value || !titleInput.value || !contentInput.value || !fileNameInfo.innerText) {
        writeBtn.disabled = true;
        return;
    }
    writeBtn.disabled = false;
}

categorySelect.addEventListener("change", () => {
    checkValuesAndValidateButton();
});
titleInput.addEventListener("keyup", () => {
    checkValuesAndValidateButton();
});
contentInput.addEventListener("keyup", () => {
    checkValuesAndValidateButton();
});

// 파일 선택 버튼 클릭 시 첨부 가능하도록
const imgUploadBtn = document.querySelector(".pr-write-box-button");

imgUploadBtn.addEventListener("click", () => {
    imgInput.click();
});

// 파일 첨부 시 용량(10MB 이하) 체크하여 클 시 에러메시지 출력 및
// 용량 문제 없을 시 파일 정보를 아래에 표시
const sizeErrorMsg = document.querySelector(".pr-write-size-error");
const fileSizeInfo = document.querySelector(".pr-file-size");
const fileNameInfo = document.querySelector(".pr-file-name");
const uploadedImageInfo = document.querySelector(".pr-write-uploaded-image-wrap");
const MAX_SIZE = 10; // 10mb

function checkFileSize(obj, size) {
    let check = false;
    let sizeInBytes = obj.files[0].size;
    if (sizeInBytes > size) {
        check = false;
    } else {
        check = true;
    }
    return check;
}

function getFileSizeWithExtension(sizeInBytes) {
    let fileSizeExt = ["bytes", "kb", "mb", "gb"];
    let i = 0;
    let checkSize = sizeInBytes;
    while (checkSize > 900) {
        checkSize /= 1024;
        i++;
    }
    checkSize = Math.round(checkSize * 100) / 100 + "" + fileSizeExt[i];
    return checkSize;
}

imgInput.addEventListener("change", (e) => {
    if (e.target.value) {
        let checkSize = 1024 * 1024 * MAX_SIZE;
        if (!checkFileSize(imgInput, checkSize)) {
            sizeErrorMsg.style.display = "block";
            e.preventDefault();
            return;
        }
        sizeErrorMsg.style.display = "none";
        fileSize = e.target.files[0].size;
        fileSizeInfo.innerText = getFileSizeWithExtension(fileSize);
        fileNameInfo.innerText = e.target.files[0].name;
        uploadedImageInfo.style.display = "block";
    }
    checkValuesAndValidateButton();
});


// 파일 첨부 후 x버튼으로 삭제하기
const fileRemoveBtn = document.querySelector(".pr-remove-btn");
fileRemoveBtn.addEventListener("click", (e) => {
    e.preventDefault();
    imgInput.value = "";
    uploadedImageInfo.style.display = "none";
    fileSizeInfo.innerText = "";
    fileNameInfo.innerText = "";
    checkValuesAndValidateButton();
});


// 드래그 앤 드롭으로 이미지 첨부하기
// const dragDropBox = document.querySelector(".pr-write-box-wrap");
//
// dragDropBox.addEventListener("dragenter", (e) => {
//     e.preventDefault();
// });
// dragDropBox.addEventListener("dragover", (e) => {
//     e.preventDefault();
// });
// dragDropBox.addEventListener("dragleave", (e) => {
//     e.preventDefault();
// });
// dragDropBox.addEventListener("drop", (e) => {
//     e.preventDefault();
//
//     console.log("들어옴")
//     let file = e.dataTransfer;
//     if (!checkFileSize(file, 1024 * 1024 * MAX_SIZE)) {
//         sizeErrorMsg.style.display = "block";
//         return;
//     }
//     sizeErrorMsg.style.display = "none";
//     fileSize = file.files[0].size;
//     fileSizeInfo.innerText = getFileSizeWithExtension(fileSize);
//     fileNameInfo.innerText = file.files[0].name;
//     uploadedImageInfo.style.display = "block";
//     tempDraggedFile = true;
//     checkValuesAndValidateButton();
// });

// 전부 다 입력 후 서버쪽 작성완료 구현은 아래 주석한 것처럼 하면 됩니다.

// writeBtn.addEventListener("click", () => {
//     if (조건이 다 맞는다면) {
//         document.write.submit(); // form 태그를 submit()시킵니다.
//     }
// })
