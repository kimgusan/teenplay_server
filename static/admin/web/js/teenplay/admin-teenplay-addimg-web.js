// 파일 선택 시 실행되는 함수
document
    .getElementById("imageInput")
    .addEventListener("change", function (event) {
        var videoFile = event.target.files[0]; // 선택된 파일
        var videoObjectURL = URL.createObjectURL(videoFile); // 비디오 파일의 URL 생성

        // 비디오 요소 가져와서 소스 변경
        var videoElement = document.getElementById("main-post-photo-img");
        videoElement.src = videoObjectURL;
        videoElement.controls = true; // 비디오 컨트롤러 추가
    });
