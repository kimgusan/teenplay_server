// 서버에서 가져오는 방식?

// function addFestivalsFromServer() {
//     fetch('https://example.com/festivals')
//     .then(response => response.json())
//     .then(data => {
//         // 받아온 데이터를 기반으로 축제 요소를 생성하고 추가하는 로직을 작성합니다.
//         data.forEach(festival => {
//             var festivalDiv = document.createElement('div');
//             festivalDiv.classList.add('main-festival-main-warp');

//             // 축제 정보에 따라 각 요소를 동적으로 생성하고 추가합니다.
//             // 예를 들어, festival.name, festival.image_url, festival.date 등을 사용할 수 있습니다.

//             festivalContainer.appendChild(festivalDiv);
//         });
//     })
//     .catch(error => {
//         console.error('Error fetching festivals:', error);
//     });
// }

function addFestivals() {
    var festivalContainer = document.querySelector(".main-festival-main");

    // 한 번에 추가할 축제 수
    var festivalsToAdd = 12;

    for (var i = 0; i < festivalsToAdd; i++) {
        // 새로운 축제 요소 생성
        var festivalDiv = document.createElement("div");
        festivalDiv.classList.add("main-festival-main-warp");

        // 축제 상단 요소 생성
        var festivalTopDiv = document.createElement("div");
        festivalTopDiv.classList.add("main-festival-main-warp-top");
        var festivalTopLink = document.createElement("a");
        festivalTopLink.setAttribute("href", "");
        festivalTopLink.classList.add("main-festival-main-warp-top-a");
        var festivalTopImage = document.createElement("img");
        festivalTopImage.setAttribute(
            "src",
            "https://tong.visitkorea.or.kr/cms/resource/49/3057649_image2_1.jpg"
        );
        festivalTopImage.setAttribute("alt", "축제사진");
        festivalTopImage.classList.add("main-festival-main-warp-top-img");
        festivalTopLink.appendChild(festivalTopImage);
        var festivalTopMessage = document.createElement("div");
        festivalTopMessage.textContent = "진행중";
        festivalTopMessage.classList.add("main-festival-main-warp-top-message");
        festivalTopDiv.appendChild(festivalTopLink);
        festivalTopDiv.appendChild(festivalTopMessage);

        // 축제 중단 요소 생성
        var festivalMidDiv = document.createElement("div");
        festivalMidDiv.classList.add("main-festival-main-warp-mid");
        var festivalMidPlace = document.createElement("div");
        festivalMidPlace.textContent = "강원";
        festivalMidPlace.classList.add("main-festival-main-warp-mid-place");
        var festivalMidBet = document.createElement("div");
        festivalMidBet.classList.add("main-festival-main-warp-mid-bet");
        var festivalMidDate = document.createElement("div");
        festivalMidDate.classList.add("main-festival-main-warp-mid-date");
        var date1 = document.createElement("div");
        date1.textContent = "24년01월27일(토)";
        var date2 = document.createElement("div");
        date2.textContent = "~";
        var date3 = document.createElement("div");
        date3.textContent = "24년02월11일(일)";
        festivalMidDate.appendChild(date1);
        festivalMidDate.appendChild(date2);
        festivalMidDate.appendChild(date3);

        // 축제 하단 요소 생성
        var festivalBottomDiv = document.createElement("div");
        festivalBottomDiv.classList.add("main-festival-main-warp-bt");
        var festivalBottomLink = document.createElement("a");
        festivalBottomLink.setAttribute("href", "");
        festivalBottomLink.classList.add("main-festival-main-warp-bt-a");
        festivalBottomLink.textContent = "대관령눈꽃축제";
        festivalBottomDiv.appendChild(festivalBottomLink);

        // 요소들을 조립하여 추가
        festivalMidDiv.appendChild(festivalMidPlace);
        festivalMidDiv.appendChild(festivalMidBet);
        festivalMidDiv.appendChild(festivalMidDate);

        festivalDiv.appendChild(festivalTopDiv);
        festivalDiv.appendChild(festivalMidDiv);
        festivalDiv.appendChild(festivalBottomDiv);

        festivalContainer.appendChild(festivalDiv);
    }
}
