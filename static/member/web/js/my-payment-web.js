// 결재 취소 클릭 시 작성 모달 여는 이벤트
const cancleModalWrap = document.querySelector(".cancle-modal-wrap");
const paymentCancle = document.querySelector(".payment-cancle");
page = 1
const payshow = document.getElementById("payment-activity-list")
const cansleButton = document.querySelector(".teenchin-more-btn")


// 결제 취소 모달 내 버튼 클릭 시 발생하는 이벤트
const cancleModalBtns = document.querySelectorAll(".cancle-modal-container button");
const textarea = cancleModalWrap.querySelector("textarea[name=cancle-content]");

cancleModalBtns.forEach((cancleModalBtn) => {
    cancleModalBtn.addEventListener("click", (e) => {
        if (e.target.className == "cancle-check-btn") {
            cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                cancleModalWrap.querySelector(".cancle-modal-container").style.display = "none";
                cancleModalWrap.querySelector(".check-modal-container").style.animation = "popUp 0.5s";
                cancleModalWrap.querySelector(".check-modal-container").style.display = "flex";
                textarea.value = ``;
                checkValue();
            }, 450);
        } else {
            cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popDown 0.5s";
            setTimeout(() => {
                cancleModalWrap.style.display = "none";
                textarea.value = ``;
                checkValue();
            }, 450);
        }
    });
});

// 보내기 확인 모달 내 확인 클릭 시 모달 종료하는 이벤트
const checkModalCheckBtn = document.querySelector(".check-modal-container .check-btn");

checkModalCheckBtn.addEventListener("click", () => {
    cancleModalWrap.querySelector(".check-modal-container").style.animation = "popDown 0.5s";
    setTimeout(() => {
        cancleModalWrap.style.display = "none";
        cancleModalWrap.querySelector(".check-modal-container").style.display = "none";
        cancleModalWrap.querySelector(".cancle-modal-container").style.display = "flex";
    }, 450);
});

// textarea에 입력값이 있을 때 발생하는 이벤트
const cancleCheckBtn = document.querySelector(".cancle-check-btn");

textarea.addEventListener("input", () => {
    checkValue();
});

const checkValue = () => {
    if (textarea.value) {
        cancleCheckBtn.disabled = false;
        return;
    }
    cancleCheckBtn.disabled = true;
};


payService.getList(member_id, page + 1).then((pay) => {
    if (pay.length !== 0){
        cansleButton.style.display = "flex";
    }
});

payService.getList(member_id, page + 1).then((pay) => {
    if (pay.length === 0){
        cansleButton.style.display = "none";
    }
});



const showList =(pay) =>{
    console.log(pay)
    let text = ''
    if (pay.length ===0){
        text +=
            `<div class="nontext">아직 새로운 결제내역이 없습니다.</div>`
    }
    else {
    pay.forEach((pay)=>{
        const activityEndDate = new Date(pay.activity__recruit_end);
        const currentDate = new Date();
        if (activityEndDate >= currentDate && pay.activity__thumbnail_path === ""){
        text += `
                <div class="payment-activity-box">
                    <div class="payment-activity-items">
                        <div class="payment-activity-details">
                            <div class="payment-activity-inside">
                                <div class="payment-activity-thumbnail-wrap">
                                    <div class="payment-activity-thumbnail-container">
                                        <!-- 해당 활동 상세보기 주소 필요 -->
                                        <a href="http://127.0.0.1:10000/member/activity/?activity_id=${pay.activity__id}" target="_blank">
                                            <!-- 해당 활동의 썸네일 경로 필요 -->
                                            <img class="payment-activity-thumbnail" src="/static/public/web/images/logo/logo8.png" />
                                        </a>
                                    </div>
                                </div>
                                <div class="club-activity-content">
                                    <div class="club-name-time">
                                        <div class="club-name-box">
                                            <span>${pay.activity__club__club_name}</span>
                                        </div>
                                        <div class="open-time-box">
                                            <div class="open-time">결제일${changeDate(pay.created_date)}</div>
                                        </div>
                                    </div>
                                    <div class="payment-activity-info">
                                        <div class="payment-activity-title">${pay.activity__activity_title}.</div>
                                        <div class="payment-activity-oneline-info">
                                            <span>${pay.activity__activity_intro}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="payment-info-box">
                            <div class="payment-info-inside">
                                <div class="payment-info">
                                    <div class="payment-amount-title">결제 금액</div>
                                    <div class="payment-amount-box">
                                        <div>20,000원</div>
                                    </div>
                                    <div class="payment-cancle-box">
                                        <!-- 활동 결제 취소하면 어케하지?  -->
                                        <button class="payment-cancle ${pay.id}" type="button">결제 취소</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>`
    }else if (activityEndDate >= currentDate && pay.activity__thumbnail_path !== ""){
        text += `
                <div class="payment-activity-box">
                    <div class="payment-activity-items">
                        <div class="payment-activity-details">
                            <div class="payment-activity-inside">
                                <div class="payment-activity-thumbnail-wrap">
                                    <div class="payment-activity-thumbnail-container">
                                        <!-- 해당 활동 상세보기 주소 필요 -->
                                        <a href="http://127.0.0.1:10000/member/activity/?activity_id=${pay.activity__id}" target="_blank">
                                            <!-- 해당 활동의 썸네일 경로 필요 -->
                                            <img class="payment-activity-thumbnail" src="/upload/${pay.activity__thumbnail_path}" />
                                        </a>
                                    </div>
                                </div>
                                <div class="club-activity-content">
                                    <div class="club-name-time">
                                        <div class="club-name-box">
                                            <span>${pay.activity__club__club_name}</span>
                                        </div>
                                        <div class="open-time-box">
                                            <div class="open-time">결제일${changeDate(pay.created_date)}</div>
                                        </div>
                                    </div>
                                    <div class="payment-activity-info">
                                        <div class="payment-activity-title">${pay.activity__activity_title}.</div>
                                        <div class="payment-activity-oneline-info">
                                            <span>${pay.activity__activity_intro}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="payment-info-box">
                            <div class="payment-info-inside">
                                <div class="payment-info">
                                    <div class="payment-amount-title">결제 금액</div>
                                    <div class="payment-amount-box">
                                        <div>20,000원</div>
                                    </div>
                                    <div class="payment-cancle-box">
                                        <!-- 활동 결제 취소하면 어케하지?  -->
                                        <button class="payment-cancle ${pay.id}" type="button">결제 취소</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>`
    }

    })
}return text;
}

payshow.addEventListener("click", async(e)=>{
    if (e.target.classList[0] === 'payment-cancle'){
        const del = e.target.classList[1]
        cancleModalWrap.querySelector(".cancle-modal-container").style.animation = "popUp 0.5s";
        cancleModalWrap.style.display = "block";
        cancleCheckBtn.addEventListener("click" , async(e)=>{
            let reason_text = document.getElementById("pay-reason")
             await payService.check({
                pay: del ,
                reason : reason_text.value
            })
            const text = await payService.getList(member_id, page, showList);
            payshow.innerHTML = text;
            page = 1
        })
    }
})

cansleButton.addEventListener("click", (e) => {
    payService.getList(member_id, ++page, showList).then((text) => {
        payshow.innerHTML += text;
    });

    payService.getList(member_id, page + 1,).then((pay) => {
    if (pay.length === 0){
        cansleButton.style.display = "none";
    }
});

});


payService.getList(member_id, page, showList).then((text) => {
    payshow.innerHTML += text;
});


payService.getList(member_id, page,showList);


const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}
