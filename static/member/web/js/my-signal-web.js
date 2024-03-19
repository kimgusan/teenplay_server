

const signalDeletes = document.querySelectorAll(".signal-delete");
const signalDeleteModal = document.querySelector(".signal-delete-modal");
const signalDeleteBox = document.querySelector(".signal-delete-box");
let signalTarget;
const tbody = document.querySelector("#inner")
const moreButton = document.getElementById("more-button")
page = 1
const canButton = document.getElementById("delete-cancle-btn")

canButton.addEventListener("click", (e)=>{
    signalDeleteModal.style.display = "none"
})


alramService.getList(member_id, page + 1).then((alram) => {
    if (alram.length !== 0){
        moreButton.style.display = "flex";
    }
});

alramService.getList(member_id, page + 1).then((alram) => {
    if (alram.length === 0){
        moreButton.style.display = "none";
    }
});


moreButton.addEventListener("click", (e) => {
    alramService.getList(member_id, ++page, showList).then((text) => {
        tbody.innerHTML += text;
    });

    alramService.getList(member_id, page + 1).then((alram) => {
    if (alram.length === 0){
        moreButton.style.display = "none";
    }
});

});





const showList = async (alram) => {
    let text = '';
    if (alram.length ===0){
            text +=
            `<div class="signal-none">아직 새로운 알림이 없습니다.</div>`
        }
    else{
        alram.forEach((alram) =>  {
            if(alram.alarm_type ===1){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">모임</div>
                            <a href="http://127.0.0.1:10000/club/pr-post-detail/?id=${alram.target_id}" class="target-title">당신이 작성한 모임홍보글에 댓글이 작성되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/club/pr-post-detail/?id=${alram.target_id}" class="signal-bottom-text">댓글을 확인하주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 2){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">활동</div>
                            <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="target-title">당신이 작성한 활동글에 댓글이 작성되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="signal-bottom-text">댓글을 확인해주세요!</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 3){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">위시리스트</div>
                            <a href="http://127.0.0.1:10000/member/mypage-wishlist" class="target-title">당신이 작성한 위시리스트에 댓글이 작성되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/member/mypage-wishlist" class="signal-bottom-text">댓글을 확인해주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 4){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">쪽지</div>
                            <a href="http://127.0.0.1:10000/member/mypage-letter/" class="target-title">당신에게 쪽지가 도착하였어요!!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/member/mypage-letter/" class="signal-bottom-text">쪽지를 확인해주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 5){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">틴친</div>
                            <a href="http://127.0.0.1:10000/member/mypage-teenchin/" class="target-title">당신에게 틴친을 신청한 사람이 있어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/member/mypage-teenchin/" class="signal-bottom-text">틴친 신청내역을 확인해주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 6){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">활동</div>
                            <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="target-title">당신이 가입한 모임에 활동이 개설되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="signal-bottom-text">활동을 확인해주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 7){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">모임</div>
                            <a href="http://127.0.0.1:10000/club/detail/?id=${alram.target_id}&view=notice" class="target-title">당신이 가입된 모임에 공지사항이 작성되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/club/detail/?id=${alram.target_id}&view=notice" class="signal-bottom-text">공지사항을 확인해주세요.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 8){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">모임</div>
                            <a href="http://127.0.0.1:10000/teenplay/club/select/${alram.target_id}" class="target-title">당신이 가입된 모임에 틴플레이가 작성되었어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/teenplay/club/select/${alram.target_id}" class="signal-bottom-text">틴플레이를 확인해주세요!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 9){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">모임</div>
                            <a href="http://127.0.0.1:10000/club/detail/?id=${alram.target_id}"  class="target-title">당신이 신청한 모임에서 소식이 왔어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/club/detail/?id=${alram.target_id}" class="signal-bottom-text">모임에 가입되었습니다!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 10){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">모임</div>
                            <div class="target-title">당신이 신청한 모임에서 소식이 왔어요!</div>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <div class="signal-bottom-text">모임에서 탈퇴되였습니다!.</div>
                </div>
            </div>
`}
            if(alram.alarm_type === 11){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">활동</div>
                            <a href="http://127.0.0.1:10000/member/activity-member/?activity_id=${alram.target_id}"  class="target-title">누군가 당신의 활동을 신쳥했어요!!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <div href="http://127.0.0.1:10000/member/activity-member/?activity_id=${alram.target_id}" class="signal-bottom-text">참가신청을 확인해주세요.</div>
                </div>
            </div>
`}
            if(alram.alarm_type === 12){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">활동</div>
                            <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="target-title">당신이 신청한 활동에서 소식이왔어요!</a>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="signal-bottom-text">신청하신 활동의 참가가 승인되었습니다!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 13){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">활동</div>
                            <div class="target-title">당신이 신청한 활동에서 소식이왔어요!</div>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <a href="http://127.0.0.1:10000/activity/detail/?id=${alram.target_id}" class="signal-bottom-text">신청하신 활동의 참가가 거절되었습니다!.</a>
                </div>
            </div>
`}
            if(alram.alarm_type === 14){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">틴친</div>
                            <div class="target-title">당신이 신청한 틴친에게서 소식이왔어요!</div>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <div class="signal-bottom-text">요청하신 틴친요청이 승인되었습니다.</div>
                </div>
            </div>
`}
            if(alram.alarm_type === 15){
            text += `
             <div class="signal-container">
                <div class="signal-box">
                    <div class="signal-top-box">
                        <div class="type-title">
                            <div class="signal-type">틴친</div>
                            <div class="target-title">당신이 신청한 틴친에게서 소식이왔어요!</div>
                        </div>
                        <div class="check-date">
                            <div class="signal-delete ${alram.id}">읽음</div>
                            <div class="signal-date">${changeDate(alram.created_date)}</div>
                        </div>
                    </div>
                    <div class="signal-bottom-text">요청하신 틴친요청이 거절되었습니다.</div>
                </div>
            </div>
`}

    });}

    return text;

};

const deleteButton = document.getElementById('delete-check-btn')
tbody.addEventListener("click", async (e) =>{
    if (e.target.classList[0] === 'signal-delete'){
        const data = e.target.classList[1];
        signalDeleteModal.style.display = "block"
        console.log(data)
        deleteButton.addEventListener("click", async(e)=>{
            await alramService.remove(data)
            page = 1
            const text = await alramService.getList(member_id, page, showList);
            tbody.innerHTML = text;
            signalDeleteModal.style.display = "none"
            await alramService.getList(member_id, page + 1).then((alram) => {
            if (alram.length !== 0){
            moreButton.style.display = "flex";
    }
});
        } )


    }
})


alramService.getList(member_id, page, showList).then((text) => {
    tbody.innerHTML += text;
});


alramService.getList(member_id, page, showList);




const changeDate = (dateStr) => {
    const days = ['일', '월', '화', '수', '목', '금', '토'];
    const date = new Date(dateStr); // string타입을 date타입으로 바꿈.
    const month = date.getMonth() + 1; // getMonth()는 0부터 시작하므로 1을 더함.
    const day = date.getDate();
    const dayOfWeek = days[date.getDay()]; // 요일을 숫자로 가져와 해당 요일 문자열로 변환

    // padStart는 ,앞의 숫자의 자릿수 만큼 0을 채워줌
    return `${month.toString().padStart(2, '0')}월 ${day.toString().padStart(2, '0')}일(${dayOfWeek})`;
}
