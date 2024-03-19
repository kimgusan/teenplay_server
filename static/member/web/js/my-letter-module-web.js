const replyService = (() => {
    const write = async (letter) => {
        const response = await fetch("/member/mypage-apiletter/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_Token
            },
            body: JSON.stringify(letter)
        });
    }

    const getList = async (member_id, page, status_letter, callback) => {
        const response = await fetch(`/member/mypage-letter/${member_id}/${page}/?status_letter=${status_letter}`);
        const response_data = await response.json();
        // 서버 응답에서 전체 페이지 수 추출

        // 각각의 데이터에 접근
        const replies = response_data.replies;
        const total_pages = response_data.total_pages;
        // callback 함수에 필요한 데이터 전달
        return callback(replies,total_pages);
}


    const remove = async (letter_id) => {
        await fetch(`/member/mypage-letter/${letter_id}/`, {
            method: 'delete',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }

    const update = async (letter_id) => {
        await fetch(`/member/mypage-letter/${letter_id}/`, {
            method: 'PATCH',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }

    const check = async (email) =>{
        const response = await fetch("/member/mypage-letterapi/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_Token
            },
            body: JSON.stringify({ member_email: email })
        });
        return await response.json();
    }


    return {write: write,getList:getList,remove:remove, update:update, check:check}
})();
















