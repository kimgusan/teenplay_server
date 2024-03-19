const replyService = (() => {
    const getList = async (member_id, page,status_reply, callback) => {
        const response = await fetch(`/member/mypage-reply/${member_id}/${page}/?status_reply=${status_reply}`);
        const response_data = await response.json();
        // 서버 응답에서 전체 페이지 수 추출

        // 각각의 데이터에 접근
        const reply = response_data.reply;
        const total_pages = response_data.total_pages;
        // callback 함수에 필요한 데이터 전달
        return callback(reply,total_pages);
}

    const remove = async (reply_id) => {
        await fetch(`/member/mypage-reply/${reply_id}/`, {
            method: 'delete',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }



    return {getList:getList, remove:remove}
})();
















