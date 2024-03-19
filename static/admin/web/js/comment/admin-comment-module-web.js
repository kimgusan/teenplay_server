const adminCommentService = (() => {
    // 페이지 가져오기
    const getPagination = async (page, category, type, keyword, callback) => {
        const response = await fetch(`/admin/comments/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'category': category,
                'type': type,
                'keyword': keyword
            })
        })
        const pagination = await response.json()

        if(callback) {
            return  callback(pagination)
        }
        return pagination;
    }

    // 회원 상태 변경
    const update = async (targetId) => {
        const member_id = targetId.targetId

        await fetch(`/admin/user/update/${member_id}/`, {
            method: 'patch',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'member_id': member_id})
        });

    }

    // 댓글 삭제
    const remove = async (replyId, memberId, createdDate) => {
        const reply_id = replyId
        const reply_member_id = memberId
        const created_date = createdDate

        await fetch(`/admin/comments/delete/?reply_id=${reply_id}&reply_member_id=${reply_member_id}&created_date=${created_date}`, {
            method: 'delete',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'reply_id': reply_id, 'reply_member_id': reply_member_id, 'created_date': created_date})
        });
    }


    return {getPagination:getPagination, update:update, remove:remove}
})();

