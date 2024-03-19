// 유저 정보 가져오기
const adminUserService = (() => {
    // 페이지 가져오기
    const getPagination = async (page, category, keyword, callback) => {
        console.log(category)
        const response = await fetch(`/admin/users/${page}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'category': category,
                'keyword': keyword
            })
        })
        const pagination = await response.json();

        if (callback){
            return callback(pagination);
        }
        return pagination;
    }

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

    return {getPagination:getPagination, update:update}
})();
