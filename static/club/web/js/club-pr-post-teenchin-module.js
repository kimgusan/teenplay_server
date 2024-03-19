const clubPostTeenchinService = (() => {
    // 틴친 신청(POST)
    const apply = async (teenchinId) => {
        await fetch(`/member/teenchin/api/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({teenchinId: teenchinId})
        });
    }

    // 틴친 상태 조회 (GET)
    const getTeenchinStatus = async (teenchinId, callback) => {
        console.log(teenchinId)
        const response = await fetch(`/member/teenchin/api?teenchin-id=${teenchinId}`);
        const teenchinStatus = await response.json();
        if (callback) {
            callback(teenchinStatus);
        }
    }

    // 틴친 취소 or 수락/거절 (PATCH)
    const cancelOrAcceptDenyTeenchin = async (teenchinId, isSender, isAccept) => {
        await fetch(`/member/teenchin/api/`, {
            method: "PATCH",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                teenchinId: teenchinId,
                isSender: isSender,
                isAccept: isAccept,
            })
        })
    }

    // 틴친 끊기 (DELETE)
    const removeTeenchin = async (teenchinId) => {
        await fetch(`/member/teenchin/api/`, {
            method: "DELETE",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({teenchinId: teenchinId})
        })
    }

    return {
        apply: apply,
        getTeenchinStatus: getTeenchinStatus,
        cancelOrAcceptDenyTeenchin: cancelOrAcceptDenyTeenchin,
        removeTeenchin: removeTeenchin
    }
})();