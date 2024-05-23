const activityReplyService = (() => {
    const write = async (activityReply) => {
        const response = await fetch(`/activity/replies/api/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(activityReply)
        });
        return await response.json()
    };

    const getList = async (isAdd, page, activityId, callback) => {
        const response = await fetch(`/activity/replies/api/?page=${page}&activity-id=${activityId}`);
        const replies = await response.json();
        if (callback) {
            callback(isAdd, replies);
        }
    }

    const update = async (activityReply) => {
        const response = await fetch(`/activity/replies/api/`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(activityReply)
        });
        return await response.json()
    }

    const remove = async (id) => {
        await fetch(`/activity/replies/api/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(id)
        });
    }

    const report = async (replyInfo) => {
        await fetch(`/ai/api/report/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(replyInfo)
        })
    }

    return {write: write, getList: getList, update: update, remove: remove, report: report}
})()