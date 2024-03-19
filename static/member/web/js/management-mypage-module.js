const mypageActivityListService = (() => {
    const list = async (club_id, page, order, callback) => {
        const response = await fetch(`/member/mypage-activity-list/${club_id}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'order': order
            })
        })
        const list = await response.json()

        if (callback) {
            return callback(list)
        }
        return list
    }

    return {list: list}
})();

const mypageMemberListService = (() => {
    const list = async (club_id, page, order, search, callback) => {
        const response = await fetch(`/member/mypage-member-list/${club_id}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'order': order,
                'search': search,
                'page': page
            })
        });
        const list = await response.json()
        if (callback) {
            return callback(list)
        }
        return list
    }
    return {list: list}
})()


const mypageClubNoticeListService = (() => {
    const list = async (club_id, page, del_item, callback) => {
        console.log(del_item)
        const response = await fetch(`/member/mypage-notice-list/${club_id}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                page: page,
                del_item: del_item
            })
        });
        const list = await response.json()
        if (callback) {
            return callback(list)
        }
        return list
    }
    return {list: list}
})()


const clubNoticeService = (() => {
    const del = async (club_id, delList) => {
        await fetch(`/member/mypage-notice-delete/${club_id}/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'del_list': delList})
        });
    }
    return {del: del}
})()


const mypageMemberStatusService = (() => {
    const del = async (club_id, memberId) => {
        const response = await fetch(`/member/mypage-member-status/${club_id}/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'member_id': memberId})
        });
        return await response.json();
    }
    const patch = async (club_id, memberId) => {
        console.log(memberId)
        const response = await fetch(`/member/mypage-member-status/${club_id}/`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'member_id': memberId})
        });
        return await response.json();
    }
    return {del: del, patch: patch}
})()

const mypageSendLetterService = (() => {
    const post = async (letter) => {
        console.log(letter)
        const response = await fetch(`/member/mypage-send-letter/api/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({'letter': letter})
        });
        return await response.json();
    }
    return {post: post}
})()

const mypageClubListService = (() => {
    const list = async (page, order, callback) => {
        const response = await fetch(`/member/mypage-my-club/api/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify({
                'page': page,
                'order': order
            })
        })
        const list = await response.json()
        if (callback) {

            return callback(list)
        }
        return list
    }

    return {list: list}
})()

const mypageClubAlarmStatusService = (() => {
    const alarm = async (clubId) => {
        const response = await fetch(`/member/mypage-my-club-alarm/${clubId}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
        });
        return await response.json();
    }

    return {alarm: alarm}
})()