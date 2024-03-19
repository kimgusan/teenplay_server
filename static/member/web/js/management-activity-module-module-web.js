const activityMemberService = (() => {
        const getList = async (member_id, page, activity_id,member_status,search, callback) => {
            const response = await fetch(`/member/activity-member/${member_id}/${page}/${activity_id}/?member_status=${member_status}&search=${search}`);
            const response_data = await response.json();
            const member_list = response_data.member_list;
            const total_pages = response_data.total_pages;
            const clubManager = response_data.club_manager;

            return callback(member_list,clubManager, total_pages);
        }

        const update = async (activity_member_id, activity_id) => {
            await fetch(`/member/activity-member/${activity_member_id}/${activity_id}/`, {
            method: 'PATCH',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }

        const remove = async (activity_member_id, activity_id) => {
            await fetch(`/member/activity-member/${activity_member_id}/${activity_id}/`, {
            method: 'delete',
            headers: {'X-CSRFToken': csrf_Token}
            });
        }

        const write = async (data) => {
            const response = await fetch("/member/activity-member/api/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_Token
            },
            body: JSON.stringify(data)
        });
    }


    return {getList:getList, update:update, remove:remove, write:write}
})();
















