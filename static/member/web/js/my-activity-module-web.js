const activityService = (() => {
        const getList = async (member_id, page,status_like, callback) => {
        const response = await fetch(`/member/mypage-activity/${member_id}/${page}/?status_like=${status_like}`);
        const activity_data = await response.json();
        if(callback){
            return callback(activity_data)
        }
        return activity_data
            ;}

    const update = async (activity_id) => {
        await fetch(`/member/mypage-activity/${activity_id}/`, {
            method: 'PATCH',
            headers: {'X-CSRFToken': csrf_Token}
        })
     }



    return {getList:getList, update:update}
})();
















