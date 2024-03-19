const activityService = (() => {
        const getList = async (member_id, page,status_list, callback) => {
        const response = await fetch(`/member/activity/${member_id}/${page}/?status_list=${status_list}/`);
        const response_data = await response.json();
        const activity_data = response_data.activity_data
        const total_pages = response_data.total_pages

        return callback(activity_data,total_pages)
            ;}


    return {getList:getList}
})();
















