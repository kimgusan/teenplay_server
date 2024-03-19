const alramService = (() => {
        const getList = async (member_id, page, callback) => {
        const response = await fetch(`/member/mypage-alram/${member_id}/${page}/`);
        const alram = await response.json();
        if(callback){
            return callback(alram)
        }
        return alram
            ;}

        const remove = async (alram_id) => {
            await fetch(`/member/mypage-alram/${alram_id}/`, {
                method: 'delete',
                headers: {'X-CSRFToken': csrf_Token}
            });
        }

    return {getList:getList, remove:remove}
})();
















