const teenchinService = (() => {
        const getList = async (member_id, page,status_teenchin,search, callback) => {
        const response = await fetch(`/member/mypage-teenchin/${member_id}/${page}/?status_teenchin=${status_teenchin}&search=${search}/`);
        const teenchin = await response.json();
        if(callback){
            return callback(teenchin)
        }
        return teenchin
            ;}

        const remove = async (friend_id) => {
            await fetch(`/member/mypage-teenchin/${friend_id}/`, {
                method: 'delete',
                headers: {'X-CSRFToken': csrf_Token}
            });
        }

        const update = async (friend_id) => {
            await fetch(`/member/mypage-teenchin/${friend_id}/`, {
            method: 'PATCH',
            headers: {'X-CSRFToken': csrf_Token}
        });
    }
        const write = async (letter) => {
            const response = await fetch("/member/mypage-teenchinapi/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_Token
            },
            body: JSON.stringify(letter)
        });
    }








    return {getList:getList,remove:remove, update:update, write:write}
})();
















