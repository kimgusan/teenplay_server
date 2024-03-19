const payService = (() => {
        const getList = async (member_id, page, callback) => {
        const response = await fetch(`/member/mypage-payment/${member_id}/${page}/`);
        const pay = await response.json();
        if(callback){
            return callback(pay)
        }
        return pay
            ;}
        const check = async (pay_reason) =>{
        const response = await fetch("/member/mypage-payment/api/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_Token
            },
            body: JSON.stringify(pay_reason)
        });
        return await response.json();
    }

    return {getList:getList, check:check}
})();
















