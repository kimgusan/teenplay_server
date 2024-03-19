const clubPostLetterService = (() => {
    const write = async (letter) => {
        await fetch(`/member/mypage-apiletter/`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8',
                'X-CSRFToken': csrf_token
            },
            body: JSON.stringify(letter)
        });
    }

    return {write: write}
})();