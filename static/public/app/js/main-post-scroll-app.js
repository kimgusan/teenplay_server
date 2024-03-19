//
const postLists = document.querySelectorAll(".post-list");

postLists.forEach((postList) => {
    // 가려진 영역(스크롤 영역)을 포함한 요소의 가로 사이즈
    const postListScrollWidth = postList.scrollWidth;
    // 가려진 영역은 제외한 현재 화면에 보이는 요소에 대한 가로 사이즈
    const postListClientWidth = postList.clientWidth;

    let startX = 0;
    let nowX = 0;
    let endX = 0;
    let listX = 0;

    // 중복 사용이 많아 함수 선언
    const onScrollStart = (e) => {
        startX = getClientX(e);
        window.addEventListener("mousemove", onScrollMove);
        window.addEventListener("touchmove", onScrollMove);
        window.addEventListener("mouseup", onScrollEnd);
        window.addEventListener("touchend", onScrollEnd);
    };

    const onScrollMove = (e) => {
        nowX = getClientX(e);
        setTranslateX(listX + nowX - startX);
    };

    const onScrollEnd = (e) => {
        endX = getClientXWhenTouchend(e);
        listX = getTranslateX();
        if (listX > 0) {
            setTranslateX(0);
            postList.style.transition = `all 0.3s ease`;
            listX = 0;
        } else if (listX < postListClientWidth - postListScrollWidth) {
            setTranslateX(postListClientWidth - postListScrollWidth);
            postList.style.transition = `all 0.3s ease`;
            listX = postListClientWidth - postListScrollWidth;
        }

        window.removeEventListener("mousedown", onScrollStart);
        window.removeEventListener("touchstart", onScrollStart);
        window.removeEventListener("mousemove", onScrollMove);
        window.removeEventListener("touchmove", onScrollMove);
        window.removeEventListener("mouseup", onScrollEnd);
        window.removeEventListener("touchend", onScrollEnd);
        window.removeEventListener("click", onClick);

        setTimeout(() => {
            bindEvents();
            postList.style.transition = "";
        }, 300);
    };
    const onClick = (e) => {
        if (startX - endX !== 0) {
            e.preventDefault();
        }
    };

    // 터치 이벤트가 발생한 경우에는 e.touches[0].clientX를 참조
    const getClientX = (e) => {
        // 터치 이벤트가 발생했는지 확인
        const isTouches = e.touches ? true : false;
        // 터치 이벤트가 발생한 경우에는 e.touches[0].clientX를 참조
        // 마우스로 클릭한 지점의 X좌표는 e.clientX
        return isTouches ? e.touches[0].clientX : e.clientX;
    };

    // touchend 이벤트에서는 e.touches[0].clientX로 가져올수 없어 추가
    const getClientXWhenTouchend = (e) => {
        const isTouches = e.touches ? true : false;
        return isTouches ? e.changedTouches[0].clientX : e.clientX;
    };

    // 두 번째 스크롤부터는 스크롤이 종료된 위치도 고려하여 계산해야 하기 때문에 요소의 translateX 위치를 가져와야 합니다.
    // window 객체에 내장된 API인 getComputedStyle 메서드를 사용하면 요소가 가진 CSS의 속성 값을 얻을 수 있는데,
    // transform의 경우 x, y, z의 값을 모두 반환하므로 정규표현식을 통해 필요한 x의 값만 얻도록 했습니다.
    const getTranslateX = () => {
        return parseInt(getComputedStyle(postList).transform.split(/[^\-0-9]+/g)[5]);
    };

    // 스크롤 됨에 따라 요소의 위치를 조정해야 하기 때문에, 간편하게 함수로 만들어 재사용했습니다.
    const setTranslateX = (x) => {
        postList.style.transform = `translateX(${x}px)`;
    };
    const bindEvents = () => {
        postList.addEventListener("mousedown", onScrollStart);
        postList.addEventListener("touchstart", onScrollStart);
        postList.addEventListener("click", onClick);
    };
    bindEvents();
});
