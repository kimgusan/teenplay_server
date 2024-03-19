// 처음 불러올 page 번호
let page = 1;

// user_info 데이터 가져오기
const getUserList =  (callback) => {
    fetch(`/admin/users/${page}`)
        .then((response) => response.json())
        .then((user_info) => {
            if (callback) {
                callback(user_info);
            }
        });
};

// 유저 목록

// 데이터 화면에 뿌리기
const showUserList = (user_info) => {
    if (!user_info) {
        const userList = document.querySelector(".user-data")
        userList.innerHTML = `
            <div class="main-user-status-list">
                <svg width="53" height="56" fill="none" xmlns="http://www.w3.org/2000/svg" class="main-user-status-list-svg">
                    <path
                        d="M27.516 1.62c-6.654-4.436-21.823.626-21.821 10.475 0 5.567 1.956 10.31 3.889 14.205-1.403-.463-2.853-1.072-3.8-2.095-.466-.502-1.151-1.115-1.802-1.35-.993-.358-2.035.135-2.572.923-1.212 1.776-1.071 2.717-.403 4.643 4.364 5.647 8.167 5.53 11.275 5.435.286-.009.566-.017.84-.021.897 3.13 2.706 6.042 4.099 8.965 1.617 5.694 9.313 4.544 12.913 1.697 2.3-1.819 5.187-2.36 8.117-1.475 2.303.419 3.88-.68 5.236-1.624 1.031-.72 1.934-1.348 2.93-1.149 1.19.238 2.511.191 2.718-1.307.521-3.781-8.391-5.535-11.284-7.062 4.98-2.883 7.416-7.67 6.968-10.861-.31-2.2-2.168-2.967-4.076-1.773-2.206 1.38-2.758 3.32-5.42 4.348 1.822-6.821 3.386-14.515-7.807-21.975Z"
                        fill="#9D8DFF"
                    ></path>
                    <path d="M15.33 17.017c.03.784-.382 1.385-.92 1.344-.54-.042-1.001-.71-1.032-1.494-.03-.784.382-1.385.921-1.343.54.041 1 .71 1.031 1.493ZM19.599 16.868c.03.784-.382 1.385-.921 1.344-.54-.042-1-.71-1.031-1.494-.03-.784.382-1.385.92-1.343.54.041 1.001.71 1.032 1.493Z" fill="#060606"></path>
                    <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M11.457 29.625a.7.7 0 0 1 .755.64c.036.44.242.87.57 1.304.322.428.721.81 1.107 1.178l.027.026a.7.7 0 1 1-.967 1.012l-.05-.048c-.37-.353-.842-.804-1.235-1.325-.414-.55-.782-1.23-.847-2.033a.7.7 0 0 1 .64-.754ZM35.844 23.41a.7.7 0 0 0-.968.205 25.97 25.97 0 0 1-.389.571c-.235.34-.487.704-.68 1.032-.33.559-.626 1.212-.626 1.941a.7.7 0 1 0 1.4 0c0-.354.145-.745.43-1.228.18-.303.368-.573.573-.868.146-.21.3-.431.465-.684a.7.7 0 0 0-.205-.969Z"
                        fill="#000"
                    ></path>
                    <path d="M17.917 19.838c-.158-.342-2.435-.62-2.138.915 1.156 3.742 2.834.598 2.138-.915Z" fill="#030303"></path>
                    <path d="M51.901 52.983c0 1.223-7.433 2.214-16.602 2.214-9.17 0-16.603-.991-16.603-2.214 0-1.222 7.433-2.213 16.603-2.213 9.17 0 16.602.99 16.602 2.213Z" fill="#E3E3E3"></path>
                </svg>
                <p class="main-user-status-list-text">유저가 없습니다.</p>
            </div>
        `
    }

    const userMoreBtn = document.querySelector(".main-user-bottom-box")

    if (!user_info.hasNext){
        userMoreBtn.style.display = "none";
    }

    let users = user_info.users;
    const userList = document.querySelector(".user-data")
    users.forEach((user) => {
        let statusData = '';
        if (user.status === 1) {
            statusData = '활동중'
        }

        else if (user.status === -1) {
            statusData = '정지'
        }

        else {
            statusData = '탈퇴'
        }

        userList.innerHTML += `
            <li class="main-user-list" data-number="3">
                <div class="main-user-list-check">
                    <input type="checkbox" class="main-comment-list-checkbox" />
                </div>
                <div class="main-user-list-name">${user.member_nickname}</div>
                <div class="main-user-list-status">${user.created_date.slice(0,10)}</div>
                <div class="main-user-list-date">${user.club_count}</div>
                <div class="main-user-list-pay">${user.club_action_count}</div>
                <div class="main-user-list-paycount">${user.activity_count}</div>
                <div data-id="3" class="main-user-list-message">${statusData}</div>
            </li>
        `
    });
};


getUserList(showUserList);