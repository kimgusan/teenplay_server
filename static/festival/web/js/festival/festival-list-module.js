const festivalService = (() => {
    const getList = async (page, keyword, category, date, ongoing, includingEnd, callback) => {
        const response = await fetch(`/festival/list/api/?page=${page}&keyword=${keyword}&category=${category}&date=${date}&ongoing=${ongoing}&including_end=${includingEnd}`)
        const festivals = await response.json()

        if (callback) {
            return callback(festivals)
        }

        return festivals
    }

    return {getList: getList}
})()