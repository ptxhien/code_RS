module.exports = {
  createID: (no = 1, numOfDigits = 3, prefix = "ABC_") => {
    let suffix = "0".repeat(numOfDigits);
    suffix = (suffix + no).slice(-numOfDigits);
    return prefix + suffix;
  },
  clamp: (val = 1, min = 0, max = 10) => val < min ? min : val > max ? max : val,
  sortObjectToArray: (object = {}, sortType="DESC") => {
    const sortable = [];
    for (key in object) {
      sortable.push([key, object[key]]);
    }
    sortable.sort(function (a, b) {
      return sortType === "ASC" ? (a[1] - b[1]) : (b[1] - a[1]);
    });
    return sortable.reduce((prev, curr) => {
      console.log(curr);
      return [...prev, curr[1]];
    }, []);
  }
};
