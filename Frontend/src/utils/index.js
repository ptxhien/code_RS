export const splitToSubArr = (arr = [], perPage = 6) => {
  const copyArr = [...arr];
  const res = [];
  while (copyArr.length) {
    res.push(copyArr.splice(0, perPage));
  }
  return res;
};
