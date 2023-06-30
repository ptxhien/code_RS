const OfflineCourseModel = require("../../models/offlineCourse");
const FreetimeController = function () {};

function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

FreetimeController.prototype.get = async (req, res, next) => {
  const studyTimes = await OfflineCourseModel.studyTimes('studyTime');
  let result = [];
  for (let i = 0; i < studyTimes.length; i++) {
    if (studyTimes[i].studyTime) {
      var times = studyTimes[i].studyTime.split('|');
      for(var j = 0; j < times.length; j++) {
        result.push(times[j]);
      }
    }
  }
  res.status(200).json({
    data: result.filter(onlyUnique),
  });
};

module.exports = new FreetimeController();
