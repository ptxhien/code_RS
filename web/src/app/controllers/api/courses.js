const OnlineCourseModel = require("../../models/onlineCourse");
const OfflineCourseModel = require("../../models/offlineCourse");


const APICousesController = function (){

};


APICousesController.prototype.get = async (req, res, next) => {
  const onlineCourses = await OnlineCourseModel.find("*", "1=1");
  const offlineCourses = await OfflineCourseModel.find("*", "1=1");

  const data = [...onlineCourses, ...offlineCourses];

  res.status(200).json({data});
};

APICousesController.prototype.getOne = async (req, res, next) => {
  const { id } = req.params;
  try {
    const offlineCourse = await OfflineCourseModel.findOne("*", `courseID="${id}"`);
    if (offlineCourse){
      res.status(200).json({data: offlineCourse});
      return;
    }
    const onlineCourse = await OnlineCourseModel.findOne("*", `courseID="${id}"`);
    if(onlineCourse) {
      res.status(200).json({data: onlineCourse});
      return;
    }

    res.status(404).json({msg: "Could not found this course"});
  } catch (err) {
    res.status(404).json({ msg: "Could not found this course" });
  }
};

module.exports = new APICousesController();