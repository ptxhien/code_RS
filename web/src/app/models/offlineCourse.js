const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const OfflineCourseModel = function (Obj) {
  this.courseID = Obj ? Obj.courseID : undefined;
  this.courseTitle = Obj ? Obj.courseTitle : undefined;
  this.URL = Obj ? Obj.URL : undefined;
  this.outcomeLearning = Obj ? Obj.outcomeLearning : undefined;
  this.majobSubject = Obj ? Obj.majobSubject : undefined;
  this.location = Obj ? Obj.location : undefined;
  this.provider = Obj ? Obj.provider : undefined;
  this.studyForm = Obj ? Obj.studyForm : undefined;
  this.duration = Obj ? Obj.duration : undefined;
  this.feeVND = Obj ? Obj.feeVND : undefined;
  this.studyTime = Obj ? Obj.studyTime : undefined;
  this.technologySkill = Obj ? Obj.technologySkill : undefined;
  this.level = Obj ? Obj.level : undefined;
  this.language = Obj ? Obj.language : undefined;
  this.durationSecond = Obj ? Obj.durationSecond : undefined;
  this.Tech_Skill = Obj ? Obj.Tech_Skill : undefined;
};



OfflineCourseModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'offlineCourse'");
    const newID = utilities.createID(nextIndex, 3, "COURSE_");
    this.offlineCourseID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='offlineCourse'");
    const [res1] = await db.insert("offlineCourse", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


OfflineCourseModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'offlineCourse'");
    const newID = utilities.createID(nextIndex, 3, "COURSE_")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


OfflineCourseModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"], isPaging = false, page = 1, perPage = 8) {
  const [result] = await db.get("offlineCourse", fileds, conditions, isPaging, page, perPage);
  return result;
};


OfflineCourseModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"], isPaging = false, page = 1, perPage = 8) {
  const [result] = await db.get("offlineCourse", fileds, conditions, isPaging, page, perPage);
  return result;
};


OfflineCourseModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("offlineCourse", fileds, conditions);
  if (result) {
    return new OfflineCourseModel(result);
  } else {
    return null;
  }
};


OfflineCourseModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("offlineCourse", fileds, conditions);
  if (result) {
    return new OfflineCourseModel(result);
  } else {
    return null;
  }
};

OfflineCourseModel.prototype.update = async function () {};


OfflineCourseModel.update = async function (obj) {};


OfflineCourseModel.count = async function (conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.count("offlineCourse", conditions);
  return result;
};


OfflineCourseModel.prototype.count = async function (conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.count("offlineCourse", conditions);
  return result;
};


OfflineCourseModel.studyTimes = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("offlineCourse", fileds, conditions);
  if (result) {
    return result;
  } else {
    return null;
  }
};


module.exports = OfflineCourseModel;