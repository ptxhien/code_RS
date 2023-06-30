const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const OnlineCourseModel = function (Obj) {
  this.courseID = Obj ? Obj.courseID : undefined;
  this.courseTitle = Obj ? Obj.courseTitle : undefined;
  this.URL = Obj ? Obj.URL : undefined;
  this.outcomeLearning = Obj ? Obj.outcomeLearning : undefined;
  this.provider = Obj ? Obj.provider : undefined;
  this.technologySkill = Obj ? Obj.technologySkill : undefined;
  this.rating = Obj ? Obj.rating : undefined;
  this.peopleRating = Obj ? Obj.peopleRating : undefined;
  this.numStudent = Obj ? Obj.numStudent : undefined;
  this.duration = Obj ? Obj.duration : undefined;
  this.durationSecond = Obj ? Obj.durationSecond : undefined;
  this.fee = Obj ? Obj.fee : undefined;
  this.feeVND = Obj ? Obj.feeVND : undefined;
  this.level = Obj ? Obj.level : undefined;
  this.majobSubject = Obj ? Obj.majobSubject : undefined;
  this.language = Obj ? Obj.language : undefined;
  this.Tech_Skill = Obj ? Obj.Tech_Skill : undefined;
};



OnlineCourseModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'onlineCourse'");
    const newID = utilities.createID(nextIndex, 3, "COURSE_");
    this.onlineCourseID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='onlineCourse'");
    const [res1] = await db.insert("onlineCourse", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


OnlineCourseModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'onlineCourse'");
    const newID = utilities.createID(nextIndex, 3, "COURSE_")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


OnlineCourseModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"], isPaging = false, page = 1, perPage = 8) {
  const [result] = await db.get("onlineCourse", fileds, conditions, isPaging, page, perPage);
  return result;
};


OnlineCourseModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"], isPaging = false, page = 1, perPage = 8) {
  const [result] = await db.get("onlineCourse", fileds, conditions, isPaging, page, perPage);
  return result;
};


OnlineCourseModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("onlineCourse", fileds, conditions);
  if (result) {
    return new OnlineCourseModel(result);
  } else {
    return null;
  }
};


OnlineCourseModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("onlineCourse", fileds, conditions);
  if (result) {
    return new OnlineCourseModel(result);
  } else {
    return null;
  }
};

OnlineCourseModel.prototype.update = async function () {};


OnlineCourseModel.update = async function (obj) {};


OnlineCourseModel.count = async function (conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.count("onlineCourse", conditions);
  return result;
};


OnlineCourseModel.prototype.count = async function (conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.count("onlineCourse", conditions);
  return result;
};

module.exports = OnlineCourseModel;