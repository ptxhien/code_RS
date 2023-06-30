const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const LearnerModel = function (Obj) {
  this.learnerID = Obj ? Obj.learnerID : undefined;
  this.email = Obj ? Obj.email : undefined;
  this.password = Obj ? Obj.password : undefined;
  this.fullname = Obj ? Obj.fullname : undefined;
  this.address = Obj ? Obj.address : undefined;
  this.address1 = Obj ? Obj.address1 : undefined;
  this.gender = Obj ? Obj.gender : undefined;
  this.learnerLevel = Obj ? Obj.learnerLevel : undefined;
  this.language = Obj ? Obj.language : undefined;
  this.technologySkill = Obj ? Obj.technologySkill : undefined;
  this.fieldOfStudy = Obj ? Obj.fieldOfStudy : undefined;
  this.jobNow = Obj ? Obj.jobNow : undefined;
  this.feeMax = Obj ? Obj.feeMax : undefined;
  this.freeTime = Obj ? Obj.freeTime : undefined;
  this.futureSelfDevelopment = Obj ? Obj.futureSelfDevelopment : undefined;
};



LearnerModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'learner'");
    const newID = utilities.createID(nextIndex, 3, "LEARNER_");
    this.learnerID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='learner'");
    const [res1] = await db.insert("learner", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


LearnerModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'learner'");
    const newID = utilities.createID(nextIndex, 3, "LEARNER_")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


LearnerModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("learner", fileds, conditions);
  return result;
};


LearnerModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("learner", fileds, conditions);
  return result;
};


LearnerModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("learner", fileds, conditions);
  return new LearnerModel(result);
};


LearnerModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("learner", fileds, conditions);
  return new LearnerModel(result);
};

LearnerModel.prototype.update = async function (obj) {};


LearnerModel.update = async function (obj) {
  try {
    const [res1] = await db.update("learner", obj, `learnerID='${obj.learnerID}'`);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


LearnerModel.prototype.hash = function () {
  const salt = bcrypt.genSaltSync(10);
  this.password = bcrypt.hashSync(this.password, salt);
  console.log("password has been encrypted");
};

LearnerModel.prototype.hash = function () {
  const salt = bcrypt.genSaltSync(10);
  this.password = bcrypt.hashSync(this.password, salt);
  console.log("password has been encrypted");
};

LearnerModel.prototype.compare = function (password = "abc") {
  return bcrypt.compareSync(password, this.password);
};

// this is bad function (just uncomment this function if u really need)
// LearnerModel.resetAllPassword = async function(password = "12345678"){
//   const salt = bcrypt.genSaltSync(10);
//   password = bcrypt.hashSync(password, salt);
//   const [result] = await db.update("learner", {password});
//   return result;
// }
module.exports = LearnerModel;