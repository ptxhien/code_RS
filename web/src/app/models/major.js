const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const MajorModel = function (Obj) {
  this.subjectID = Obj ? Obj.subjectID : undefined;
  this.subjectName = Obj ? Obj.subjectName : undefined;
};



MajorModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'major'");
    const newID = utilities.createID(nextIndex, 3, "LAN_");
    this.subjectID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='major'");
    const [res1] = await db.insert("major", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


MajorModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'major'");
    const newID = utilities.createID(nextIndex, 3, "LAN_")
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


MajorModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("major", fileds, conditions);
  return result;
};


MajorModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("major", fileds, conditions);
  return result;
};


MajorModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("major", fileds, conditions);
  return result;
};


MajorModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("major", fileds, conditions);
  return new MajorModel(result);
};

MajorModel.prototype.update = async function () {};


MajorModel.update = async function (obj) {};


module.exports = MajorModel;