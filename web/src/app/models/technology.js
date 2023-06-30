const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const TechnologyModel = function (Obj) {
  this.techID = Obj ? Obj.techID : undefined;
  this.techName = Obj ? Obj.techName : undefined;
};



TechnologyModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'technology'");
    const newID = utilities.createID(nextIndex, 3, "TEC_");
    this.technologyID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='technology'");
    const [res1] = await db.insert("technology", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


TechnologyModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'technology'");
    const newID = utilities.createID(nextIndex, 3, "TEC_")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


TechnologyModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("technology", fileds, conditions);
  return result;
};


TechnologyModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("technology", fileds, conditions);
  return result;
};


TechnologyModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("technology", fileds, conditions);
  return result;
};


TechnologyModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("technology", fileds, conditions);
  return new TechnologyModel(result);
};

TechnologyModel.prototype.update = async function () {};


TechnologyModel.update = async function (obj) {};


module.exports = TechnologyModel;