const db = require("../../db");
const utilities = require("../../utilities/functions");
const bcrypt = require("bcryptjs");


const LanguageModel = function (Obj) {
  this.languageID = Obj ? Obj.languageID : undefined;
  this.lanName = Obj ? Obj.lanName : undefined;
};



LanguageModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'language'");
    const newID = utilities.createID(nextIndex, 3, "LAN_");
    this.languageID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='language'");
    const [res1] = await db.insert("language", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


LanguageModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'language'");
    const newID = utilities.createID(nextIndex, 3, "LAN_")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


LanguageModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("language", fileds, conditions);
  return result;
};


LanguageModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("language", fileds, conditions);
  return result;
};


LanguageModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("language", fileds, conditions);
  return result;
};


LanguageModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("language", fileds, conditions);
  return new LanguageModel(result);
};

LanguageModel.prototype.update = async function () {};


LanguageModel.update = async function (obj) {};


module.exports = LanguageModel;