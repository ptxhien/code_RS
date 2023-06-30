const db = require("../../db");
const utilities = require("../../utilities/functions");

const getCurrentDatetime = () => {
  const date = new Date();
  const dateStr =
    date.getFullYear() +
    "/" +
    ("00" + (date.getMonth() + 1)).slice(-2) +
    "/" +
    ("00" + date.getDate()).slice(-2) +
    " " +
    ("00" + date.getHours()).slice(-2) +
    ":" +
    ("00" + date.getMinutes()).slice(-2) +
    ":" +
    ("00" + date.getSeconds()).slice(-2);
  console.log(dateStr);
  return dateStr;
}

const RatingLearnerModel = function (Obj) {
  this.ID = Obj ? Obj.ID : undefined;
  this.LearnerID = Obj ? Obj.LearnerID : undefined;
  this.Rating = Obj ? Obj.Rating : undefined;
  this.RatingDate = getCurrentDatetime();
};

RatingLearnerModel.prototype.create = async function () {
  try {
    const [[{ nextIndex }]] = await db.get(
      "tables",
      "*",
      "tableName = 'RatingLearner'"
    );
    const newID = utilities.createID(nextIndex, 3, "");
    this.ID = newID;
    const [res0] = await db.update(
      "tables",
      { nextIndex: nextIndex + 1 },
      "tableName='RatingLearner'"
    );
    const [res1] = await db.insert("RatingLearner", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

RatingLearnerModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{ nextIndex }]] = await db.get(
      "tables",
      "*",
      "tableName = 'RatingLearner'"
    );
    const newID = utilities.createID(nextIndex, 3, "LAN_");
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

RatingLearnerModel.prototype.find = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [result] = await db.get("RatingLearner", fileds, conditions);
  return result;
};

RatingLearnerModel.find = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [result] = await db.get("RatingLearner", fileds, conditions);
  return result;
};

RatingLearnerModel.prototype.findOne = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [[result]] = await db.get("RatingLearner", fileds, conditions);
  return result;
};

RatingLearnerModel.findOne = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [[result]] = await db.get("RatingLearner", fileds, conditions);
  return new RatingLearnerModel(result);
};

RatingLearnerModel.prototype.update = async function () {};

RatingLearnerModel.update = async function (obj) {};

module.exports = RatingLearnerModel;
