const db = require("../../db");
const utilities = require("../../utilities/functions");

const JobModel = function (Obj) {
  this.jobID = Obj ? Obj.jobID : undefined;
  this.jobTitle = Obj ? Obj.jobTitle : undefined;
  this.technologySkill = Obj ? Obj.technologySkill : undefined;
  this.weightTechnology = Obj ? Obj.weightTechnology : undefined;
};

JobModel.prototype.create = async function () {
  try {
    const [[{ nextIndex }]] = await db.get(
      "tables",
      "*",
      "tableName = 'job'"
    );
    const newID = utilities.createID(nextIndex, 3, "LAN_");
    this.jobID = newID;
    const [res0] = await db.update(
      "tables",
      { nextIndex: nextIndex + 1 },
      "tableName='job'"
    );
    const [res1] = await db.insert("job", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

JobModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{ nextIndex }]] = await db.get(
      "tables",
      "*",
      "tableName = 'job'"
    );
    const newID = utilities.createID(nextIndex, 3, "LAN_");
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

JobModel.prototype.find = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [result] = await db.get("job", fileds, conditions);
  return result;
};

JobModel.find = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [result] = await db.get("job", fileds, conditions);
  return result;
};

JobModel.prototype.findOne = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [[result]] = await db.get("job", fileds, conditions);
  return result;
};

JobModel.findOne = async function (
  fileds = "*" || ["*"],
  conditions = "1=1" || ["1=1"]
) {
  const [[result]] = await db.get("job", fileds, conditions);
  return new JobModel(result);
};

JobModel.prototype.update = async function () {};

JobModel.update = async function (obj) {};

module.exports = JobModel;
