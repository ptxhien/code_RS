const db = require("../../db");
const utilities = require("../../utilities/functions");


const InvoiceModel = function (Obj) {
  this.InvoiceNo = Obj ? Obj.InvoiceNo : undefined;
  this.LearnerID = Obj ? Obj.LearnerID : undefined;
  this.CourseID = Obj ? Obj.CourseID : undefined;
  this.Quality = Obj ? Obj.Quality : undefined;
  this.ItemPrice = Obj ? Obj.ItemPrice : undefined;
  this.InvoiceDate = new Date().toLocaleDateString("en-ZA");
};



InvoiceModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'Invoice'");
    const newID = utilities.createID(nextIndex, 3, "");
    this.InvoiceNo = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='Invoice'");
    const [res1] = await db.insert("Invoice", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


InvoiceModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'Invoice'");
    const newID = utilities.createID(nextIndex, 3, "")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};


InvoiceModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("Invoice", fileds, conditions);
  return result;
};


InvoiceModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("Invoice", fileds, conditions);
  return result;
};


InvoiceModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("Invoice", fileds, conditions);
  return result;
};


InvoiceModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("Invoice", fileds, conditions);
  return new InvoiceModel
(result);
};

InvoiceModel.prototype.update = async function () {};

InvoiceModel.update = async function (obj) {
  try {
    const [res1] = await db.update("Invoice", obj, `InvoiceNo='${obj.InvoiceNo}'`);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to update record!!");
  }
};


module.exports = InvoiceModel;