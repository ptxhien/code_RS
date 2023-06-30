const db = require("../../db");
const utilities = require("../../utilities/functions");


const CartModel = function (Obj) {
  this.CartID = Obj ? Obj.CartID : undefined;
  this.LearnerID = Obj ? Obj.LearnerID : undefined;
  this.CourseID = Obj ? Obj.CourseID : undefined;
  this.Status = Obj ? Obj.Status : undefined;
};

CartModel.prototype.create = async function () {
  try {
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'Cart'");
    const newID = utilities.createID(nextIndex, 3, "");
    this.CartID = newID;
    const [res0] = await db.update("tables", { nextIndex: nextIndex + 1}, "tableName='Cart'");
    const [res1] = await db.insert("Cart", this);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

CartModel.create = async function (obj) {
  try {
    // get next index for specific table
    const [[{nextIndex}]] = await db.get("tables", '*', "tableName = 'Cart'");
    const newID = utilities.createID(nextIndex, 3, "")
    console.log(newID);
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to create new record!!");
  }
};

CartModel.prototype.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("Cart", fileds, conditions);
  return result;
};

CartModel.find = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [result] = await db.get("Cart", fileds, conditions);
  return result;
};

CartModel.prototype.findOne = async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("Cart", fileds, conditions);
  return result;
};

CartModel.findOne =  async function (fileds = "*" || ["*"], conditions = "1=1" || ["1=1"]) {
  const [[result]] = await db.get("Cart", fileds, conditions);
  return new CartModel
(result);
};

CartModel.prototype.update = async function () {};

CartModel.update = async function (obj) {
  try {
    const [res1] = await db.update("Cart", obj, `CartID='${obj.CartID}'`);
    return res1;
  } catch (err) {
    console.error(err.message);
    throw new Error("fail to update record!!");
  }
};

module.exports = CartModel;