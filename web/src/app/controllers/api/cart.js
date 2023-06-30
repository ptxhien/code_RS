const InvoiceModel = require("../../models/invoice");
const CartModel = require("../../models/cart");
const OnlineCourseModel = require("../../models/onlineCourse");
const OfflineCourseModel = require("../../models/offlineCourse");
const LearnerModel = require("../../models/learner");
const jwt = require("jsonwebtoken");

const CartController = function () {};

CartController.prototype.get = async (req, res, next) => {
  const { id: LearnerID } = req.user;

  const carts = await CartModel.find("*", `LearnerID = "${LearnerID}"`);
  let CourseIDs = carts.map(a => a.CourseID);
  let courses = []; 
  if (CourseIDs && CourseIDs.length > 0) {
    let offlinecourses = await OfflineCourseModel.find("*", `courseID in ('${CourseIDs.join("','")}')`);
    courses = offlinecourses;
  }
  if (CourseIDs && CourseIDs.length > 0) {
    let onlinecourses = await OnlineCourseModel.find("*", `courseID in ('${CourseIDs.join("','")}')`);
    courses = courses.concat(onlinecourses);
  }
  for (var i = 0; i < carts.length; i++) {
    let findcourse = courses.find(item => item.courseID == carts[i].CourseID);
    carts[i].course = findcourse || {};
  }
  res.status(200).json({ data: carts });
};

CartController.prototype.post = async (req, res, next) => {
  //try {
    var userDecoded = jwt.verify(req.get('auth'), "zFUVn{;Sd4!]#lN");
    let LearnerID = userDecoded.id;
    const { CourseID } = req.body;
    const findInvoices = await InvoiceModel.find("*", `LearnerID="${LearnerID}" and CourseID="${CourseID}"`)
    const findCarts = await CartModel.find("*", `LearnerID="${LearnerID}" and CourseID="${CourseID}"`)
    if (findInvoices.length || findCarts.length) {
      res.status(402).json({msg: "Course bought"})
    } else {
      const newCart = new CartModel({
        LearnerID,
        CourseID,
      });

      const _ = await newCart.create();
      res.status(200).json({
        cart: newCart,
      });
    }
  //} catch (err) {
    res.status(500).json({msg: "Something wrong when create a new invoice. Please try again"})
  //}

};
module.exports = new CartController();
