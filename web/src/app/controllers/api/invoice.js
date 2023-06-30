const InvoiceModel = require("../../models/invoice");
const OnlineCourseModel = require("../../models/onlineCourse");
const OfflineCourseModel = require("../../models/offlineCourse");
const LearnerModel = require("../../models/learner");
const jwt = require("jsonwebtoken");

const InvoiceController = function () {};

InvoiceController.prototype.get = async (req, res, next) => {
  const { id: LearnerID } = req.user;

  const Invoices = await InvoiceModel.find("*", `LearnerID = "${LearnerID}"`);
  let CourseIDs = Invoices.map(a => a.CourseID);
  let courses = []; 
  if (CourseIDs && CourseIDs.length > 0) {
    let offlinecourses = await OfflineCourseModel.find("*", `courseID in ('${CourseIDs.join("','")}')`);
    courses = offlinecourses;
  }
  if (CourseIDs && CourseIDs.length > 0) {
    let onlinecourses = await OnlineCourseModel.find("*", `courseID in ('${CourseIDs.join("','")}')`);
    courses = courses.concat(onlinecourses);
  }
  for (var i = 0; i < Invoices.length; i++) {
    let findcourse = courses.find(item => item.courseID == Invoices[i].CourseID);
    Invoices[i].course = findcourse || {};
  }
  res.status(200).json({ data: Invoices });
};

InvoiceController.prototype.post = async (req, res, next) => {
  try {
    var userDecoded = jwt.verify(req.get('auth'), "zFUVn{;Sd4!]#lN");
    let LearnerID = userDecoded.id;
    const { CourseID, Quality, ItemPrice } = req.body;
    const finds = await InvoiceModel.find("*", `LearnerID="${LearnerID}" and CourseID="${CourseID}"`)
    if (finds.length) {
      res.status(200).json({msg: ""});
      // res.status(402).json({msg: "Course bought" + finds[0].CourseID});
    } else {
      const newInvoice = new InvoiceModel({
        LearnerID,
        CourseID,
        Quality,
        ItemPrice,
      });

      const _ = await newInvoice.create();
      res.status(200).json({
        invoice: newInvoice,
      });
    }
  } catch (err) {
    res.status(500).json({msg: "Something wrong when create a new invoice. Please try again"})
  }

};
InvoiceController.prototype.postCompleted = async (req, res, next) => {
  try {
    var userDecoded = jwt.verify(req.get('auth'), "zFUVn{;Sd4!]#lN");
    let LearnerID = userDecoded.id;
    const { InvoiceNo, CourseID } = req.body;

    let user = await LearnerModel.findOne("*", `learnerID='${LearnerID}'`);
    let technologySkills = user.technologySkill ? user.technologySkill.split(', ') : [];
    const offlineCourse = await OfflineCourseModel.findOne("*", `courseID="${CourseID}"`);
    if (offlineCourse && offlineCourse.technologySkill){
      technologySkills = technologySkills.concat(offlineCourse.technologySkill.split(', '));
    }
    const onlineCourse = await OnlineCourseModel.findOne("*", `courseID="${CourseID}"`);
    if(onlineCourse && onlineCourse.technologySkill) {
      technologySkills = technologySkills.concat(onlineCourse.technologySkill.split(', '));
    }
    technologySkills = [...new Set(technologySkills)];

    var params = {
      learnerID: user.learnerID,
      technologySkill: technologySkills.join(', ')
    };
    await LearnerModel.update(params);
    await InvoiceModel.update({
      InvoiceNo: InvoiceNo,
      Completed: 1
    });

    invoice = await InvoiceModel.findOne("*", `InvoiceNo='${InvoiceNo}'`);
    user = await LearnerModel.findOne("*", `learnerID='${LearnerID}'`);
    res.status(200).json({
      invoice: invoice,
      user: user
    });
  } catch (err) {
    res.status(500).json({msg: "Something wrong when create a new invoice. Please try again"})
  }

};

InvoiceController.prototype.checkExist = async (req, res, next) => {
  try {
    var userDecoded = jwt.verify(req.get('auth'), "zFUVn{;Sd4!]#lN");
    let LearnerID = userDecoded.id;
    const { CourseID, Quality, ItemPrice } = req.body;
    const finds = await InvoiceModel.find("*", `LearnerID="${LearnerID}" and CourseID="${CourseID}"`)
    if (finds.length) {
      res.status(402).json({msg: "Course bought"});
    } else {
      res.status(200).json({msg: ""});
    }
  } catch (err) {
    res.status(500).json({msg: "Something wrong when create a new invoice. Please try again"})
  }
};
module.exports = new InvoiceController();
