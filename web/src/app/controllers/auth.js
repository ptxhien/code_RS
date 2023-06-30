const LearnerModel = require("../models/learner");
const TechnologyModel = require("../models/technology");
const LanguageModel = require("../models/language");
const jwt = require("jsonwebtoken");
const axios = require("axios").default;

const AuthController = function () {};

// [GET] /auth/registry
AuthController.prototype.registry = async (req, res, next) => {
  const technologies = await TechnologyModel.find();
  const languages = await LanguageModel.find();
  const RS_API = process.env.RS_API || "http://127.0.0.1:6868";

  let freetimeSlots = [];
  try {
    const freetimeRes = await axios.get(RS_API + "/freetime");
    freetimeSlots = freetimeRes.data.lst_StudyTime;
  } catch (err) {
    freetimeSlots = [
      "08:00-17:00 (2, 4, 6)",
      "08:00-17:00 (7 days)",
      "08:30-11:30 (7)",
      "18:00-21:00 (2, 4, 6)",
      "18:00-21:00 (3, 5)",
    ];
  }



  res.render("sites/register/index", {
    title: "Register",
    css: "dynamic_css/register",
    js: "dynamic_js/register",
    nav: "register",
    layout: "register",
    breadcrumb: [],
    technologies, 
    languages,
    freetimeSlots,
  });
};

// [POST] /auth/registry
AuthController.prototype.submit = async (req, res, next) => {
  const newLearner = new LearnerModel(req.body);
  try {
    const oldUser = await LearnerModel.findOne("*", `email='${email}'`);

    if (oldUser.email) {
      res.status(400).json({ message: "this account was already registried" });
      return;
    }
    
    newLearner.hash();
    const result = await newLearner.create();
    const authToken = jwt.sign({ username: newLearner.fullname, id: newLearner.learnerID, email: newLearner.email }, "zFUVn{;Sd4!]#lN");
    res.status(200).cookie("authToken", authToken).json({ authToken });
  } catch (err) {
    res.status(500).json({ message: "failed to add a new record!" });
  }
};

// [POST] /auth/login
AuthController.prototype.login = async (req, res, next) => {
  const {email, password} = req.body;
  try {
    const user = await LearnerModel.findOne("*", `email='${email}'`);
    if (user.email && user.compare(password)){
      const authToken = jwt.sign(
        {
          username: user.fullname,
          id: user.learnerID,
          email: user.email,
        },
        "zFUVn{;Sd4!]#lN"
      );
      res.status(200).cookie("authToken", authToken).json({ authToken });
    } else {
      res.status(400).json({ message: "email or password is not correct" });
    }
    return;
  } catch (err) {
    console.log(err.message);
    res.status(500).json({ message: "Opps! there was a problem. Try again later :("});
    return;
  }
  
};

// [POST] /auth/logout
AuthController.prototype.logout = async (req, res, next) => {
  res.status(200).json({message: "logout successfully!"});
};

// AuthController.prototype.hashAll = async (req, res, next) => {
//   try {
//     await LearnerModel.resetAllPassword();
//     res.json({message: "done"}).status(200);
//   } catch (err) {
//     console.error(err.message);
//     res.json({ message: "error" }).status(401);
//   }
// };
module.exports = new AuthController();
