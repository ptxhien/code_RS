const LearnerModel = require("../../models/learner");
const TechnologyModel = require("../../models/technology");
const LanguageModel = require("../../models/language");
const jwt = require("jsonwebtoken");

const AuthController = function () {};

// [POST] /auth/registry
AuthController.prototype.register = async (req, res, next) => {
  const newLearner = new LearnerModel(req.body);
  try {
    const oldUser = await LearnerModel.findOne("*", `email='${newLearner.email}'`);

    if (oldUser.email) {
      res.status(400).json({ message: "this account was already registried" });
      return;
    }

    newLearner.hash();
    const result = await newLearner.create();
    const authToken = jwt.sign(
      {
        username: newLearner.fullname,
        id: newLearner.learnerID,
        email: newLearner.email,
      },
      "zFUVn{;Sd4!]#lN" || "privatekey"
    );
    delete newLearner.password;
    const learner = await LearnerModel.findOne("*", `learnerID='${newLearner.learnerID}'`);
    res.status(200).header("authToken", authToken).json({ authToken, user: learner });
  } catch (err) {
    res.status(500).json({ message: "failed to register!" });
  }
};

// [POST] /auth/login
AuthController.prototype.login = async (req, res, next) => {
  const { email, password } = req.body;
  try {
    const user = await LearnerModel.findOne("*", `email='${email}'`);
    if (user.email && user.compare(password)) {
      const authToken = jwt.sign(
        {
          username: user.fullname,
          id: user.learnerID,
          email: user.email,
        },
        "zFUVn{;Sd4!]#lN"
      );
      delete user.password;
      res.status(200).header("authToken", authToken).json({ authToken, user });
    } else {
      res.status(400).json({ message: "email or password is not correct" });
    }
    return;
  } catch (err) {
    console.log(err.message);
    res
      .status(500)
      .json({ message: "Opps! there was a problem. Try again later :(" });
    return;
  }
};

// [POST] /auth/update
AuthController.prototype.update = async (req, res, next) => {
  try {
    var userDecoded = jwt.verify(req.get('auth'), "zFUVn{;Sd4!]#lN");
    var params = req.body;
    params.learnerID = userDecoded.id;
    const result = await LearnerModel.update(params);
    const oldUser = await LearnerModel.findOne("*", `learnerID='${params.learnerID}'`);

    res.status(200).json({user: oldUser, message: "successfully updated" });
  } catch (err) {
    res.status(500).json({ message: "failed to update!" });
  }
};

module.exports = new AuthController();
