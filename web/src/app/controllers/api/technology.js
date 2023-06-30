const TectnologyModel = require("../../models/technology");

const TechnologyController = function () {};

TechnologyController.prototype.get = async (req, res, next) => {
  const jobs = await TectnologyModel.find("*", "1=1");
  res.status(200).json({ data: jobs });
};

module.exports = new TechnologyController();
