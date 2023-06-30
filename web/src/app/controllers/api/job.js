const JobModel = require("../../models/job");

const JobController = function () {};

JobController.prototype.get = async (req, res, next) => {
  const jobs = await JobModel.find("*", "1=1");
  res.status(200).json({ data: jobs });
};

module.exports = new JobController();
