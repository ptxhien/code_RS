const MajorModel = require("../../models/major");

const MajorController = function () {};

MajorController.prototype.get = async (req, res, next) => {
  const majors = await MajorModel.find("*", "1=1");
  res.status(200).json({ data: majors });
};

module.exports = new MajorController();
