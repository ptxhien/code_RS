const LanguageModel = require("../../models/language");

const LanguageController = function () {};

LanguageController.prototype.get = async (req, res, next) => {
  const languages = await LanguageModel.find("*", "1=1");
  res.status(200).json({ data: languages });
};

module.exports = new LanguageController();
