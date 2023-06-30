const RatingModel = require("../../models/rating");

const RatingController = function () {};

RatingController.prototype.post = async (req, res, next) => {
  const { id: LearnerID } = req.user;
  const { Rating } = req.body;

  const newRating = new RatingModel({
    LearnerID,
    Rating,
  });

  try {
    const _ = await newRating.create();
    res.status(200).json(newRating);
  } catch (err) {
    res.status(500).json({msg: "Something wrong when create a new invoice. Please try again"})
  }
};

module.exports = new RatingController();
