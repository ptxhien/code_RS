const JobModel = require("../models/job");
const jwt = require("jsonwebtoken");
const OnlineCourseModel = require("../models/onlineCourse");
const OfflineCourseModel = require("../models/offlineCourse");
const { clamp, sortObjectToArray } = require("../../utilities/functions");
const axios = require("axios").default; 


const CourseController = function () {};

// [GET] /courses
CourseController.prototype.get = async (req, res, next) => {
  // handling pagination
  let data = [] 
    page = 1,
    perPage = 9,
    query = req.query;
  if (query.hasOwnProperty("page")) {
    page = parseInt(query.page);
    perPage = query.perPage ? query.perPage : 9;
  }

  // calculating data
  const onlineCount = (await OnlineCourseModel.count())["count"];
  const offlineCount = (await OfflineCourseModel.count())["count"];
  const total = onlineCount + offlineCount;
  const onlinePages = Math.ceil(onlineCount / perPage); // a number of online course pages
  const totalPages = Math.ceil(total / perPage); // a number of total course pages
  const offlineOffset = onlinePages * perPage - onlineCount;
  // get data
  page = clamp(page, 1, totalPages);
  if (page < onlinePages) {
    data = await OnlineCourseModel.find("*", "1=1", true, page, perPage);
  } else if (page == onlinePages) {
    const onlineCourses = await OnlineCourseModel.find(
      "*",
      "1=1",
      true,
      page,
      perPage
    );
    const offlineCourses = await OfflineCourseModel.find(
      "*",
      "1=1",
      true,
      1,
      offlineOffset
    );
    data = [...onlineCourses, ...offlineCourses];
  } else {
    data = await OfflineCourseModel.find(
      "*",
      "1=1",
      true,
      page - onlinePages,
      perPage
    );
  }

  // return pagination nav
  const pagination = {
    hasPrev: true,
    hasNext: true,
    page,
    nav: [1, 2, 3, 4, 5],
  };
  if (page < 3) {
    if (page === 1) {
      pagination.hasPrev = false;
    }
    pagination.nav = [1, 2, 3, 4, 5];
  } else if (page > 2 && page < totalPages - 2) {
    pagination.nav = [page - 2, page - 1, page, page + 1, page + 2];
  } else {
    if (page === totalPages) {
      pagination.hasNext = false;
    }
    pagination.nav = [
      totalPages - 4,
      totalPages - 3,
      totalPages - 2,
      totalPages - 1,
      totalPages,
    ];
  }

  // get jobs
  const jobs = await JobModel.find("*", "1=1");

  res.render("sites/courses/index", {
    title: "Courses",
    css: "dynamic_css/courses",
    js: "dynamic_js/courses",
    nav: "courses",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Courses", link: "" },
    ],
    jobTitle: "",
    jobID: "",
    form: "",
    durationFrom: "0",
    durationTo: "7",
    data,
    hasPagination: true,
    pagination,
    jobs,
    skillsProvided: [],
    skillsNotProvided: [],
    hasMissingSkills: false
  });
};

// [POST] /courses
CourseController.prototype.recommend = async (req, res, next) => {

  const prefix = "00";
  const from = (prefix).substring(0, 2 - req.body?.durationFrom?.length) + req.body.durationFrom;
  const to = (prefix).substring(0, 2 - req.body?.durationTo?.length) + req.body.durationTo;
  const body = {
    // occupation: req.body?.jobTitle || "",
    occupation: req.body?.jobID || "",
    form: req.body?.form || "",
    month: `${from}-${to}`,
    email: req.user?.email,
  };

  const qs = Object.keys(body).reduce((prev, curr) => prev + `&${curr}=${body[curr]}`, '?q')

  const RS_API = process.env.RS_API || "http://127.0.0.1:6868";

  const rsRes = await axios.get(RS_API + "/recommendation" + qs);
  const data = rsRes.data.df_rule_ngoaile[0].course_lan_remain;
  const skillsProvided = rsRes.data.df_rule_ngoaile[0].lstSkill_Provider_ngoaile;
  const skillsNotProvided = Object.keys(rsRes.data.df_rule_ngoaile[0].lstSkill_notProvider_ngoaile);
  const sortedProvidedSkills = sortObjectToArray(skillsNotProvided);
  sortedProvidedSkills.unshift("All");

  // get jobs
  const jobs = await JobModel.find("*", "1=1");

  res.render("sites/courses/index", {
    title: "Courses",
    css: "dynamic_css/courses",
    js: "dynamic_js/courses",
    nav: "courses",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Courses", link: "" },
    ],
    jobTitle: req.body?.jobTitle || "",
    jobID: req.body?.jobID || "",
    form: req.body?.form || "",
    durationFrom: parseInt(from),
    durationTo: parseInt(to),
    data,
    hasPagination: false,
    pagination: {
      hasPrev: true,
      hasNext: true,
      page: 1,
      nav: [1, 2, 3, 4, 5],
    },
    jobs,
    skillsProvided: sortedProvidedSkills,
    skillsNotProvided,
    hasMissingSkills: true,
  });
};

// [GET] /courses/:id
CourseController.prototype.getOne = async (req, res, next) => {
  const courseID = req.params.id;

  let course;

  course = await OnlineCourseModel.findOne("*", `courseID='${courseID}'`);
  if (!course.courseID){
    course = await OfflineCourseModel.findOne("*", `courseID='${courseID}'`);
  }
  if (!course.courseID){
    res.status(404).render("sites/err/404", {
      title: "Not found",
      css: "dynamic_css/index",
      js: "dynamic_js/index",
      nav: "Not found",
    });
    return;
  }


  const tags = course.technologySkill.split(", ");

  res.status(200).render("sites/courses/detail", {
    title: course.courseTitle,
    css: "dynamic_css/course",
    js: "dynamic_js/course",
    nav: "courses",
    breadcrumb: [
      { name: "Home", link: "/" },
      { name: "Courses", link: "/courses" },
      { name: course.courseTitle, link: "" },
    ],
    course,
    tags,
  });
};

module.exports = new CourseController();
