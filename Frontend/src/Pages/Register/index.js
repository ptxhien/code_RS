import React, { Fragment, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import Slider from "react-slick";
import {Redirect} from "react-router-dom";

import bg3 from "../../assets/utils/images/originals/citynights.jpg";

import {
  Col,
  Row,
  Button,
  Form,
  FormGroup,
  Label,
  Input,
  Card,
  CardBody,
} from "reactstrap";
import MultiStep from "../Forms/Elements/Wizard/Wizard";
import Step1 from "./Steps/step1";
import Step2 from "./Steps/step2";
import Step3 from "./Steps/step3";
import {
  GetAllLanguageAction,
  GetAllTechnologyAction,
  GetFreeTimeAction,
  GetAllMajorAction,
} from "../../redux/masterdata/masterDataAction";
import { registerAction } from "../../redux/actions/account/accountAction";
import { toastErrorText } from "../../helpers/toastify";

export default function Register() {
  let settings = {
    dots: true,
    infinite: true,
    speed: 500,
    arrows: true,
    slidesToShow: 1,
    slidesToScroll: 1,
    fade: true,
    initialSlide: 0,
    autoplay: true,
    adaptiveHeight: true,
  };

  const dispatch = useDispatch();
  const accountReducer = useSelector(state => state.accountReducer);

  const [DTO, setDTO] = useState({
    email: "",
    password: "",
    fullname: "",
    gender: "",
    city_id: "",
    city: "",
    district_id: "",
    district: "",
    ward_id: "",
    ward: "",
    learnerLevel: "",
    language: [],
    jobNow: "",
    technologySkill: [],
    feeMax: "",
    feeMaxText: "",
    freeTime: [],
    futureSelfDevelopment: "",
    fieldOfStudy: "",
    address1: '',
    cities: [],
    districts: [],
    wards: [],
  });
  const steps = [
    {
      name: "Account Information",
      component: <Step1 DTO={DTO} setDTO={setDTO} />,
      shouldGoNext: false,
      errors: [],
    },
    {
      name: "Personal Information",
      component: <Step2 DTO={DTO} setDTO={setDTO} />,
      shouldGoNext: false,
      errors: [],
    },
    {
      name: "Finish Register",
      component: <Step3 DTO={DTO} onClickRegister={onClickRegister} />,
      shouldGoNext: true,
    },
  ];
  
  const [validSteps, setValidSteps] = useState([
    { shouldGoNext: false, errors: [] },
    { shouldGoNext: false, errors: [] },
    { shouldGoNext: true, errors: [] },
  ]);

  useEffect(() => {
    dispatch(GetAllLanguageAction());
    dispatch(GetAllTechnologyAction());
    dispatch(GetFreeTimeAction());
    dispatch(GetAllMajorAction());
  }, []);

  function onClickRegister() {
    let addressArray = [];
    if (DTO.ward) {
        addressArray.push(DTO.ward)
    }
    if (DTO.district) {
        addressArray.push(DTO.district)
    }
    if (DTO.city) {
        addressArray.push(DTO.city)
    }
    let postdata = {
      email: DTO.email,
      password: DTO.password,
      fullname: DTO.fullname,
      gender: DTO.gender,
      address: addressArray.length ? addressArray.join(', ') : null,
      address1: DTO.address1 ? DTO.address1 : null,
      learnerLevel: DTO.learnerLevel,
      language: DTO.language.map(({lanName}) => lanName).join(', '),
      jobNow: DTO.jobNow,
      technologySkill: DTO.technologySkill.map(({techName}) => techName).join(', '),
      fieldOfStudy: DTO.fieldOfStudy,
      feeMax: DTO.feeMax,
      freeTime: DTO.freeTime && DTO.freeTime.join('|') || '',
      futureSelfDevelopment: DTO.futureSelfDevelopment,
    }
    dispatch(registerAction(postdata));
  }
  
  const checkFirstStep = () => {
    const errors = [];

    if (!/^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/g.test(DTO.email)){
      errors.push("Email is not valid!");
    }
    if (!/^(?=.*?[a-z|A-Z|0-9]).{8,}$/g.test(DTO.password)) {
      errors.push("Password must have at least 8 characters");
    }
    if (!DTO.fullname.trim()) {
      errors.push("Invalid name");
    }
    
    if (errors.length) {
      return {
        shouldGoNext: false,
        errors,
      };
    }

    return {shouldGoNext: true, errors: []};
  }

  const checkSecondStep = () => {
    const errors = [];
    if(!(DTO.language && DTO.language.length > 0)) {
      errors.push("Please choose at least one language you know");
    }
    if (!(DTO.technologySkill && DTO.technologySkill.length > 0)) {
      errors.push("Please choose at least one technology skill you know");
    }

    let count = 0;
    if (DTO.address1) {
        count++;
    }
    if (DTO.ward || (DTO.wards.length == 0 && DTO.districts.length > 0)) {
        count++;
    }
    if (DTO.district) {
        count++;
    }
    if (DTO.city) {
        count++;
    }
    if (count > 0 && count < 4) {
      // errors.push("bạn phải nhập đầy đủ địa chỉ");
      errors.push("You must type in the complete address (including house number, ward, district, city)")
      
    }

    if (errors.length) {
      return {
        shouldGoNext: false,
        errors,
      };
    }

    return { shouldGoNext: true, errors: [] };
  }

  useEffect(() => {
    const newValidSteps = [...validSteps];
    newValidSteps[0] = checkFirstStep();
    newValidSteps[1] = checkSecondStep();
    setValidSteps(newValidSteps);

  }, [DTO]);

  useEffect(() =>{
    if (accountReducer.error){
      toastErrorText(accountReducer.error);
    }
  }, [accountReducer]);

  return (
    <Fragment>
      {accountReducer.isLogged ? 
      (<Redirect to="/"></Redirect>) : 
      (<div className="">
        <Row className="no-gutters">
          <Col lg="7" md="12" className="">
            <Card className="main-card mb-3 ">
              <CardBody>
                <Col lg="9" md="10" sm="12" className="mx-auto app-login-box">
                  <div className="app-logo" />
                  <h4>
                    <div>Welcome,</div>
                    <span>
                      It only takes a{" "}
                      <span className="text-success">few seconds</span> to
                      create your account
                    </span>
                  </h4>
                </Col>
                <MultiStep showNavigation={true} steps={steps} validSteps={validSteps} />
              </CardBody>
            </Card>
          </Col>
          <Col lg="5" className="d-lg-flex d-xs-none">
            <div className="slider-light">
              <Slider {...settings}>
                <div className="h-100 d-flex justify-content-center align-items-center bg-premium-dark">
                  <div
                    className="slide-img-bg"
                    style={{
                      backgroundImage: "url(" + bg3 + ")",
                    }}
                  />
                  <div className="slider-content">
                    <h3>Scalable, Modular, Consistent</h3>
                    <p>
                      Easily exclude the components you don't require.
                      Lightweight, consistent Bootstrap based styles across all
                      elements and components
                    </p>
                  </div>
                </div>
              </Slider>
            </div>
          </Col>
        </Row>
      </div>)}
    </Fragment>
  );
}
