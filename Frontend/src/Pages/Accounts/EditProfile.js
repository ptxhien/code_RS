import React, { Fragment, useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import ThemeOptions from "../../Layout/ThemeOptions";
import AppHeader from "../../Layout/AppHeader";
import { FormGroup, Label, CustomInput, Input, Col, Form, FormText, Button, Alert, ListGroup,ListGroupItem, CloseButton} from "reactstrap";

import avatar1 from "../../assets/utils/images/avatars/2.jpg";
import course1 from "../../assets/utils/images/courses/course-1.jpg";
import image1 from "../../assets/images/slider-img1.jpg";
import image2 from "../../assets/images/slider-img2.jpg";
import image3 from "../../assets/images/slider-img3.jpg";
import { useHistory, useParams } from "react-router";
import http from "../../redux/utils/http";
import Select from "react-select";
import makeAnimated from "react-select/animated";
import axios from "axios";
import {
  GetAllLanguageAction,
  GetAllTechnologyAction,
  GetFreeTimeAction,
  GetAllMajorAction,
} from "../../redux/masterdata/masterDataAction";
import { updateAction } from "../../redux/actions/account/accountAction";
import { toastErrorText, toastSuccessText } from "../../helpers/toastify";

export default function EditProfile() {

  const dispatch = useDispatch();
  const history = useHistory();

  const auth = useSelector((state) => state.accountReducer.user);
  const [DTO, setDTO] = useState({
    city_id: "",
    city: "",
    district_id: "",
    district: "",
    ward_id: "",
    ward: "",
    learnerLevel: "",
    language: "",
    jobNow: "s",
    technologySkill: "",
    feeMax: "",
    feeMaxText: "",
    freeTime: [],
    futureSelfDevelopment: "",
    fieldOfStudy: "",
    address1: "",
    cities: [],
    districts: [],
    wards: [],
  });

  const { lsLanguage, lsFreeTime, lsTechnology, lsMajor } = useSelector((state) => state.masterdataReducer);

  const [techSkill, settechSkill] = useState([])
  const [language, setlanguage] = useState([])
  // const [freeTime, setFreeTime] = useState([])
  const [lstFeeMax, setLstFeeMax] = useState([
          'Less 5 millions VND',
          'From 5 to 15 millions VND',
          'From 15 to 30 millions VND',
          'From 30 to 50 millions VND',
      ]);
  const [cities, setCities] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [wards, setWards] = useState([]);

  const [loading, setLoading] = useState(true);

  const weekdays  = [{
        key: '2',
        text: 'Monday'
    },{
        key: '3',
        text: 'Tuesday'
    },{
        key: '4',
        text: 'Wednesday'
    },{
        key: '5',
        text: 'Thursday'
    },{
        key: '6',
        text: 'Friday'
    },{
        key: '7',
        text: 'Saturday'
    },{
        key: 'CN',
        text: 'Sunday'
    },
  ];
  const [timeStart, setTimeStart] = useState("00:00")
  const [timeEnd, setTimeEnd] = useState("23:59")
  const [weekday, setWeekday] = useState([]);
    
  useLayoutEffect(() => {
      const updateCities = async () => {
          const {data} = await axios.get("https://vapi.vnappmob.com/api/province/");
          // setCities(data.results);
          setDTO(pre => ({ ...pre, cities: data.results }));
      }
      updateCities();

  }, []);

  useEffect(() => {
      setDTO((prev) => {
        return {...prev, technologySkill: [{techName: "API"}, {techName:"AI"}],
          feeMax: auth.feeMax,
          learnerLevel: auth.learnerLevel,
          futureSelfDevelopment: auth.futureSelfDevelopment,
          jobNow: auth.jobNow,
          address1: auth.address1 || '',
          fieldOfStudy: auth.fieldOfStudy
        }
      });
    if(auth.freeTime) {
      let freeTimes = auth.freeTime.split('|');
      setDTO((prev) => {
        return {...prev, freeTime: freeTimes}
      });
    }
    if (auth.language) {
      setlanguage(auth.language.split(', ').map((el) => ({lanName: el})));
    }
    if (auth.technologySkill) {
      settechSkill(auth.technologySkill.split(', ').map((el) => ({techName: el})));
    }
    
  }, [auth]);
  useEffect(() => {
    if (auth.address && DTO.cities) {
      let address = auth.address.split(', ');
      if (address[2]) {
        let cityIndex = DTO.cities.findIndex(el => el.province_name.indexOf(address[2]) !== -1);
        
        if (cityIndex !== -1) {
          getDistricts(DTO.cities[cityIndex].province_id, {city_id: cityIndex, city: DTO.cities[cityIndex].province_name, district_text: address[1], ward_text: address[0]});
        }
      }
    }
  }, [DTO.cities]);

  useEffect(() => {
    dispatch(GetAllLanguageAction());
    dispatch(GetAllTechnologyAction());
    dispatch(GetFreeTimeAction());
    dispatch(GetAllMajorAction());
  }, []);

  const getDistricts = async (id, params) => {
      const {data} = await axios.get(`https://vapi.vnappmob.com/api/province/district/${id}`);
      if (params.district_text && data.results) {
        let districtIndex = data.results.findIndex(el => el.district_name.indexOf(params.district_text) !== -1);
        if (districtIndex !== -1) {
          getWards(data.results[districtIndex].district_id, { ...params, 
            district_id: districtIndex,
            district: data.results[districtIndex].district_name,
            districts: data.results});
          return;
        } else {
          setDTO(pre => ({ ...pre, ...params, districts: data.results}));
          return;
        }
      }
      setDTO(pre => ({ ...pre, ...params, districts: data.results}));
  };

  const getWards = async (id, params) => {
      const {data} = await axios.get(`https://vapi.vnappmob.com/api/province/ward/${id}`);
      if (params.ward_text && data.results) {
        let wardIndex = data.results.findIndex(el => el.ward_name.indexOf(params.ward_text) !== -1);
        if (wardIndex !== -1) {
          setDTO(pre => ({ ...pre, ...params, wards: data.results, ward: data.results[wardIndex].ward_name, ward_id: wardIndex}));
          return;
        } else {
          setDTO(pre => ({ ...pre, ...params, wards: data.results}));
          return;
        }
      }

      setDTO(pre => ({ ...pre, ...params, wards: data.results }));
  }

  function onClickUpdate() {
    if(checkSecondStep() == false) {
      return;
    }
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
      learnerID: auth.learnerID,
      address: addressArray.length ? addressArray.join(', ') : null,
      address1: DTO.address1 ? DTO.address1 : null,
      learnerLevel: DTO.learnerLevel,
      language: language.map(({lanName}) => lanName).join(', '),
      jobNow: DTO.jobNow,
      technologySkill: techSkill.map(({techName}) => techName).join(', '),
      fieldOfStudy: DTO.fieldOfStudy,
      feeMax: DTO.feeMax,
      freeTime: DTO.freeTime && DTO.freeTime.join('|') || '',
      futureSelfDevelopment: DTO.futureSelfDevelopment,
    }
    dispatch(updateAction(postdata)).then(() => {
      toastSuccessText("Update success!");
    });
  }

  const checkSecondStep = () => {
    const errors = [];
    if(!(language && language.length > 0)) {
      errors.push("Please choose at least one language you know");
    }
    if (!(techSkill && techSkill.length > 0)) {
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

    errors.forEach((err) => {
      toastErrorText(err);
    });

    return errors.length == 0;
  }
  const addFreeTime = () => {
    let fullFreeTime = timeStart + '-' + timeEnd + ' (' + weekday.map((el) => el.key).join('-') + ')';
    setDTO({...DTO, freeTime: [...DTO.freeTime, fullFreeTime]});
    setTimeStart('00:00');
    setTimeEnd('23:59');
    setWeekday([]);
  }
  const onRemoveTime = (index) => {
    setDTO({...DTO, freeTime: DTO.freeTime.filter((el, i) => i != index)});
  }

  return (
    <>
      <ThemeOptions />
      <AppHeader />
      <br/><br/><br/><br/>
      <div className="app-main__inner card-body main-card card">
        <div className="form-wizard-content">
            <Form>
                <FormGroup row>
                    <Label for="exampleEmail" sm={4}>
                        What do you do?
                    </Label>
                    <Col sm={7}>
                        <Input type="select" value={DTO.jobNow} onChange={(e) => setDTO({ ...DTO, jobNow: e.target.value })}>
                            <option value={""}>Not employed</option>
                            <option value={"work"}>Employed full-time</option>
                            <option value={"study"}>Student full-time</option>
                        </Input>
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label sm={4}>
                        What is your major?
                    </Label>
                    <Col sm={7}>
                        <Input type="select" value={DTO.fieldOfStudy} onChange={(e) => setDTO({ ...DTO, fieldOfStudy: e.target.value })}>
                            <option value={""}>Please select a major</option>
                            {lsMajor.length && lsMajor.map(({subjectName, subjectID}, index) => (<option key={subjectID} value={subjectID}>{subjectName}</option>))}
                        </Input>
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="examplePassword" sm={4}>
                        Which of the following best describes the highest level of formal education that you've completed?
                    </Label>
                    <Col sm={7}>
                        <Input type="select" value={DTO.learnerLevel} onChange={(e) => setDTO({ ...DTO, learnerLevel: e.target.value })}>
                            <option value={""}>Please select a degree</option>
                            <option value={"Secondary school"}>Secondary school</option>
                            <option value={"Some college/university study without earning a degree Associate degree (A.A., A.S., etc.)"}>Some college/university study without earning a degree Associate degree (A.A., A.S., etc.)</option>
                            <option value={"Bachelor’s degree (B.A., B.S., B.Eng., etc.)"}>Bachelor’s degree (B.A., B.S., B.Eng., etc.)</option>
                            <option value={"Master’s degree (M.A., M.S., M.Eng., MBA, etc.)"}>Master’s degree (M.A., M.S., M.Eng., MBA, etc.)</option>
                            <option value={"Professional degree (JD, MD, etc.)"}>Professional degree (JD, MD, etc.))</option>
                            <option value={"Other doctoral degree (Ph.D., Ed.D., etc.)"}>Other doctoral degree (Ph.D., Ed.D., etc.)</option>
                            <option value={"Something else"}>Something else</option>
                        </Input>
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="exampleSelect" sm={4}>
                        Where do you live?
                    </Label>
                    <Col sm={7}>
                        <Input type="text" value={DTO.address1} onChange={(e) => setDTO({...DTO, address1: e.target.value})} placeholder="Example: 62 Trương Phước Phan" />
                    </Col>
                </FormGroup>
                <FormGroup row>
                  <Label sm={4}>
                            {/* Select Ward, District, City */}
                  </Label>
                    <Col sm={2}>
                        <Input type="select" value={DTO.ward_id} onChange={(e) => {
                            if (e.target.value === "-1") { 
                                setDTO({ ...DTO, ward_id: e.target.value, ward: "" })
                            } else {
                                setDTO({ ...DTO, ward_id: e.target.value, ward: DTO.wards[e.target.value].ward_name })
                            }
                        }}>
                            <option value={-1}>Please select Ward</option>
                            {DTO.wards.length && DTO.wards.map(({ward_id, ward_name}, index) => (<option key={ward_id} value={index}>{ward_name}</option>))}
                        </Input>
                    </Col>
                    <Col sm={2}>
                        <Input type="select" value={DTO.district_id} onChange={(e) => {
                            if (e.target.value === "-1") { 
                                getWards(0, { ...DTO, district_id: e.target.value, district: "", ward_id: -1, ward: "" });
                            } else {
                                getWards(DTO.districts[e.target.value].district_id, { ...DTO, district_id: e.target.value, district: DTO.districts[e.target.value].district_name, ward_id: -1, ward: "" });
                            }
                        }}>
                            <option value={-1}>Please select a district</option>
                            {DTO.districts.length && DTO.districts.map(({district_id, district_name}, index) => (<option key={district_id} value={index}>{district_name}</option>))}
                            
                        </Input>
                    </Col>
                    <Col sm={2}>
                        <Input type="select" value={DTO.city_id} onChange={(e) => {
                            if (e.target.value === "-1") { 
                                getDistricts(0, { ...DTO, city_id: e.target.value, city: "", district_id: -1, district: "", ward_id: -1, ward: "" });
                            } else {
                                getDistricts(DTO.cities[e.target.value].province_id, { ...DTO, city_id: e.target.value, city: DTO.cities[e.target.value].province_name, district_id: -1, district: "", ward_id: -1, ward: "" });
                                
                            }
                        }}>
                            <option value={-1} key={0}>Please select a city</option>
                            {DTO.cities.length && DTO.cities.map(({province_name, province_id}, index) => (<option key={province_id} value={index}>{province_name}</option>))}
                        </Input>
                    </Col>
                    
                </FormGroup>

                <FormGroup row>
                    {/* <Label for="exampleSelectMulti" sm={4}>
                        Which languages can you know?
                    </Label> */}
                    <Label for="exampleSelectMulti" sm={4}><span className="text-danger">*</span>Which languages can you know?</Label>

                    <Col sm={7}>
                        <Select isMulti components={makeAnimated()}
                            closeMenuOnSelect={false}
                            getOptionLabel={option => option.lanName}
                            getOptionValue={option => option.lanName}
                            options={lsLanguage} className="basic-multi-select" classNamePrefix="select"
                            value={language}
                            onChange={(value) => {
                                setlanguage(value);
                            }}
                        />
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="exampleSelectMulti" sm={4}><span className="text-danger">*</span>What skills do you know?</Label>

                    <Col sm={7}>
                        <Select isMulti components={makeAnimated()}
                            closeMenuOnSelect={false}
                            getOptionLabel={option => option.techName}
                            getOptionValue={option => option.techName}
                            options={lsTechnology} className="basic-multi-select" classNamePrefix="select"
                            value={techSkill}
                            onChange={(value) => {
                                settechSkill(value);
                            }} />
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="examplePassword" sm={4}>
                      What is your budget for courses?
                    </Label>
                    <Col sm={7}>
                        <Input type="select" value={DTO.feeMax} onChange={(e) => setDTO({ ...DTO, 
                                feeMax: e.target.value, feeMaxText: lstFeeMax[e.target.value] })}>
                            <option value={""}>Please select a fee</option>
                            {lstFeeMax.map((value, index) => (<option key={value} value={index}>{value}</option>))}
                        </Input>
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="exampleSelectMulti" sm={4}>
                      If we are offering offline classes, what time would be best for you?
                    </Label>
                    <Col sm={2}>
                        Start time
                        <Input
                          placeholder="time placeholder"
                          type="time"
                          value={timeStart}
                          onChange={(e) => { setTimeStart(e.target.value)}}
                        />
                    </Col>
                    <Col sm={2}>
                        End time
                        <Input
                          placeholder="time placeholder"
                          type="time"
                          value={timeEnd}
                          onChange={(e) => { setTimeEnd(e.target.value)}}
                        />
                    </Col>
                    <Col sm={3}>
                        Weekday
                        <Select isMulti components={makeAnimated()}
                            closeMenuOnSelect={false}
                            getOptionLabel={option => option.text}
                            getOptionValue={option => option.key}
                            options={weekdays} className="basic-multi-select" classNamePrefix="select"
                            value={weekday}
                            onChange={(value) => {
                                setWeekday(value)
                            }}
                        />
                    </Col>
                    <Col sm={4}>
                    </Col>
                    <Col sm={8}>
                        <div>&nbsp;&nbsp;&nbsp;&nbsp;</div>
                        <Button color="secondary" className="btn-shadow float-left btn-wide btn-pill" outline
                          onClick={addFreeTime}>
                          Add freetime
                        </Button>
                        <br/><br/>
                    </Col>
                    <Col sm={4}>
                    </Col>
                    <Col sm={7}>
                        <ListGroup>
                            {DTO.freeTime.map((el, index) => (
                                <ListGroupItem key={index} style={{borderTop: "1px solid rgba(0, 0, 0, 0.125)"}}>
                                    {el}
                                    <button type="button" className="close" aria-label="Close" onClick={() => onRemoveTime(index)}>
                                        <span aria-hidden="true">×</span>
                                    </button>
                                </ListGroupItem>)
                            )}
                        </ListGroup>
                    </Col>
                </FormGroup>
                <FormGroup row>
                    <Label for="examplePassword" sm={4}>
                      Which of the following best reflects how you plan to grow as a person?
                    </Label>
                    <Col sm={7}>
                        <Input type="select" value={DTO.futureSelfDevelopment} onChange={(e) => setDTO({ ...DTO, futureSelfDevelopment: e.target.value })}>
                            <option value={""}>Select your future self-development</option>
                            <option value={"Data or business analyst"}>Data or business analyst</option>
                            <option value={"Data scientist or machine learning specialist"}>Data scientist or machine learning specialist</option>
                            <option value={"Database administrator"}>Database administrator</option>
                            <option value={"Designer"}>Designer</option>
                            <option value={"Developer, back-end"}>Developer, back-end</option>
                            <option value={"Developer, desktop or enterprise applications"}>Developer, desktop or enterprise applications</option>
                            <option value={"Developer, embedded applications or devices"}>Developer, embedded applications or devices</option>
                            <option value={"Developer, front-end"}>Developer, front-end</option>
                            <option value={"Developer, full-stack"}>Developer, full-stack</option>
                            <option value={"Developer, game or graphics"}>Developer, game or graphics</option>
                            <option value={"Developer, mobile"}>Developer, mobile</option>
                            <option value={"Developer, QA or test"}>Developer, QA or test</option>
                            <option value={"DevOps specialist"}>DevOps specialist</option>
                            <option value={"Engineer, data"}>Engineer, data</option>
                            <option value={"Engineer, site reliability"}>Engineer, site reliability</option>
                            <option value={"Engineering manager"}>Engineering manager</option>
                            <option value={"Marketing or sales professional"}>Marketing or sales professional</option>
                            <option value={"Product manager"}>Product manager</option>
                            <option value={"Scientist"}>Scientist</option>
                            <option value={"Senior Executive (C-Suite, VP, etc.)"}>Senior Executive (C-Suite, VP, etc.)</option>
                            <option value={"System administrator"}>System administrator</option>
                        </Input>
                    </Col>
                </FormGroup>
            </Form>

            <div className="text-center">
                <Button color="success" size="lg" className="btn-shadow btn-wide" onClick={() => onClickUpdate()}>
                    Update
                </Button>
            </div>
            <br />
        </div>
      </div>
      {/* </div> */}
      {/* </div> */}
    </>
  );
}
