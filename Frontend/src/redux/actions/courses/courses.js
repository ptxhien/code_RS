import * as Types from "./../../constants/actionType";
import http from "../../utils/http";
import axios from "axios";


export const getCourses = () => async dispatch => {
  dispatch({
    type: Types.COURSE_LOADING,
  });
  try {
    const {data} = await http.get("/courses");
    dispatch({
      type: Types.GET_COURSES,
      payload: {
        data,
      },
    });
  } catch (err) {
    console.log(err.message)
    dispatch({
      type: Types.GET_COURSES_FAILED,
    });
  }
};

export const recommendCourses = (occupation, form, month, email, typeFilter) => async dispatch => {
  dispatch({
    type: Types.COURSE_LOADING,
  });
  try {
    console.log(`http://127.0.0.1:6868/recommendation?occupation=${occupation}&form=${form}&month=${month}&email=${email}&typeFilter=${typeFilter}`)
    const { data } = await axios.get(
      `http://127.0.0.1:6868/recommendation?occupation=${occupation}&form=${form}&month=${month}&email=${email}&typeFilter=${typeFilter}`
    );
    dispatch({
      type: Types.RECOMMENDATION,
      payload: {
        data,
      },
    });
  } catch (err) {
    
  }
}
