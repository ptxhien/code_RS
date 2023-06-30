import * as Types from "../../constants/actionType";

const initialState = {
  isLoading: false,
  isRecommended: false,
  shouldShowException: true,
  data: [],
  online: {},
  offline: {},
  skills_acquired: "",
  skills_to_learn: "",
  occupation: "",
  typeFilter:""

};

const coursesReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.COURSE_LOADING: {
      return {
        ...state,
        isLoading: true,
        isRecommended: false,
      };
    }
    case Types.RECOMMENDATION: {
      const { data } = action.payload; 
      return {
        ...state,
        isLoading: false,
        isRecommended: true,
        shouldShowException: false,
        online: data.courses_online,
        offline: data.courses_offline, 
        skills_acquired: data.skills_acquired,
        skills_to_learn: data.skills_to_learn, 
        typeFilter: data.typeFilter,
        occupation: data.occupation,
        data: [],
      };
    }
    case Types.GET_COURSES: {
      const { data } = action.payload;
      return {
        data,
        isLoading: false,
        shouldShowException: false,
      };
    }
    case Types.RECOMMENDATION_FAIL: {
      return initialState;
    }
    default:
      return state;
  }
};

export default coursesReducer;
