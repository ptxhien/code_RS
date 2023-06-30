import { stringify } from "query-string";
import * as Types from "../../constants/actionType";

var initialState = {
  isLoading: false,
  lsLanguage: [],
  lsTechnology: [],
  lsFreeTime: [],
  lsJob: [],
  lsMajor: [],
};

const masterdataReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.LOADING:{
      return {
        ...state,
        isLoading: true,
      }
    }
    case Types.GET_LANGUAGE:
      return {
        ...state,
        isLoading: false,
        lsLanguage: action.data,
      };
    case Types.GET_TECHNOLOGY:
      return {
        ...state,
        isLoading: false,
        lsTechnology: action.data,
      };
    case Types.GET_FREE_TIME:
      return {
        ...state,
        isLoading: false,
        lsFreeTime: action.data,
      };
    case Types.GET_JOB:
      return {
        ...state,
        isLoading: false,
        lsJob: action.data,
      };
    case Types.GET_MAJOR:
      return {
        ...state,
        isLoading: false,
        lsMajor: action.data,
      };
    default:
      return state;
  }
};

export default masterdataReducer;
