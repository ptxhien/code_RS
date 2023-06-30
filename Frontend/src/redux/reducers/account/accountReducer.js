import { stringify } from "query-string";
import * as Types from "../../constants/actionType";

var initialState = {
  user: {},
  isLogged: false,
  error: "",
  authToken: "",
  isLoading: false,
};

const account = (state = initialState, action) => {
  switch (action.type) {
    case Types.LOG_OUT: {
      localStorage.removeItem("authToken");
      localStorage.removeItem("learnerID");
      localStorage.removeItem("email");
      localStorage.removeItem("fullname");
      return initialState;
    }
    case Types.LOGIN_FAILURE: {
      const { message } = action.payload;
      return {
        ...state,
        isLoading: false,
        user: {},
        error: message,
      };
    }
    case Types.LOGIN: {
      const { authToken, user } = action.payload;
      localStorage.setItem("authToken", authToken);
      localStorage.setItem("user", JSON.stringify(user));
      return {
        user,
        authToken,
        isLogged: true,
        isLoading: false,
        error: "",
      };
    }
    case Types.REGISTER: {
      const { authToken, user } = action.payload;
      localStorage.setItem("authToken", authToken);
      localStorage.setItem("user", JSON.stringify(user));
      return {
        user,
        authToken,
        isLogged: true,
        isLoading: false,
        error: "",
      };
    }
    case Types.AUTH_UPDATE: {
      const { user } = action.payload;
      const authToken = localStorage.getItem("authToken");
      localStorage.setItem("user", JSON.stringify(user));
      return {
        user,
        authToken,
        isLogged: true,
        isLoading: false,
        error: ""
      };
    }
    case Types.RETRIEVE: {
      const { user, authToken } = action.payload;
      return {
        ...state,
        user,
        authToken,
        isLogged: true,
      };
    }
    case Types.LOADING: {
      return { ...state, error: "", isLoading: true };
    }
    default:
      return state;
  }
};

export default account;
