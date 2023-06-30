import * as Types from "../../constants/actionType";

const initialState = {
  isLoading: false,
  data: []
};

const invoicesReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.INVOICE_LOADING: {
      return {
        ...state,
        isLoading: true,
      };
    }
    case Types.GET_INVOICE: {
      const { data } = action.payload;
      return {
        ...state,
        isLoading: false,
        data,
      };
    }
    case Types.INVOICE_FAILED: {
      return {
        ...state,
        isLoading: false,
      };
    }
    default:
      return state;
  }
};

export default invoicesReducer;
