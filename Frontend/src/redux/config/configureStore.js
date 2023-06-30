import { createStore, combineReducers, compose, applyMiddleware } from 'redux';
import reducers from '../reducers';
import thunk from 'redux-thunk';

export default function configureStore() {
  const composeEnhancer = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
  return createStore(
    combineReducers({
      ...reducers
    }),
    composeEnhancer(applyMiddleware(thunk))
  );
}