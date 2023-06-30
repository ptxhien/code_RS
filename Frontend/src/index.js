import "./polyfills";
import React from "react";
import ReactDOM from "react-dom";

import * as serviceWorker from "./serviceWorker";

import { HashRouter } from "react-router-dom";
import "./assets/base.scss";
import Main from "./Pages/Main";
import configureStore from "./redux/config/configureStore";
import { Provider } from "react-redux";
import { ProvideCart } from "./hooks/useCart";

const store = configureStore();
const rootElement = document.getElementById("root");

const renderApp = (Component) => {
  ReactDOM.render(
    <Provider store={store}>
      <HashRouter>
        <ProvideCart>
          <Component />
        </ProvideCart>
      </HashRouter>
    </Provider>,
    rootElement
  );
};

renderApp(Main);

if (module.hot) {
  module.hot.accept("./Pages/Main", () => {
    const NextApp = require("./Pages/Main").default;
    renderApp(NextApp);
  });
}
serviceWorker.unregister();
