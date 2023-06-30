import React, { Fragment } from "react";
import { Route } from "react-router-dom";

// DASHBOARDS

import AnalyticsDashboard from "./Analytics/";
import SalesDashboard from "./Sales/";
import CommerceDashboard from "./Commerce/";
import CRMDashboard from "./CRM/";
import MinimalDashboard1 from "./Minimal/Variation1";
import MinimalDashboard2 from "./Minimal/Variation2";

// Layout

import AppHeader from "../../Layout/AppHeader/";
import AppSidebar from "../../Layout/AppSidebar/";
import AppFooter from "../../Layout/AppFooter/";

// Theme Options
import ThemeOptions from "../../Layout/ThemeOptions/";
import { PrivateRoute } from "../../Layout/AppMain/PrivateRoute";

const Dashboards = ({ match }) => (
  <Fragment>
    <ThemeOptions />
    <AppHeader />
    <div className="app-main">
      <AppSidebar />
      <div className="app-main__outer">
        {match ? (
          <div className="app-main__inner">
            <PrivateRoute
              path={`${match.url}/AnalyticsDashboard`}
              component={AnalyticsDashboard}
            />
            <PrivateRoute
              path={`${match.url}/CRM`}
              component={CommerceDashboard}
            />
          </div>
        ) : (
          <div className="app-main__inner">
            <PrivateRoute path={`/`} component={AnalyticsDashboard} />
          </div>
        )}
      </div>
    </div>
  </Fragment>
);

export default Dashboards;
