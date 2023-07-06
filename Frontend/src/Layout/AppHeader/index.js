import React, { Fragment } from "react";
import cx from "classnames";

import { connect } from "react-redux";

import CSSTransitionGroup from "react-transition-group/CSSTransitionGroup";

import HeaderLogo from "../AppLogo";

import SearchBox from "./Components/SearchBox";
import MegaMenu from "./Components/MegaMenu";
import UserBox from "./Components/UserBox";
import HeaderRightDrawer from "./Components/HeaderRightDrawer";

import HeaderDots from "./Components/HeaderDots";
import { Retrieve } from "../../redux/actions/account/accountAction";
import Select from "react-select";
import { useCart } from "../../hooks/useCart";
import { useHistory, useParams, useLocation } from "react-router";

class Header extends React.Component {
  componentWillMount() {
    this.props.retrieve();
  }

  render() {
    let { headerBackgroundColor, enableMobileMenuSmall, enableHeaderShadow } =
      this.props;
    return (
      <Fragment>
        <CSSTransitionGroup
          component="div"
          className={cx("app-header", headerBackgroundColor, {
            "header-shadow": enableHeaderShadow,
          })}
          transitionName="HeaderAnimation"
          transitionAppear={true}
          transitionAppearTimeout={1500}
          transitionEnter={false}
          transitionLeave={false}
        >
          <HeaderLogo />
          <div
            className={cx("app-header__content", {
              "header-mobile-open": enableMobileMenuSmall,
            })}
          >
            <div
              style={{
                width: "100%",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ width: 200 }}>
                {/* <Select options={[{ label: "Category", value: "" }]} /> */}
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  alignItems: "center",
                }}
              >
                <MenuItem
                  onClick={() => {
                    window.open(
                     "https://docs.google.com/forms/d/e/1FAIpQLSe5ZE6Nw7o4IB2jT2gFgSNzLlknfddTShnEshSYLRgqK9wJYw/viewform",
                      
                      "_blank" // <- This is what makes it open in a new window.
                    );
                  }}
                  src="/images/three-star.svg"
                  title="Take survey"
                />

                <MenuItem
                  onClick={() => {window.location = '/#/dashboard'}}
                  src="/images/monitor.svg"
                  title="My Course"
                />
                <MenuItemCart />
              </div>
            </div>
            {/* <div className="app-header-left">
              <SearchBox />
              <MegaMenu />
            </div> */}
            <div className="app-header-right" style={{ width: 200 }}>
              {/* <HeaderDots /> */}
              <UserBox />
            </div>
          </div>
        </CSSTransitionGroup>
      </Fragment>
    );
  }
}

const mapStateToProps = (state) => ({
  enableHeaderShadow: state.ThemeOptions.enableHeaderShadow,
  closedSmallerSidebar: state.ThemeOptions.closedSmallerSidebar,
  headerBackgroundColor: state.ThemeOptions.headerBackgroundColor,
  enableMobileMenuSmall: state.ThemeOptions.enableMobileMenuSmall,
});

const mapDispatchToProps = (dispatch) => ({
  retrieve: () => dispatch(Retrieve()),
});

export default connect(mapStateToProps, mapDispatchToProps)(Header);

function MenuItem(props) {
  return (
    <div
      onClick={props.onClick}
      style={{ cursor: "pointer", marginLeft: 10, marginRight: 10 }}
    >
      <img width={props.width} src={props.src} />
      <span style={{ color: "white" }}>{props.title}</span>
    </div>
  );
}

function MenuItemCart() {
  const { cart } = useCart();
  const history = useHistory();
  return (
    <div
      style={{
        cursor: "pointer",
        marginLeft: 10,
        marginRight: 10,
        position: "relative",
      }}
      onClick={() => history.push("/cart")}
    >
      <img width={37} src="/images/cart.svg" />
      <div
        style={{
          position: "absolute",
          padding: 3,
          backgroundColor: "red",
          color: "white",
          borderRadius: "50%",
          width: 25,
          height: 25,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          top: -5,
          right: -20,
        }}
      >
        <span>{cart.length}</span>
      </div>
    </div>
  );
}
