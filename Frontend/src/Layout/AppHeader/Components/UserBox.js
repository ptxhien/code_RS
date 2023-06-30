import React, { Fragment } from "react";

// import Ionicon from 'react-ionicons';

import { IoIosCalendar } from "react-icons/io";

import PerfectScrollbar from "react-perfect-scrollbar";
import {connect} from 'react-redux';

import {
  DropdownToggle,
  DropdownMenu,
  Nav,
  Col,
  Row,
  Button,
  NavItem,
  NavLink,
  UncontrolledTooltip,
  UncontrolledButtonDropdown,
} from "reactstrap";

import { toast, Bounce } from "react-toastify";

import { faAngleDown } from "@fortawesome/free-solid-svg-icons";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import city3 from "../../../assets/utils/images/dropdown-header/city3.jpg";
import { history } from './../../../helpers/history';
import { Logout } from "../../../redux/actions/account/accountAction";

class UserBox extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      active: false,
    };
  }

  notify2 = () =>
  (this.toastId = toast(
    "You don't have any new items in your calendar for today! Go out and play!",
    {
      transition: Bounce,
      closeButton: true,
      autoClose: 5000,
      position: "bottom-center",
      type: "success",
    }
  ));

  onClickLogOut = () => {
    if (!this.props.accountReducer.isLogged) {
      history.push("/#/login");
      window.location.reload();
    }
    this.props.logout(); 
  }

  render() {
    const retrievedUser = this.props.accountReducer.user;
    const authToken = this.props.accountReducer.authToken;
    return (
      <Fragment>
        <div className="header-btn-lg pr-0">
          <div className="widget-content p-0">
            <div className="widget-content-wrapper">
              <div className="widget-content-left">
                <UncontrolledButtonDropdown>
                  <DropdownToggle color="link" className="p-0">
                    <img
                      width={42}
                      className="rounded-circle"
                      src={'https://png.pngtree.com/png-clipart/20190904/original/pngtree-hand-drawn-flat-wind-user-avatar-icon-png-image_4492039.jpg'}
                      alt=""
                    />
                    <FontAwesomeIcon
                      className="ml-2 opacity-8"
                      icon={faAngleDown}
                    />
                  </DropdownToggle>
                  <DropdownMenu right className="rm-pointers dropdown-menu-lg">
                    <div className="dropdown-menu-header">
                      <div className="dropdown-menu-header-inner bg-info">
                        <div
                          className="menu-header-image opacity-2"
                          style={{
                            backgroundImage: "url(" + city3 + ")",
                          }}
                        />
                        <div className="menu-header-content text-left">
                          <div className="widget-content p-0">
                            <div className="widget-content-wrapper">
                              <div className="widget-content-left mr-3">
                                <img
                                  width={45}
                                  className="rounded-circle"
                                  src={'https://png.pngtree.com/png-clipart/20190904/original/pngtree-hand-drawn-flat-wind-user-avatar-icon-png-image_4492039.jpg'}
                                  alt=""
                                />
                              </div>
                              <div className="widget-content-left">
                                <div className="widget-heading text-light">
                                <b>{retrievedUser.fullname}</b>
                                </div>
                                <div className="widget-subheading text-light">
                                  <b>{retrievedUser.email}</b>
                                </div>
                              </div>
                              <div className="widget-content-right mr-2">
                                <Button
                                  className="btn-pill btn-shadow btn-shine"
                                  color="focus"
                                  onClick={this.onClickLogOut}
                                >
                                  {authToken ? "Logout" : "Login"}
                                </Button>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <PerfectScrollbar>
                      <Nav vertical>
                        <NavItem className="nav-item-header">
                          Your account
                        </NavItem>
                        <NavItem>
                          {authToken ? <NavLink href="/#/update-profile" className="text-primary">Edit profile</NavLink> : ""}
                        </NavItem>
                      </Nav>
                    </PerfectScrollbar>
                  </DropdownMenu>
                </UncontrolledButtonDropdown>
              </div>
              <div className="widget-content-left  ml-3 header-user-info">
                {/* <div className="widget-heading">{activeUser.Username}</div>
                <div className="widget-subheading">{activeUser.FullName}</div> */}
              </div>
            </div>
          </div>
        </div>
      </Fragment>
    );
  }
}

const mapStateToProps = (state) => ({
  accountReducer: state.accountReducer,
});

const mapDispatchToProps = (dispatch) => ({
  logout: () => dispatch(Logout())
});

export default connect(mapStateToProps, mapDispatchToProps)(UserBox);
