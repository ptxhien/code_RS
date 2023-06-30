import React, { Fragment } from "react";
import CSSTransitionGroup from "react-transition-group/CSSTransitionGroup";
import {
  Button,
  ModalHeader,
  ModalBody,
  ModalFooter,
} from "reactstrap";
import Rodal from "rodal";

class ModalFaded extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      visible: false,
      animation: "zoom",
    };
  }

  show(animation) {
    this.setState({
      animation,
      visible: true,
    });
  }

  hide() {
    this.setState({ visible: false });
  }

  render() {
    let types = [
      "fade",
    ];
    let buttons = types.map((value, index) => {
      let style = {
        animationDelay: index * 100 + "ms",
        WebkitAnimationDelay: index * 100 + "ms",
        marginTop: "30px",
      };
      return (
        <Button key={index}  color="primary" onClick={this.props.submit} outline
        style={style}>
          {/* Recommend
           */}
          Search
        </Button>
      );
    });
    return (
      <Fragment>
        <CSSTransitionGroup component="div" transitionName="TabsAnimation" transitionAppear={true}
          transitionAppearTimeout={0} transitionEnter={false} transitionLeave={false}>
          {buttons}
        </CSSTransitionGroup>
      </Fragment>
    );
  }
}

export default ModalFaded;
