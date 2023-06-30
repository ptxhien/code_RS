
(function($) {
    "use strict";

    /*================================
    Preloader
    ==================================*/
    var preloader = $('#preloader');
    $(window).on('load', function() {
        preloader.fadeOut('slow', function() { $(this).remove(); });
    });

    /*================================
    select2
    ==================================*/
    if ($.fn.select2) {
      $(".course-select select").select2();
    }

    /*================================
    stickey Header
    ==================================*/
    $(window).on('scroll', function() {
        var scroll = $(window).scrollTop(),
            hbHeight = $('.header-top').innerHeight(),
            headerBottom = $('.header-bottom');

        if (scroll >= hbHeight ) {
            headerBottom.addClass("sticky-header");
        } else {
            headerBottom.removeClass("sticky-header");
        }
    });


    /*================================
    slicknav
    ==================================*/
    $('#nav_mobile_menu').slicknav({
        prependTo: "#mobile_menu"
    });

    
    /*================================
    countdown
    ==================================*/
    $('[data-countdown]').each(function() {
        var $this = $(this),
            finalDate = $(this).data('countdown');
        $this.countdown(finalDate, function(event) {
            $this.html(event.strftime('<span class="cdown days"><span class="time-count">%-D</span> <p>Days</p></span> <span class="cdown hour"><span class="time-count">%-H</span> <p>Hours</p></span> <span class="cdown minutes"><span class="time-count">%M</span> <p>Minutes</p></span> <span class="cdown second"> <span><span class="time-count">%S</span> <p>Seconds</p></span>'));
        });
    });

    /*================================
    Student Testimonial
    ==================================*/
    function singleGalleryCarousel() {
        if ($('.student-cmnt-active').length && $('.student-thumbnil-active').length) {

            var $sync1 = $(".student-cmnt-active"),
                $sync2 = $(".student-thumbnil-active"),
                flag = false,
                duration = 500;

            $sync1
                .owlCarousel({
                    items: 1,
                    margin: 0,
                    nav: false,
                    dots: true,
                    loop: true
                })
                .on('changed.owl.carousel', function(e) {
                    if (!flag) {
                        flag = true;
                        $sync2.trigger('to.owl.carousel', [e.item.index, duration, true]);
                        flag = false;
                    }
                });

            $sync2
                .owlCarousel({
                    margin: 20,
                    items: 3,
                    nav: false,
                    dots: false,
                    center: true,
                    loop: true,
                    stagePadding: 70,
                    navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
                    responsive: {
                        0: {
                            items: 2
                        },
                        400: {
                            items: 2
                        },
                        500: {
                            items: 3
                        },
                        800: {
                            items: 5
                        },
                        1400: {
                            items: 3
                        }
                    },
                })
                .on('click', '.owl-item', function() {
                    $sync1.trigger('to.owl.carousel', [$(this).index(), duration, true]);

                })
                .on('changed.owl.carousel', function(e) {
                    if (!flag) {
                        flag = true;
                        $sync1.trigger('to.owl.carousel', [e.item.index, duration, true]);
                        flag = false;
                    }
                });
        };
    }
    singleGalleryCarousel();

   /*--------------------------------------------------------------
         Ajax Contact Form
    --------------------------------------------------------------*/
    function removeVietnameseTones(str) {
      const newString = str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
      return newString;
    }
    $('.screen-reader-response').hide();
    //edit here
    $("#login-form").on("submit", async function (e) {
      e.preventDefault();
      // for api
      const api_url = "/auth/login";
      const headers = {
        "Content-type": "application/json; charset=UTF-8",
      };
      const body = {
        email: $(this).find("#c_email").val(),
        password: $(this).find("#password").val(),
      };

      // animation
      const submitBtn = $(this).find("#c_submit");
      const resBox = $(this).parent().find(".screen-reader-response");

      // for submiting data
      try {
        submitBtn.html(`<span>Loading ...</span>`);
        submitBtn.attr("disabled", "disabled");
        const {data} = await axios.post(api_url, body);
        localStorage.setItem("authToken", data["authToken"]);
        location.reload();
      }
      catch (err) {
        submitBtn.html(`Login`);
        submitBtn.removeAttr("disabled");
        resBox.css("display", "block");
        resBox.append(`
          <div class="alert alert-danger alert-dismissible fade show" role="alert">
            ${err.response.data.message}
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
        `);
      }
      
    });
    $("#logout-btn").on("click", async function(e){
      e.preventDefault();
      const api_url = "/auth/logout";
      try {
        const { data } = await axios.post(api_url);
        localStorage.removeItem("authToken");
        location.reload();
      } catch (err) {
        location.reload();
      }
    })
    
    $("#recommendation").on('submit', async function(e){
      e.preventDefault();

      if ($("#not-logged").length){
        $("#loginModal").modal("show");
        return;
      }

      const errs = [];
      const jobTitle = $("#job-title").val() || undefined;
      const learningMethod = $("#learning-method").val();
      const durationFrom = $("#duration-from").val();
      const durationTo = $("#duration-to").val();


      if (!jobTitle){
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Job title is an required field 
            <a type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </a>
          </div>`);
      }

      if (errs.length) {
        $(".err-display").html(errs.reduce((prev, curr) => prev + "\n" + curr));
        window.scrollTo(0, 0);
        return;
      }

      $(this).off();
      $(this).submit();

    });

    $("#register-btn").on("click", function (e) {
      e.preventDefault();
      sessionStorage.setItem("curr_url", location.href);
      $(this).off();
      $(this)[0].click();
      
    });

    $("#cities").on("change", async function (){
      const val = $(this).val();
      const res = await fetch("https://vapi.vnappmob.com/api/province/district/" + val);
      const data = await res.json();
      const defaultDistrict = `<option value='0'>Select district</option>`;
      const districtsHTML = defaultDistrict + data["results"].map(district => {
        return `
          <option value='${district["district_id"]}'>${district["district_name"]}</option>
        `;
      });
      $("#districts").html(districtsHTML);
      $("#wards").html(`<option value='0'>Select ward</option`);
    });
    $("#districts").on("change", async function (){
      const val = $(this).val();
      const res = await fetch("https://vapi.vnappmob.com/api/province/ward/" + val);
      const data = await res.json();
      const defaultWard = `<option value='0'>Select ward</option>`;
      const wardsHTML = defaultWard + data["results"].map(ward => {
        return `
          <option value='${ward["ward_id"]}'>${ward["ward_name"]}</option>
        `;
      });
      $("#wards").html(wardsHTML);
    });
    $("#wizard").on("submit", async function (e){
      e.preventDefault();

      const submitBtn = $("#sign-up-btn");

      const api_url = "/auth/register";
      const headers = {
        "Content-type": "application/json; charset=UTF-8",
      };


			const email = $("#r-email").val() || "";
			const fullname = $("#r-fullname").val() || "";
			const password = $("#r-password").val() || "";
			const gender = $('[name="gender"]:checked').val();
			const jobNow = $('[name="rdbJob"]:checked').length ? $('[name="rdbJob"]:checked').val() : "";
			const learnerLevel = $("#r-learnerLevel").val();
      const feeMax = $('#fee-max').val();
      const freeTime = $('#freetime').val()?.join(", ") || "";
      const technologySkill = $("#skills").val()?.join(", ") || "";
      const language = $("#languages").val()?.join(", ") || "";

      // location processing
      const cityOption = $("#cities").find("option:selected");
      const districtOption = $("#districts").find("option:selected");
      const wardOption = $("#wards").find("option:selected");
      let address;
      if (
        cityOption.attr("value") === "0" ||
        districtOption.attr("value") === "0" ||
        wardOption.attr("value") === "0"
      ) {
        address = "";
      } else {
        address =
          removeVietnameseTones(cityOption.text()) + "," +
          removeVietnameseTones(districtOption.text()) + "," +
          removeVietnameseTones(wardOption.text()); 
      }

      const errs = []
      if (!email) {
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Email is an required field 
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
      }

      if (!fullname) {
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Full name is an required field 
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
      }

      if (!password) {
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Password is an required field 
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
      }

      if (!technologySkill){
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Technology is an required field 
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
      }

      if (!language) {
        errs.push(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Language is an required field 
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
      }

      if (errs.length) {
        $(".err-display").html(errs.reduce((prev, curr) => prev + "\n" + curr));
        window.scrollTo(0, 0);
        return;
      }

      const body = {
        email,
        fullname,
        password,
        gender,
        jobNow,
        learnerLevel,
        address,
        feeMax,
        freeTime,
        language,
        technologySkill,
        phone: "",
      };

      try {
        submitBtn.html(`Loading ...`);
        submitBtn.attr("disabled", "disabled");
        const { data } = await axios.post(api_url, body);
        localStorage.setItem("authToken", data["authToken"]);
        location.href = "/";
      } catch (error) {
        console.log(error.response.data);
        $(".err-display").html(
          `<div class="alert alert-warning alert-dismissible fade show" role="alert">
            Opps! an error occurred, try again later :(
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>`
        );
        submitBtn.html(`SUBMIT <i class="fa fa-arrow-right"></i>`);
        submitBtn.removeAttr("disabled");
      }
    });

    const maxChange = function () {
      const max = parseInt($(this).val());
      const min = parseInt($("#duration-from").val());
      let optionString = `
        <option value="0" ${min === 0 ? "selected" : ""}>Min duration</option>
      `;
      for (let i = 1; i < max; i++) {
        optionString += `
          <option value="${i}" ${min === i ? "selected" : ""} >${i}</option>
        `;
      }
      $("#duration-from").off("change", minChange);
      $("#duration-from").html(optionString);
      $("#duration-from").trigger("change");
      $("#duration-from").on("change", minChange);
    };

    const minChange = function () {
      const min = parseInt($(this).val());
      const max = parseInt($("#duration-to").val());
      let optionString = `
        <option value="7" ${max === 7 ? "selected" : ""}>Max duration</option>
      `;
      for (let i = min + 1; i < 7; i++) {
        optionString += `
          <option value="${i}" ${max === i ? "selected" : ""}>${i}</option>
        `;
      }
      $("#duration-to").off("change", maxChange);
      $("#duration-to").html(optionString);
      $("#duration-to").trigger("change");
      $("#duration-to").on("change", maxChange);
    };

    $("#duration-from").on("change", minChange);

    $("#duration-to").on("change", maxChange);
  
})(jQuery);