window.HELP_IMPROVE_VIDEOJS = false;

var SLIDER_BASE = "./static/videos/n_actions_slider";
var N_ACTIONS = [2, 4, 7, 14, 28, 56, 112];
var NUM_VIDEOS = N_ACTIONS.length;

var interp_videos = [];
var videosLoaded = 0;
var allVideosStarted = false;

function preloadActionSliderVideos() {
  for (var i = 0; i < NUM_VIDEOS; i++) {
    var path =
      SLIDER_BASE + "/measurements_reconstruction_" + N_ACTIONS[i] + ".webm";
    interp_videos[i] = document.createElement("video");
    interp_videos[i].src = path;
    interp_videos[i].loop = true;
    interp_videos[i].muted = true;
    interp_videos[i].controls = false;
    interp_videos[i].preload = "auto";

    // Use absolute positioning but maintain proper video sizing
    interp_videos[i].style.position = "absolute";
    interp_videos[i].style.top = "0";
    interp_videos[i].style.left = "0";
    interp_videos[i].style.width = "100%";
    interp_videos[i].style.height = "auto";
    interp_videos[i].style.maxWidth = "100%";
    interp_videos[i].style.objectFit = "contain";
    interp_videos[i].style.zIndex = "1";
    interp_videos[i].style.opacity = "0";

    // Use loadeddata event which is more reliable
    interp_videos[i].addEventListener("loadeddata", function () {
      videosLoaded++;
      if (videosLoaded === NUM_VIDEOS && !allVideosStarted) {
        startAllVideosSimultaneously();
      }

      // Set container height based on first loaded video
      if (videosLoaded === 1) {
        var video = this;
        // Set height immediately if metadata is already loaded
        if (video.videoHeight && video.videoWidth) {
          var aspectRatio = video.videoHeight / video.videoWidth;
          var containerWidth = $("#interpolation-image-wrapper").width();
          var containerHeight = containerWidth * aspectRatio;
          $("#interpolation-image-wrapper").css(
            "height",
            containerHeight + "px",
          );
        } else {
          // Otherwise wait for metadata
          video.addEventListener("loadedmetadata", function () {
            var aspectRatio = video.videoHeight / video.videoWidth;
            var containerWidth = $("#interpolation-image-wrapper").width();
            var containerHeight = containerWidth * aspectRatio;
            $("#interpolation-image-wrapper").css(
              "height",
              containerHeight + "px",
            );
          });
        }
      }
    });

    interp_videos[i].addEventListener("error", function (e) {
      console.error("Video loading error for path:", this.src);
    });

    // Add all videos to the wrapper but keep them hidden
    $("#interpolation-image-wrapper").append(interp_videos[i]);
  }

  // Make the wrapper positioned relative and ensure it has proper height
  $("#interpolation-image-wrapper").css({
    position: "relative",
    width: "100%",
    "min-height": "300px", // Ensure minimum space for slider
    display: "block",
  });
}

function startAllVideosSimultaneously() {
  if (allVideosStarted) return; // Prevent multiple starts
  allVideosStarted = true;

  // Start all videos at the same time
  for (var i = 0; i < NUM_VIDEOS; i++) {
    interp_videos[i].play().catch(function (error) {
      console.error("Error playing video:", error);
    });
  }

  // Show the initial video
  setVideo(3);
}
var currentVideoIndex = -1;

function setVideo(i) {
  if (currentVideoIndex === i) {
    return; // Same video, don't change anything
  }

  var video = interp_videos[i];
  if (!video) {
    return;
  }

  video.ondragstart = function () {
    return false;
  };
  video.oncontextmenu = function () {
    return false;
  };

  // Use opacity for smooth transitions - show new video
  video.style.opacity = "1";
  video.style.zIndex = "2";

  // Hide the previously visible video
  if (currentVideoIndex >= 0 && interp_videos[currentVideoIndex]) {
    interp_videos[currentVideoIndex].style.opacity = "0";
    interp_videos[currentVideoIndex].style.zIndex = "1";
  }

  currentVideoIndex = i;
}

$(document).ready(function () {
  // Dark mode functionality
  const darkModeToggle = document.getElementById("darkModeToggle");
  const darkModeIcon = document.getElementById("darkModeIcon");
  const body = document.body;
  const html = document.documentElement;

  // Check for saved dark mode preference or default to light mode
  const isDarkMode = localStorage.getItem("darkMode") === "enabled";

  if (isDarkMode) {
    body.classList.add("dark-mode");
    html.classList.add("dark-mode");
    if (darkModeIcon) {
      darkModeIcon.classList.remove("fa-moon");
      darkModeIcon.classList.add("fa-sun");
    }
  }

  // Dark mode toggle event listener
  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", function () {
      body.classList.toggle("dark-mode");
      html.classList.toggle("dark-mode");

      if (body.classList.contains("dark-mode")) {
        darkModeIcon.classList.remove("fa-moon");
        darkModeIcon.classList.add("fa-sun");
        localStorage.setItem("darkMode", "enabled");
      } else {
        darkModeIcon.classList.remove("fa-sun");
        darkModeIcon.classList.add("fa-moon");
        localStorage.setItem("darkMode", "disabled");
      }
    });
  }

  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  };

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach(".carousel", options);

  // Loop on each carousel initialized
  for (var i = 0; i < carousels.length; i++) {
    // Add listener to  event
    carousels[i].on("before:show", (state) => {
      console.log(state);
    });
  }

  // Access to bulmaCarousel instance of an element
  var element = document.querySelector("#my-element");
  if (element && element.bulmaCarousel) {
    // bulmaCarousel instance is available as element.bulmaCarousel
    element.bulmaCarousel.on("before-show", function (state) {
      console.log(state);
    });
  }

  preloadActionSliderVideos();

  // Fallback: if videos don't start within 3 seconds, try to start them anyway
  setTimeout(function () {
    if (!allVideosStarted) {
      console.log("Fallback: forcing video start");
      startAllVideosSimultaneously();
    }
  }, 3000);

  $("#interpolation-slider").on("input", function (event) {
    setVideo(this.value);
  });
  $("#interpolation-slider").prop("max", NUM_VIDEOS - 1);

  bulmaSlider.attach();
});
