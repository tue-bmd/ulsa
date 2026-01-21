var videos = [];
var videosLoaded = 0;
var allVideosStarted = false;
var currentVideoIndex = -1;

var sliderElement;
var SLIDER_BASE;
var N_ACTIONS;
var NUM_VIDEOS;

function preloadActionSliderVideos() {
  for (var i = 0; i < NUM_VIDEOS; i++) {
    var path =
      SLIDER_BASE + "/measurements_reconstruction_" + N_ACTIONS[i] + ".webm";
    videos[i] = document.createElement("video");
    videos[i].src = path;
    videos[i].loop = true;
    videos[i].muted = true;
    videos[i].controls = false;
    videos[i].preload = "auto";

    // Use absolute positioning but maintain proper video sizing
    videos[i].style.position = "absolute";
    videos[i].style.top = "0";
    videos[i].style.left = "0";
    videos[i].style.width = "100%";
    videos[i].style.height = "auto";
    videos[i].style.maxWidth = "100%";
    videos[i].style.objectFit = "contain";
    videos[i].style.zIndex = "1";
    videos[i].style.opacity = "0";

    // Use loadeddata event which is more reliable
    videos[i].addEventListener("loadeddata", function () {
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
          var containerWidth = $("#action-image-wrapper").width();
          var containerHeight = containerWidth * aspectRatio;
          $("#action-image-wrapper").css(
            "height",
            containerHeight + "px",
          );
        } else {
          // Otherwise wait for metadata
          video.addEventListener("loadedmetadata", function () {
            var aspectRatio = video.videoHeight / video.videoWidth;
            var containerWidth = $("#action-image-wrapper").width();
            var containerHeight = containerWidth * aspectRatio;
            $("#action-image-wrapper").css(
              "height",
              containerHeight + "px",
            );
          });
        }
      }
    });

    videos[i].addEventListener("error", function (e) {
      console.error("Video loading error for path:", this.src);
    });

    // Add all videos to the wrapper but keep them hidden
    $("#action-image-wrapper").append(videos[i]);
  }
}

function startAllVideosSimultaneously() {
  if (allVideosStarted) return; // Prevent multiple starts
  allVideosStarted = true;

  // Start all videos at the same time
  for (var i = 0; i < NUM_VIDEOS; i++) {
    videos[i].play().catch(function (error) {
      console.error("Error playing video:", error);
    });
  }

  // Show the initial video
  setVideo(3);
}


function setVideo(i) {
  if (currentVideoIndex === i) {
    return; // Same video, don't change anything
  }

  var video = videos[i];
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
  if (currentVideoIndex >= 0 && videos[currentVideoIndex]) {
    videos[currentVideoIndex].style.opacity = "0";
    videos[currentVideoIndex].style.zIndex = "1";
  }

  currentVideoIndex = i;
}

$(document).ready(function () {
  // Initialize variables after DOM is ready
  sliderElement = document.getElementById("action-image-wrapper");
  SLIDER_BASE = "./static/videos/" + sliderElement.dataset.videoFolder;
  N_ACTIONS = sliderElement.dataset.nActions;
  N_ACTIONS = N_ACTIONS.split(',').map(Number);
  NUM_VIDEOS = N_ACTIONS.length;

  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  preloadActionSliderVideos();

  $("#action-slider").on("input", function (event) {
    setVideo(this.value);
  });
  $("#action-slider").prop("max", NUM_VIDEOS - 1);

  bulmaSlider.attach();
});