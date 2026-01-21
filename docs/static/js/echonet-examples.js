var videos = [];
var videosLoaded = 0;
var allVideosStarted = false;
var currentVideoIndex = -1;

var sliderElement;
var SLIDER_BASE;
var N_ACTIONS;
var NUM_VIDEOS;
var currentPatient = "0X286856FF99946B18";
var availablePatients = [
  "0X286856FF99946B18",
  "0X331101FA7F150A51",
  "0X305357FDBA449A7E",
];

function getURLParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    patient: params.get("patient") || currentPatient,
    nActions: params.get("nActions") || "14",
  };
}

function updateURL(patient, nActions) {
  const url = new URL(window.location);
  url.searchParams.set("patient", patient);
  url.searchParams.set("nActions", nActions);
  window.history.pushState({}, "", url);
}

function preloadActionSliderVideos() {
  // Clear existing videos
  $("#action-image-wrapper").empty();
  videos = [];
  videosLoaded = 0;
  allVideosStarted = false;
  currentVideoIndex = -1;

  for (var i = 0; i < NUM_VIDEOS; i++) {
    var path =
      SLIDER_BASE + "/" + currentPatient + "/" + N_ACTIONS[i] + ".webm";
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
          $("#action-image-wrapper").css("height", containerHeight + "px");
        } else {
          // Otherwise wait for metadata
          video.addEventListener("loadedmetadata", function () {
            var aspectRatio = video.videoHeight / video.videoWidth;
            var containerWidth = $("#action-image-wrapper").width();
            var containerHeight = containerWidth * aspectRatio;
            $("#action-image-wrapper").css("height", containerHeight + "px");
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

  // Show the initial video based on URL params
  const urlParams = getURLParams();
  const nActionsIndex = N_ACTIONS.indexOf(parseInt(urlParams.nActions));
  const initialIndex = nActionsIndex >= 0 ? nActionsIndex : 1; // Default to 14 (index 1)
  setVideo(initialIndex);
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

  // Update URL with current selection
  updateURL(currentPatient, N_ACTIONS[i]);
}

function switchPatient(patient) {
  if (currentPatient === patient) {
    return;
  }

  currentPatient = patient;
  $("#current-patient").text(patient);

  // Update active button state
  $(".patient-button").removeClass("is-info is-active");
  $("#btn-" + patient).addClass("is-info is-active");

  // Reload videos for new patient
  preloadActionSliderVideos();

  // Update URL
  const currentSliderValue = $("#action-slider").val();
  updateURL(currentPatient, N_ACTIONS[currentSliderValue]);
}

function generatePatientButtons() {
  const container = $("#patient-buttons");
  container.empty();

  availablePatients.forEach(function (patient) {
    const isActive = patient === currentPatient;
    const button = $(
      '<button class="button is-medium patient-button" id="btn-' +
        patient +
        '">' +
        patient.toUpperCase() +
        "</button>",
    );

    if (isActive) {
      button.addClass("is-info is-active");
    } else {
      button.addClass("is-light");
    }

    button.on("click", function () {
      switchPatient(patient);
    });

    container.append(button);
  });
}

$(document).ready(function () {
  // Initialize variables after DOM is ready
  sliderElement = document.getElementById("action-image-wrapper");
  SLIDER_BASE = "./static/videos/" + sliderElement.dataset.videoFolder;
  N_ACTIONS = sliderElement.dataset.nActions;
  N_ACTIONS = N_ACTIONS.split(",").map(Number);
  NUM_VIDEOS = N_ACTIONS.length;

  // Get initial state from URL
  const urlParams = getURLParams();
  currentPatient = urlParams.patient;

  // Update UI with current patient
  $("#current-patient").text(currentPatient);

  // Generate patient navigation buttons
  generatePatientButtons();

  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Set initial slider value from URL
  const nActionsIndex = N_ACTIONS.indexOf(parseInt(urlParams.nActions));
  const initialSliderValue = nActionsIndex >= 0 ? nActionsIndex : 1;
  $("#action-slider").val(initialSliderValue);

  // Update slider label
  const values = N_ACTIONS;
  const label = $("#slider-value-label");
  label.text(values[initialSliderValue]);

  preloadActionSliderVideos();

  $("#action-slider").on("input", function (event) {
    const index = parseInt(this.value);
    setVideo(index);
    label.text(values[index]);
  });
  $("#action-slider").prop("max", NUM_VIDEOS - 1);

  bulmaSlider.attach();
});
