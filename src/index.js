let model;
let videoWidth, videoHeight;
let ctx, canvas;
const log = document.querySelector("#array");
const VIDEO_WIDTH = 720;
const VIDEO_HEIGHT = 405;
const k = 3;
const machine = new kNear(k);

// Start the application
async function main() {
  model = await handpose.load();
  const video = await setupCamera();
  video.play();
  startLandmarkDetection(video);
}

// Start the webcam
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Webcam not available");
  }

  const video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: VIDEO_WIDTH,
      height: VIDEO_HEIGHT,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

// Predict the finger positions in the video stream
async function startLandmarkDetection(video) {
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById("output");

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  ctx = canvas.getContext("2d");

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1); // Flip the video because the webcam is mirrored

  predictLandmarks();
}

// Predict the location of the fingers with the model
async function predictLandmarks() {
  ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
  // prediction!
  const predictions = await model.estimateHands(video); // ,true for flip
  if (predictions.length > 0) {
    drawHand(ctx, predictions[0].landmarks, predictions[0].annotations);
    // updateKNN(predictions[0].landmarks);
  }
  requestAnimationFrame(predictLandmarks);
}

// Update the KNN model with the current hand pose
function updateKNN(landmarks, label) {
  const pose = landmarks.flat();
  machine.learn(pose, label);
  console.log(label);
}
// Draw hand and fingers with the x,y coordinates. We don't draw the z value.
function drawHand(ctx, keypoints, annotations) {
  // Show all x,y,z points of the whole hand in the log window
  log.innerText = keypoints.flat();

  // Points on all knuckles can be directly retrieved from keypoints
  for (let i = 0; i < keypoints.length; i++) {
    const y = keypoints[i][0];
    const x = keypoints[i][1];
    drawPoint(ctx, x - 2, y - 2, 3);
  }

  // Palm base as the last point added to each finger
  let palmBase = annotations.palmBase[0];
  for (let key in annotations) {
    const finger = annotations[key];
    finger.unshift(palmBase);
    drawPath(ctx, finger, false);
  }
}

// Draw a point
function drawPoint(ctx, y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

// Draw a line
function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

// Capture the current pose for training
async function capturePose(label) {
  // Get current hand pose landmarks
  const predictions = await model.estimateHands(video);
  if (predictions.length > 0) {
    const landmarks = predictions[0].landmarks;
    // Flatten the landmarks array to pass to KNN
    const flatLandmarks = landmarks.flatMap((point) => [point[0], point[1], point[2]]);
    console.log(flatLandmarks);
    // Learn the pose with the associated label
    updateKNN(flatLandmarks, label);
    console.log(`Captured pose for label: ${label}`);
  }
}

// Predict the current pose
async function predictPose() {
  if (machine.training.length > 0) {
    const predictions = await model.estimateHands(video);
    if (predictions.length > 0) {
      const landmarks = predictions[0].landmarks;
      // Flatten the landmarks array to pass to KNN
      const flatLandmarks = landmarks.flatMap((point) => [point[0], point[1], point[2]]);
      const prediction = machine.classify(flatLandmarks);
      handlePrediction(prediction);
    } else {
      console.log("No hand detected");
    }
  } else {
    console.log("No training examples available");
  }
}

// Handle the prediction result
function handlePrediction(label) {
  switch (label) {
    case "move":
      console.log("Move detected");
      // Add code for moving the mouse
      break;
    case "click":
      console.log("Click detected");
      // Add code for clicking the mouse
      break;
    case "doubleClick":
      console.log("Double Click detected");
      // Add code for double-clicking the mouse
      break;
    default:
      console.log(`Unknown gesture: ${label}`);
  }
}

// Event listeners for buttons
document.getElementById("moveButton").addEventListener("click", () => capturePose("move"));
document.getElementById("clickButton").addEventListener("click", () => capturePose("click"));
document.getElementById("doubleClickButton").addEventListener("click", () => capturePose("doubleClick"));
document.getElementById("predictButton").addEventListener("click", predictPose);

// Start the application
main();
