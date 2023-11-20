let model;

async function loadModel() {
  model = await tf.loadLayersModel(
    "https://teachablemachine.withgoogle.com/models/QPKFycXgR/model.json"
  );
}

loadModel();

function preprocessImage(image) {
  // Resize the image to 224x224 pixels
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  // Normalize the pixel values to be between -1 and 1
  const normalizedImage = resizedImage.div(255).sub(0.5).mul(2);

  // Add a batch dimension to the image (the model expects a batch of images
  //The purpose of adding a batch dimension to the preprocessed image using the expandDims() method is to ensure that the input data is in the correct format expected by the model. Even if the batch size is 1, the model expects the input data to be in the form of batches.
  //By adding a batch dimension at index 0, the resulting shape of the preprocessed image will be (1, 224, 224, 3). This means that there is one image in the batch, with a height and width of 224 pixels, and 3 color channels (for red, green, and blue). This shape is what the model expects as input, so adding the batch dimension ensures that the preprocessed image is in the correct format for the model to make predictions.
  const batchedImage = normalizedImage.expandDims(0);
  return batchedImage;
}

async function handleImageUpload(event) {
  // Read the uploaded image file
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.readAsDataURL(file); //= reader.result
  reader.onload = async () => {
    //display uploaded photo in the box
    let photo = document.getElementById("photo");
    let inputfile = document.getElementById("inputfile");
    const selectedFile = inputfile.files[0];
    photo.src = URL.createObjectURL(selectedFile);
    //

    // Create a tensor from the image data
    const image = new Image();
    image.src = reader.result;

    //is an asynchronous method that decodes the binary data of an image file into an image bitmap. This is required because the tf.browser.fromPixels() method expects an ImageBitmap
    await image.decode();
    //The tensor represents the pixel data of the image in a multi-dimensional array format
    //that can be easily processed by the machine learning model.
    const tensor = tf.browser.fromPixels(image);
    // Preprocess the tensor
    const preprocessed = preprocessImage(tensor);
    // Feed the preprocessed tensor to the model and get the prediction
    const prediction = await model.predict(preprocessed).data();
    const classes = ["cat", "dog", "eagle", "fish", "crocodile", "turtle"];

    const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
    const predictedClass = classes[predictedClassIndex];
    const confidenceScore = prediction[predictedClassIndex].toFixed(2);
    // console.log(confidenceScore);
    // Display the prediction
    const resultContainer = document.getElementById("result-container");

    resultContainer.innerHTML = `<span class="classname">${predictedClass}</span>
    <input type="range" name="" id="range" value="${confidenceScore * 100}"> 
     `;
    //  ${confidenceScore * 100}%
    let rangevalue = document.getElementById("range");
    let percentage = document.createElement("div");
    percentage.innerHTML = `${rangevalue.value}%`;
    resultContainer.appendChild(percentage);
    rangevalue.style.width = "90%";
    rangevalue.addEventListener("change", function () {
      percentage.innerHTML = `${rangevalue.value}%`;
    });

    let switchprdiction = predictedClass;
    let dogoverlay = document.getElementById("dogspan");
    let catoverlay = document.getElementById("catspan");
    let eagleoverlay = document.getElementById("eaglespan");
    let fishoverlay = document.getElementById("fishspan");
    let crocodileoverlay = document.getElementById("crocodilespan");
    let turtleoverlay = document.getElementById("turtlespan");

    switch (switchprdiction) {
      case "dog":
        dogoverlay.style.display = "block";
        dogoverlay.style.animation = "flash";
        dogoverlay.style.animationDuration = "2.5s";
        setTimeout(() => {
          dogoverlay.style.display = "none";
        }, 4000);
        break;
      case "cat":
        catoverlay.style.display = "block";
        catoverlay.style.animation = "flash";
        catoverlay.style.animationDuration = "2.5s";
        setTimeout(() => {
          catoverlay.style.display = "none";
        }, 4000);
        break;
      case "eagle":
        eagleoverlay.style.display = "block";
        eagleoverlay.style.animation = "flash";
        eagleoverlay.style.animationDuration = "2.5s";
        setTimeout(() => {
          eagleoverlay.style.display = "none";
        }, 4000);
        break;
      case "fish":
        fishoverlay.style.display = "block";
        fishoverlay.style.animation = "flash";
        fishoverlay.style.animationDuration = "2.5s";
        setTimeout(() => {
          fishoverlay.style.display = "none";
        }, 4000);
        break;
      case "crocodile":
        crocodileoverlay.style.display = "block";
        crocodileoverlay.style.animation = "flash";
        crocodileoverlay.style.animationDuration = "2s";
        setTimeout(() => {
          crocodileoverlay.style.display = "none";
        }, 4000);
        break;
      case "turtle":
        turtleoverlay.style.display = "block";
        turtleoverlay.style.animation = "flash";
        turtleoverlay.style.animationDuration = "5s";
        setTimeout(() => {
          turtleoverlay.style.display = "none";
        }, 4000);
        break;

      default:
        break;
    }
  };
}

const inputElement = document.getElementById("inputfile");
inputElement.addEventListener("change", handleImageUpload);
