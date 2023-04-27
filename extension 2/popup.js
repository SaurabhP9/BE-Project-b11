chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
  var youtubeUrl = tabs[0].url;
  var videoId = getVideoId(youtubeUrl);

  // Send the URL to the Node server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "http://localhost:3000/check-id", true);
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.onreadystatechange = function () {
    // var prediction = "saurabh";

    if (xhr.readyState === 4 && xhr.status === 200) {
      // Response from the PHP server
      console.log(xhr.responseText);
      display(xhr.responseText);
      // prediction = xhr.responseText;
    }
  };
  xhr.send("id=" + videoId);

  // console.log(prediction);
});

function display(pred) {
  document.getElementById("videoId").innerHTML = "Prediction " + pred;
  chrome.runtime.sendMessage({
    action: "showPrediction",
    result: pred,
  });
}

function getVideoId(url) {
  var videoId = "";
  if (url.indexOf("youtube.com/watch?v=") != -1) {
    videoId = url.split("v=")[1];
  } else if (url.indexOf("youtu.be/") != -1) {
    videoId = url.split("youtu.be/")[1];
  }
  return videoId;
}
