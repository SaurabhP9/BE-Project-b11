// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.action === "showPrediction") {
//     if (request.result == 1) {
//       alert("Video is Educational or Entertainment");
//     } else {
//       alert("not to play these video ");
//       chrome.tabs.create(sender.tab.id, { url: "https://www.youtube.com" });
//     }
//     // alert(request.result);
//   }
// });

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "showPrediction") {
    if (request.result == 1) {
      alert("Video is Educational 👨🏻‍🎓 🌡️");
    } else {
      alert("Video is Entertainment 🎬 😂");
    }
  }
});
