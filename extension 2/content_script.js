chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action == "getYouTubeUrl") {
    var url = window.location.href;
    sendResponse({ url: url });
  }
});
