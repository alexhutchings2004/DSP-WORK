// Listen for the message to reopen the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "reopen_popup") {
      // Programmatically open the popup
      chrome.action.openPopup();
    }
  });