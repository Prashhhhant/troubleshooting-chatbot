// Get the DOM elements
const clearButton = document.getElementById("clear-button");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatbotMessages = document.querySelector(".chatbot-messages");
const loadingIndicator = document.getElementById("loading-indicator");

// Add event listeners
clearButton.addEventListener("click", clearChat);
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});
// Functions
function clearChat() {
  chatbotMessages.innerHTML = "";
  addBotMessage("🌟 Welcome to the world of possibilities with ComputerGini, your personal computer expert chatbot! 🤖");
  scrollToBottom();
}

function sendMessage() {
  const message = chatInput.value.trim();

  if (message !== "") {
    addMessage(message, "user");
    showLoadingIndicator();

    // Send the message to the server
    fetch("/", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `question=${encodeURIComponent(message)}`,
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoadingIndicator();
        const response = data.response;
        addBotMessage(response);
        scrollToBottom(); // Scroll to bottom after adding bot message
      })
      .catch((error) => {
        console.error("Error:", error);
        hideLoadingIndicator();
        addBotMessage("Sorry, something went wrong.");
        scrollToBottom(); // Scroll to bottom after adding error message
      });

    chatInput.value = "";
  }
}

function addBotMessage(message) {
  const chatbotMessage = document.createElement("div");
  chatbotMessage.classList.add("chatbot-message", "chatbot-response");
  chatbotMessage.innerHTML = `<p>${message}</p>`;
  chatbotMessages.appendChild(chatbotMessage);
  scrollToBottom(); // Scroll to bottom after adding bot message
}

function addMessage(message, sender) {
  const messageElement = document.createElement("div");
  messageElement.classList.add("chatbot-message", `chatbot-${sender}`);
  messageElement.innerHTML = `<p>${message}</p>`;
  chatbotMessages.appendChild(messageElement);
  setTimeout(() => {
    messageElement.classList.add("visible");
    scrollToBottom(); // Scroll to bottom after adding user message
  }, 100);
}

function scrollToBottom() {
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}


function showLoadingIndicator() {
  loadingIndicator.innerHTML = "Loading...";
}

function hideLoadingIndicator() {
  loadingIndicator.innerHTML = "";
}
