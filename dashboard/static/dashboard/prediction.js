const predictionForm = document.getElementById("prediction-form");

if (predictionForm) {
  const status = document.getElementById("prediction-status");
  const originalStatus = status.textContent;
  predictionForm.addEventListener("submit", () => {
    const button = document.getElementById("prediction-button");
    button.disabled = true;
    button.textContent = "Running…";
    status.textContent = "Loading daily prices and running the models. This may take a few seconds.";
  });
  window.addEventListener("pageshow", () => {
    const button = document.getElementById("prediction-button");
    button.disabled = false;
    button.textContent = "Run prediction";
    status.textContent = originalStatus;
  });
}
