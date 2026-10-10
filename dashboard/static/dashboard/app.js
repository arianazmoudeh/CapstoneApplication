document.querySelectorAll('.loading-form').forEach(form => {
  const button = form.querySelector('button[type=submit]');
  form.addEventListener('submit', () => {
    button.disabled = true;
    button.textContent = 'Loading prices…';
  });
  window.addEventListener('pageshow', () => {
    button.disabled = false;
    button.textContent = 'Refresh prices';
  });
});
