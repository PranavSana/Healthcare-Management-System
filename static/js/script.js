// ---------- THEME ----------
const toggleBtn = document.getElementById("themeToggle");

function setTheme(mode) {
  if (mode === "dark") {
    document.body.classList.add("dark");
    toggleBtn.textContent = "☀️";
  } else {
    document.body.classList.remove("dark");
    toggleBtn.textContent = "🌙";
  }
}

const savedTheme = localStorage.getItem("theme") || "light";
setTheme(savedTheme);

toggleBtn.addEventListener("click", () => {
  const isDark = document.body.classList.contains("dark");
  const newTheme = isDark ? "light" : "dark";
  localStorage.setItem("theme", newTheme);
  setTheme(newTheme);
});

// ---------- INPUT FOCUS ENHANCEMENTS ----------
document.querySelectorAll('.form-input').forEach(input => {
  input.addEventListener('focus', () => {
    input.parentElement.querySelector('.form-label')?.classList.add('active');
  });
  input.addEventListener('blur', () => {
    input.parentElement.querySelector('.form-label')?.classList.remove('active');
  });
});