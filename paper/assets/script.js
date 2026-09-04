// Research Paper Interactive Logic & Dynamic Metric Binding
document.addEventListener("DOMContentLoaded", () => {
  // 1. Theme toggle
  const themeToggleBtn = document.getElementById("themeToggle");
  const currentTheme = localStorage.getItem("theme") || "light";

  if (currentTheme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const isDark = document.documentElement.getAttribute("data-theme") === "dark";
      const newTheme = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("theme", newTheme);
    });
  }

  // 2. Fetch live execution manifest if available
  fetch("../outputs/run_manifest.json")
    .then((res) => {
      if (res.ok) return res.json();
      throw new Error("No manifest found");
    })
    .then((manifest) => {
      if (manifest.execution_status === "COMPLETED_REAL_DATA_RUN") {
        updateDOMWithLiveResults(manifest);
      }
    })
    .catch(() => {
      // Manifest not generated yet - remains in authoritative [PENDING REAL DATA RUN] mode
    });
});

function updateDOMWithLiveResults(manifest) {
  const primary = manifest.primary_temporal_evaluation || {};
  const k50 = (primary.metrics_by_k && primary.metrics_by_k.k_50) || {};

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el && val !== undefined) {
      el.textContent = typeof val === "number" ? val.toFixed(4) : val;
      el.classList.remove("pending-pill");
    }
  };

  setEl("val-sample-size", manifest.sample_size);
  setEl("val-eligible-pop", manifest.eligible_population);
  setEl("val-positive-rate", manifest.positive_rate ? (manifest.positive_rate * 100).toFixed(2) + "%" : undefined);

  setEl("val-primary-base-p50", k50.baseline_precision);
  setEl("val-primary-model-p50", k50.model_precision);
  setEl("val-primary-lift-p50", k50.precision_lift ? k50.precision_lift + "x" : undefined);

  // If live chart image exists, swap placeholder
  const chartContainer = document.getElementById("fig-precision-container");
  if (chartContainer) {
    chartContainer.innerHTML = `<img src="../outputs/figures/fig3_precision_at_k.png" alt="Precision@K Curve" style="max-width:100%; border-radius:8px;" />`;
  }
}
