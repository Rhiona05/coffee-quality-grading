let featureChart;
let latestPredictionData = null;

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");

const resultCard = document.getElementById("resultCard");
const gradeText = document.getElementById("gradeText");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const recommendationText = document.getElementById("recommendationText");
const loading = document.getElementById("loading");

// Preview Image
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        previewImg.src = URL.createObjectURL(file);
        previewImg.style.display = "block";
        resultCard.classList.add("hidden");
    }
});

// Form Submit
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) { alert("Please select an image first!"); return; }

    const formData = new FormData();
    formData.append("file", file);

    loading.classList.remove("hidden");
    resultCard.classList.add("hidden");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        loading.classList.add("hidden");

        if (data.grade) {
            latestPredictionData = data; // store for PDF

            // Update UI
            gradeText.textContent = `Predicted Grade: ${data.grade}`;
            confidenceFill.style.width = data.confidence + "%";
            confidenceText.textContent = `Confidence: ${data.confidence}%`;
            recommendationText.textContent = data.recommendation;

            // Feature Chart
            const features = data.features;
            const labels = Object.keys(features);
            const values = Object.values(features);

            document.getElementById("featureValues").innerHTML = `
                Contrast: ${features.contrast.toFixed(4)}<br>
                Energy: ${features.energy.toFixed(4)}<br>
                Homogeneity: ${features.homogeneity.toFixed(4)}<br>
                Correlation: ${features.correlation.toFixed(4)}<br>
                LBP Mean: ${features.lbp_mean.toFixed(4)}<br>
                LBP Std: ${features.lbp_std.toFixed(4)}
            `;

            const ctx = document.getElementById('featureChart').getContext('2d');
            if (featureChart) featureChart.destroy();
            featureChart = new Chart(ctx, {
                type: 'bar',
                data: { labels: labels, datasets: [{ label: 'Feature Values', data: values, backgroundColor: 'rgba(90, 42, 39, 0.7)' }] },
                options: { responsive: true }
            });

            resultCard.classList.remove("hidden");
        } else {
            alert(data.error || "Unexpected server response");
        }

    } catch (error) {
        loading.classList.add("hidden");
        alert("Server error. Check backend.");
        console.error(error);
    }
});

// Download PDF Report
document.getElementById("downloadPDFBtn").addEventListener("click", function() {
    if (!latestPredictionData) { alert("No analysis available."); return; }
    const filename = latestPredictionData.filename;
    window.location.href = "/export/pdf/" + filename;
});