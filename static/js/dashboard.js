document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const clearBtn = document.getElementById("clearBtn");
    const reviewBox = document.getElementById("reviewBox");
    const analysisPanel = document.querySelector(".analysis-panel");
    const typewriterText = document.getElementById("typewriterText");

    if (form && analysisPanel) {
        form.addEventListener("submit", () => {
            analysisPanel.innerHTML = `
                <div class="inline-loading">
                    <div class="inline-spinner"></div>
                    <div class="loading-text" style="font-size: 18px;">Analyzing Review...</div>
                </div>
            `;
        });
    }

    if (clearBtn && reviewBox && analysisPanel) {
        clearBtn.addEventListener("click", () => {
            reviewBox.value = "";
            analysisPanel.innerHTML = `
                <div class="empty-state">
                    <h2>Awaiting Analysis</h2>
                    <p>
                        Enter a movie review and select a model
                        to begin AI-powered sentiment analysis.
                    </p>
                </div>
            `;
        });
    }

    if (typewriterText) {
        const originalText = typewriterText.innerText;
        typewriterText.innerText = "";
        let index = 0;

        function typeWriter() {
            if (index < originalText.length) {
                typewriterText.innerHTML += originalText.charAt(index);
                index++;
                setTimeout(typeWriter, 22);
            }
        }
        typeWriter();
    }

    const confidenceFill = document.querySelector(".confidence-fill");
    const confidencePercent = document.querySelector(".confidence-percent");

    if (confidenceFill) {
        const finalWidth = confidenceFill.style.width;
        confidenceFill.style.width = "0%";

        setTimeout(() => {
            confidenceFill.style.width = finalWidth;
        }, 300);
    }

    if (confidencePercent && confidenceFill) {
        const value = parseFloat(confidencePercent.innerText);
        if (value >= 85) {
            confidenceFill.style.boxShadow = "0 0 30px rgba(74,222,128,0.7)";
        } else if (value >= 70) {
            confidenceFill.style.boxShadow = "0 0 25px rgba(56,189,248,0.6)";
        } else {
            confidenceFill.style.boxShadow = "0 0 20px rgba(248,113,113,0.5)";
        }
    }

    if (reviewBox) {
        reviewBox.addEventListener("input", () => {
            reviewBox.style.boxShadow = "0 0 30px rgba(56,189,248,0.4)";
        });

        reviewBox.addEventListener("blur", () => {
            reviewBox.style.boxShadow = "none";
        });
    }

    document.addEventListener("mousemove", (e) => {
        const glow = document.querySelector(".sidebar-glow");
        if (!glow) return;

        const x = e.clientX * 0.02;
        const y = e.clientY * 0.02;
        glow.style.transform = `translate(${x}px, ${y}px)`;
    });

    const sentimentHeader = document.querySelector(".sentiment-header");
    if (sentimentHeader) {
        if (sentimentHeader.innerText.includes("POSITIVE")) {
            document.title = "Positive Sentiment • CineSense AI";
        } else {
            document.title = "Negative Sentiment • CineSense AI";
        }
    }

    const dropdownSelected = document.getElementById("dropdownSelected");
    const dropdownOptions = document.getElementById("dropdownOptions");
    const modelInput = document.getElementById("modelInput");
    const dropdownOptionItems = document.querySelectorAll(".dropdown-option");

    if (dropdownSelected && dropdownOptions) {
        dropdownSelected.addEventListener("click", () => {
            dropdownOptions.classList.toggle("hidden");
        });

        dropdownOptionItems.forEach((option) => {
            option.addEventListener("click", () => {
                const value = option.dataset.value;
                dropdownSelected.innerText = value;
                if (modelInput) modelInput.value = value;
                dropdownOptions.classList.add("hidden");
            });
        });

        document.addEventListener("click", (e) => {
            if (!e.target.closest(".custom-dropdown")) {
                dropdownOptions.classList.add("hidden");
            }
        });
    }

    if (modelInput && dropdownSelected) {
        dropdownSelected.innerText = modelInput.value;
    }

    const reviewData = document.getElementById("reviewData");
    if (reviewBox && reviewData) {
        reviewBox.value = reviewData.value.trim();
    }
});