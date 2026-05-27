document.addEventListener("DOMContentLoaded", () => {

    const cards = document.querySelectorAll(
        ".feature-card, .model-card, .stat-box"
    );

    cards.forEach((card) => {

        card.addEventListener("mousemove", (e) => {

            const rect = card.getBoundingClientRect();

            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            card.style.background = `
                radial-gradient(
                    circle at ${x}px ${y}px,
                    rgba(59,130,246,0.18),
                    rgba(255,255,255,0.04)
                )
            `;

        });

        card.addEventListener("mouseleave", () => {

            card.style.background =
                "rgba(255,255,255,0.05)";

        });

    });

    // Smooth scroll navigation

    const navLinks = document.querySelectorAll(
        '.nav-links a'
    );

    navLinks.forEach((link) => {

        link.addEventListener("click", (e) => {

            e.preventDefault();

            const targetId =
                link.getAttribute("href");

            const target =
                document.querySelector(targetId);

            if (target) {

                target.scrollIntoView({

                    behavior: "smooth"

                });

            }

        });

    });

});