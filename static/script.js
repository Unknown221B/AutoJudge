function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = (textarea.scrollHeight + 4) + "px";
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("textarea").forEach(t => {
        autoResize(t);
        t.addEventListener("input", () => autoResize(t));
    });

    const result = document.getElementById("result");
    if (result) {
        result.scrollIntoView({ behavior: "smooth" });
    }
});