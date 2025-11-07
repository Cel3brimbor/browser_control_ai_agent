document.addEventListener('DOMContentLoaded', () => {
    const closeButton = document.getElementById('closeButton');

    if (closeButton) {
        closeButton.onclick = () => {
            window.close();
        };
    }
});