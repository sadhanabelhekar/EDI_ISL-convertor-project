
const video = document.getElementById('preview');
const status = document.getElementById('status');
const startBtn = document.getElementById('startBtn');

let currentStream = null;

startBtn.addEventListener('click', async () => {
    status.textContent = 'Requesting camera access...';

    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
            video.srcObject = null;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        currentStream = stream;
        video.srcObject = stream;

        await video.play();

        status.textContent = 'Webcam active';
    } catch (err) {
        status.textContent = 'Error: ' + err.name + ' — ' + err.message;
        console.error('Camera error:', err.name, err.message);
    }
});
