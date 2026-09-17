const video = document.getElementById('preview');
const status = document.getElementById('status');
const startBtn = document.getElementById('startBtn');

startBtn.addEventListener('click', async () => {
  status.textContent = 'Requesting camera...';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    video.srcObject = stream;
    status.textContent = 'Webcam active';
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    console.error('getUserMedia failed:', err);
  }
});
