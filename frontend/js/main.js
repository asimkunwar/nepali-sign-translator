let websocket;
let webcamElement = document.getElementById('webcam');
let startBtn = document.getElementById('start-btn');
let nepaliTextDisplay = document.getElementById('nepali-text');
let confidenceDisplay = document.getElementById('confidence-score');

// 1. Initialize WebSocket Connection to our FastAPI Server
function connectWebSocket() {
    websocket = new WebSocket("ws://127.0.0.1:8000/api/stream");

    websocket.onopen = () => {
        console.log("📡 Connected to SanketAI Live Stream Engine!");
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Remove placeholder styling once we get actual words
        nepaliTextDisplay.classList.remove('placeholder-text');
        
        // Dynamic UI Updates
        nepaliTextDisplay.innerText = data.nepali_text;
        confidenceDisplay.innerText = data.confidence;
    };

    websocket.onclose = () => {
        console.log("❌ Stream disconnected. Reconnecting...");
        setTimeout(connectWebSocket, 2000); // Auto-reconnect safety loop
    };
}

// 2. Start Hardware Webcam Stream
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: { ideal: 15 } },
            audio: false
        });
        webcamElement.srcObject = stream;
        startBtn.style.display = 'none'; // Hide button once streaming
        
        // Connect to our AI backend pipeline
        connectWebSocket();
        
        // Kick off the frame processing loop
        requestAnimationFrame(processWebcamFrame);
    } catch (err) {
        alert("Unable to access webcam. Please verify browser permissions!");
        console.error(err);
    }
}

// 3. Frame Engine Loop (Fakes landmark extraction for testing connection first)
function processWebcamFrame() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Mocking an array of 84 coordinates (2 hands * 21 points * 2 dimensions [x, y])
        // We will tie the real MediaPipe Vision bundle object directly to this array next!
        let mockCoordinates = new Array(84).fill(0.0);
        
        websocket.send(JSON.stringify({
            "coordinates": mockCoordinates
        }));
    }
    // Keep looping smoothly
    setTimeout(processWebcamFrame, 100); 
}

// Event Listeners
startBtn.addEventListener('click', startWebcam);