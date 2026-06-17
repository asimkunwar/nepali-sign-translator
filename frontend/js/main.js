/* ============================================================
   SanketAI — main.js
   Captures webcam frames, extracts hand landmarks with the
   MediaPipe Tasks Vision bundle, and streams them to the
   FastAPI backend over WebSocket for prediction.
   ============================================================ */

const { HandLandmarker, FilesetResolver } = window;

// --- DOM references ---
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const videoPlaceholder = document.getElementById('video-placeholder');

const nepaliTextDisplay = document.getElementById('nepali-text');
const confidenceDisplay = document.getElementById('confidence-score');
const handsTag = document.getElementById('hands-tag');
const stateTag = document.getElementById('state-tag');
const connDot = document.getElementById('conn-dot');
const connLabel = document.getElementById('conn-label');
const signalBars = document.querySelectorAll('#signal-bars .bar');
const logList = document.getElementById('log-list');
const footerFps = document.getElementById('footer-fps');

// --- Config ---
const WS_URL = "ws://127.0.0.1:8000/api/stream";
const TOTAL_EXPECTED = 84; // 2 hands x 21 joints x 2 coords (x, y)
const PLACEHOLDER_TEXT = "हात देखाउनुहोस्"; // "Show your hand"
const MAX_LOG_ENTRIES = 6;

let websocket = null;
let handLandmarker = null;
let lastVideoTime = -1;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastLoggedText = null;

// ============================================================
// 1. WebSocket connection to the FastAPI inference server
// ============================================================
function connectWebSocket() {
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
        console.log("📡 Connected to SanketAI Live Stream Engine!");
        setConnectionState('connected');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateTranslation(data);
    };

    websocket.onclose = () => {
        console.log("❌ Stream disconnected. Reconnecting...");
        setConnectionState('disconnected');
        setTimeout(connectWebSocket, 2000); // auto-reconnect
    };

    websocket.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}

function setConnectionState(state) {
    connDot.classList.remove('connected', 'disconnected');
    if (state === 'connected') {
        connDot.classList.add('connected');
        connLabel.textContent = 'Engine connected';
    } else if (state === 'disconnected') {
        connDot.classList.add('disconnected');
        connLabel.textContent = 'Engine disconnected — retrying…';
    } else {
        connLabel.textContent = 'Connecting to engine…';
    }
}

// ============================================================
// 2. Update result UI from a backend message
//    Expected shape: { nepali_text: string, confidence: number, hands_detected?: number }
// ============================================================
function updateTranslation(data) {
    const text = data.nepali_text || PLACEHOLDER_TEXT;
    const confidence = typeof data.confidence === 'number' ? data.confidence : parseFloat(data.confidence) || 0;
    const isMatch = text !== PLACEHOLDER_TEXT && text.trim().length > 0;

    nepaliTextDisplay.classList.toggle('placeholder-text', !isMatch);
    nepaliTextDisplay.classList.toggle('matched', isMatch && confidence >= 70);
    nepaliTextDisplay.innerText = text;

    confidenceDisplay.innerText = `${confidence.toFixed(0)}%`;
    updateSignalBars(confidence);

    stateTag.textContent = isMatch ? 'Translating' : 'Idle';
    stateTag.classList.toggle('active', isMatch);

    if (isMatch && confidence >= 60 && text !== lastLoggedText) {
        addLogEntry(text, confidence);
        lastLoggedText = text;
    }
    if (!isMatch) lastLoggedText = null;
}

function updateSignalBars(confidence) {
    const litCount = Math.round((confidence / 100) * signalBars.length);
    signalBars.forEach((bar, i) => {
        const lit = i < litCount;
        bar.classList.toggle('lit', lit);
        bar.classList.toggle('strong', lit && confidence >= 70);
        bar.style.height = lit ? `${40 + (i / signalBars.length) * 60}%` : '30%';
    });
}

function addLogEntry(text, confidence) {
    const emptyState = logList.querySelector('.log-empty');
    if (emptyState) emptyState.remove();

    const li = document.createElement('li');
    li.innerHTML = `<span>${text}</span><span class="log-conf">${confidence.toFixed(0)}%</span>`;
    logList.prepend(li);

    while (logList.children.length > MAX_LOG_ENTRIES) {
        logList.removeChild(logList.lastChild);
    }
}

// ============================================================
// 3. MediaPipe HandLandmarker setup
//    Tries the GPU delegate first (faster), and falls back to CPU
//    if the GPU backend isn't supported on this machine/browser —
//    which is the most common reason model creation fails.
// ============================================================
async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );

    const modelAssetPath = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
    const commonOptions = {
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3
    };

    try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath, delegate: "GPU" },
            ...commonOptions
        });
        console.log("✅ HandLandmarker created with GPU delegate.");
    } catch (gpuErr) {
        console.warn("⚠️ GPU delegate failed, falling back to CPU:", gpuErr);
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath, delegate: "CPU" },
            ...commonOptions
        });
        console.log("✅ HandLandmarker created with CPU delegate.");
    }
}

// ============================================================
// 4. Start webcam + landmarker, then begin the frame loop
//
//    IMPORTANT: the permission prompt only appears in response to a
//    direct user gesture (the click) and only on the FIRST await
//    inside that gesture. So getUserMedia() must be called immediately,
//    before any other async work (like loading the ML model) — otherwise
//    some browsers silently drop the "is a user gesture" flag and the
//    permission dialog never shows up, which looks exactly like a
//    permissions error even though the user was never actually asked.
// ============================================================
async function startWebcam() {
    // Guard: getUserMedia requires a secure context (https:// or localhost).
    // On a plain file:// page or http:// over LAN, the API doesn't exist at all.
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(
            "क्यामेरा यो पेजमा उपलब्ध छैन। कृपया यो साइट http://localhost वा https:// मार्फत खोल्नुहोस्, फाइल सीधै नखोल्नुहोस्।\n\n" +
            "(Camera isn't available on this page. Open this site via http://localhost or https:// — not by double-clicking the HTML file directly.)"
        );
        return;
    }

    startBtn.disabled = true;
    startBtn.innerText = "अनुमति पर्खँदै…"; // "Waiting for permission…"

    let stream;
    try {
        // Request camera FIRST — this is what triggers the browser's
        // native Allow / Block permission popup.
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: { ideal: 24 } },
            audio: false
        });
    } catch (err) {
        console.error("getUserMedia failed:", err);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            alert(
                "क्यामेरा अनुमति अस्वीकृत भयो। ब्राउजरको एड्रेस बारमा रहेको क्यामेरा आइकनमा क्लिक गरी 'Allow' छान्नुहोस्, अनि पेज रिफ्रेस गर्नुहोस्।\n\n" +
                "(Camera permission was blocked. Click the camera icon in the browser's address bar, choose 'Allow', then refresh the page.)"
            );
        } else if (err.name === 'NotFoundError') {
            alert(
                "कुनै क्यामेरा फेला परेन। क्यामेरा जडित छ कि छैन जाँच गर्नुहोस्।\n\n(No camera device was found. Please check that a webcam is connected.)"
            );
        } else if (err.name === 'NotReadableError') {
            alert(
                "क्यामेरा अर्को एपमा प्रयोग भइरहेको छ। अन्य एप (Zoom, Teams आदि) बन्द गरी पुन: प्रयास गर्नुहोस्।\n\n" +
                "(The camera is already in use by another app. Close other apps using the camera — Zoom, Teams, etc. — and try again.)"
            );
        } else {
            alert(
                "क्यामेरा खोल्न सकिएन। ब्राउजर अनुमति जाँच गर्नुहोस्।\n\n(Unable to access webcam. Please check browser permissions.)"
            );
        }
        startBtn.disabled = false;
        startBtn.innerText = "क्यामेरा सुरु गर्नुहोस्";
        return;
    }

    // Camera is granted and live — show it immediately so the user gets
    // instant feedback, then load the ML model in the background.
    webcamElement.srcObject = stream;
    await webcamElement.play();
    videoPlaceholder.classList.add('hidden');
    startBtn.innerText = "मोडेल लोड हुँदैछ…"; // "Loading model…"

    try {
        if (!handLandmarker) {
            await createHandLandmarker();
        }
    } catch (err) {
        console.error("Model load failed:", err);
        alert(
            "AI मोडेल लोड हुन सकेन। इन्टरनेट जडान जाँच गर्नुहोस् र पेज रिफ्रेस गर्नुहोस्।\n\n" +
            "(The hand-tracking model failed to load. Check your internet connection and refresh the page.)\n\n" +
            "Technical detail: " + (err && err.message ? err.message : String(err))
        );
        startBtn.disabled = false;
        startBtn.innerText = "क्यामेरा सुरु गर्नुहोस्";
        return;
    }

    startBtn.style.display = 'none';
    connectWebSocket();
    requestAnimationFrame(processWebcamFrame);
}

// ============================================================
// 5. Per-frame loop: detect hands, draw landmarks, send coords
// ============================================================
const HAND_COLORS = ['#C8273D', '#D6A24A']; // accent crimson, warm gold — distinct per hand

function processWebcamFrame() {
    if (webcamElement.readyState >= 2 && webcamElement.currentTime !== lastVideoTime) {
        lastVideoTime = webcamElement.currentTime;

        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        const nowMs = performance.now();
        const result = handLandmarker.detectForVideo(webcamElement, nowMs);

        let handsDetected = 0;
        let coordinates = [];

        if (result && result.landmarks && result.landmarks.length > 0) {
            handsDetected = result.landmarks.length;

            // Sort left-to-right by wrist x, identical convention to the Python training pipeline
            const sortedHands = [...result.landmarks].sort((a, b) => a[0].x - b[0].x);

            sortedHands.forEach((handLandmarks, handIndex) => {
                const color = HAND_COLORS[handIndex % HAND_COLORS.length];
                handLandmarks.forEach((lm) => {
                    const cx = lm.x * canvasElement.width;
                    const cy = lm.y * canvasElement.height;
                    canvasCtx.beginPath();
                    canvasCtx.arc(cx, cy, 4, 0, 2 * Math.PI);
                    canvasCtx.fillStyle = color;
                    canvasCtx.fill();

                    coordinates.push(lm.x, lm.y);
                });
            });
        }

        canvasCtx.restore();

        handsTag.textContent = `${handsDetected} / 2 hands`;
        handsTag.classList.toggle('active', handsDetected === 2);

        // Only send a full 84-value frame; otherwise tell the backend hands are missing
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (handsDetected === 2 && coordinates.length === TOTAL_EXPECTED) {
                websocket.send(JSON.stringify({ coordinates }));
            } else {
                websocket.send(JSON.stringify({ coordinates: null }));
            }
        }

        // Lightweight FPS counter in the footer
        frameCount++;
        const elapsed = nowMs - lastFpsTime;
        if (elapsed >= 1000) {
            footerFps.textContent = `${Math.round((frameCount * 1000) / elapsed)} fps`;
            frameCount = 0;
            lastFpsTime = nowMs;
        }
    }

    requestAnimationFrame(processWebcamFrame);
}

// ============================================================
// Event listeners
// ============================================================
startBtn.addEventListener('click', startWebcam);