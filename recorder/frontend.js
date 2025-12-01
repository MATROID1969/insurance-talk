const streamlit = window.streamlit;

let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        audioChunks = [];
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            let blob = new Blob(audioChunks, { type: "audio/webm" });
            let reader = new FileReader();

            reader.onloadend = () => {
                streamlit.setComponentValue(reader.result);
            };

            reader.readAsDataURL(blob);  // Base64
        };
    });
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
}

streamlit.onRenderEvent((event) => {
    let action = event.args.action;

    if (action === "start") {
        startRecording();
    } else if (action === "stop") {
        stopRecording();
    }
});
