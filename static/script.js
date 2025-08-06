document.getElementById("uploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please choose an audio file.");
        return;
    }

    // Show loading spinner
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/transcribe-and-summarize", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        window.alert(JSON.stringify(data));

        // document.getElementById("transcript").innerHTML = marked.parse(data.transcript_text || "No transcript returned.");

        // document.getElementById("summary").innerHTML = marked.parse(data.summary_text || "No summary returned.");

        // document.getElementById("downloadTranscript").href = data.download_links.transcript;
        // document.getElementById("downloadSummary").href = data.download_links.summary;

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("results").classList.remove("hidden");

    } catch (error) {
        console.error(error);
        alert("An error occurred during transcription. Check the console for details.");
        document.getElementById("loading").classList.add("hidden");
    }
});
