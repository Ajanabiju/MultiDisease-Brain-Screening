/* =====================================================
   NEUROSCREEN FRONTEND
===================================================== */


/* =====================================================
   ELEMENTS
===================================================== */

const mriInput = document.getElementById("mriInput");

const uploadArea = document.getElementById("uploadArea");

const selectedFile = document.getElementById("selectedFile");

const analyzeBtn = document.getElementById("analyzeBtn");

const mriPreview = document.getElementById("mriPreview");

const previewStatus = document.getElementById("previewStatus");

const previewFile = document.getElementById("previewFile");

const previewFormat = document.getElementById("previewFormat");

const resultSection = document.getElementById("resultSection");

const resultStatus = document.getElementById("resultStatus");

const resultTitle = document.getElementById("resultTitle");

const resultDescription = document.getElementById("resultDescription");


/* =====================================================
   MRI FILE SELECTION
===================================================== */

mriInput.addEventListener("change", function () {

    if (this.files.length === 0) {

        resetMRI();

        return;

    }


    handleMRIFile(this.files[0]);

});


/* =====================================================
   HANDLE MRI
===================================================== */

function handleMRIFile(file) {

    selectedFile.textContent =
        "✓ " + file.name;

    selectedFile.classList.add("selected");


    previewFile.textContent =
        file.name;


    const extension =
        getFileExtension(file.name);

    previewFormat.textContent =
        extension.toUpperCase();


    previewStatus.textContent =
        "READY";

    previewStatus.classList.add("ready");


    /* Image preview */

    if (file.type.startsWith("image/")) {

        const reader =
            new FileReader();


        reader.onload = function (event) {

            mriPreview.innerHTML = `
                <img
                    src="${event.target.result}"
                    alt="MRI Preview"
                >
            `;

        };


        reader.readAsDataURL(file);

    }

    else {

        mriPreview.innerHTML = `

            <div class="preview-placeholder">

                <div class="large-brain">
                    🧠
                </div>

                <p>
                    NIfTI MRI file selected
                </p>

                <small>
                    ${file.name}
                </small>

            </div>

        `;

    }


    /* Show result area */

    resultSection.classList.add("show");

}


/* =====================================================
   RESET MRI
===================================================== */

function resetMRI() {

    selectedFile.textContent =
        "No MRI selected";

    selectedFile.classList.remove("selected");


    previewFile.textContent =
        "—";


    previewFormat.textContent =
        "—";


    previewStatus.textContent =
        "WAITING";

    previewStatus.classList.remove("ready");


    mriPreview.innerHTML = `

        <div class="preview-placeholder">

            <div class="large-brain">
                🧠
            </div>

            <p>
                MRI preview will appear here
            </p>

        </div>

    `;

}


/* =====================================================
   GET EXTENSION
===================================================== */

function getFileExtension(filename) {

    if (filename.toLowerCase().endsWith(".nii.gz")) {

        return "nii.gz";

    }


    const parts =
        filename.split(".");

    return parts.length > 1
        ? parts.pop()
        : "unknown";

}


/* =====================================================
   DRAG & DROP
===================================================== */

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.classList.add("dragging");

    }
);


uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.classList.remove("dragging");

    }
);


uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadArea.classList.remove("dragging");


        const files =
            event.dataTransfer.files;


        if (files.length > 0) {

            const file =
                files[0];

            handleMRIFile(file);

        }

    }
);


/* =====================================================
   START AI ANALYSIS
===================================================== */

analyzeBtn.addEventListener(
    "click",
    function () {

        if (!mriInput.files.length) {

            alert(
                "Please select an MRI scan first."
            );

            return;

        }


        /* Loading state */

        analyzeBtn.disabled = true;

        analyzeBtn.innerHTML = `
            <span>Processing MRI...</span>
            <span>⟳</span>
        `;


        resultSection.classList.add("show");


        resultStatus.textContent =
            "PROCESSING";


        resultTitle.textContent =
            "AI Pipeline Running";


        resultDescription.textContent =
            "The selected MRI has entered the planned research pipeline: preprocessing → brain parcellation → ViT → GNN → feature fusion.";


        /* Simulated processing */

        setTimeout(function () {

            analyzeBtn.disabled = false;

            analyzeBtn.innerHTML = `
                <span>Start AI Analysis</span>
                <span>→</span>
            `;


            resultStatus.textContent =
                "MODEL NOT CONNECTED";


            resultTitle.textContent =
                "MRI Successfully Submitted";


            resultDescription.textContent =
                "The frontend workflow completed successfully. A trained and validated AI model is not connected yet, so no disease prediction is generated.";

        }, 2500);

    }
);


/* =====================================================
   NAVIGATION
===================================================== */

function scrollToAnalysis() {

    document
        .getElementById("analysis")
        .scrollIntoView({
            behavior: "smooth"
        });

}


function scrollToPipeline() {

    document
        .getElementById("pipeline")
        .scrollIntoView({
            behavior: "smooth"
        });

}


/* =====================================================
   LOGIN MODAL
===================================================== */

const loginBtn =
    document.getElementById("loginBtn");

const loginModal =
    document.getElementById("loginModal");

const closeModal =
    document.getElementById("closeModal");


loginBtn.addEventListener(
    "click",
    function () {

        loginModal.classList.remove("hidden");

    }
);


closeModal.addEventListener(
    "click",
    function () {

        loginModal.classList.add("hidden");

    }
);


loginModal.addEventListener(
    "click",
    function (event) {

        if (event.target === loginModal) {

            loginModal.classList.add("hidden");

        }

    }
);


/* =====================================================
   DEMO LOGIN
===================================================== */

function fakeLogin() {

    alert(
        "Login interface is currently a frontend demonstration."
    );

    loginModal.classList.add("hidden");

}


/* =====================================================
   INITIAL STATE
===================================================== */

console.log(
    "NeuroScreen frontend loaded successfully."
);

console.log(
    "Research prototype - AI model not connected."
);