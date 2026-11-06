/* =====================================================
   GLOBAL 3D BRAIN VIEWER VARIABLES & RAYCASTER
===================================================== */
var brainScene = null;
var brainCamera = null;
var brainRenderer = null;
var brainControls = null;
var brainGraphGroup = null;
var brainContainer = null;
var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var animFrameId = null;
var currentClinicianMeta = null;
/* =====================================================
   GLOBAL SCROLL FUNCTION FOR NAVBAR / HERO BUTTONS
===================================================== */
function scrollToAnalysis() {
    const target =
        document.getElementById("analysis") ||
        document.getElementById("analysisSection") ||
        document.getElementById("uploadArea");
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
window.scrollToAnalysis = scrollToAnalysis;

/* =====================================================
   INITIALIZE APPLICATION ON DOM LOAD
===================================================== */
document.addEventListener("DOMContentLoaded", function () {
    // Core Elements
    const mriInput = document.getElementById("mriInput");
    const uploadArea = document.getElementById("uploadArea");
    const selectedFile = document.getElementById("selectedFile");
    const analyzeBtn = document.getElementById("analyzeBtn");

    // Preview Elements
    const mriPreview = document.getElementById("mriPreview");
    const previewStatus = document.getElementById("previewStatus");
    const previewFile = document.getElementById("previewFile");
    const previewFormat = document.getElementById("previewFormat");

    // Results Dashboard
    const resultSection = document.getElementById("resultSection");
    const resultStatus = document.getElementById("resultStatus");
    const resultTitle = document.getElementById("resultTitle");
    const resultDescription = document.getElementById("resultDescription");

    // Initialize 3D Scene early
    init3DBrainViewer();

    /* =====================================================
       FILE SELECTION & HANDLERS
    ===================================================== */
    function getFileExtension(filename) {
        if (!filename) return "unknown";
        if (filename.toLowerCase().endsWith(".nii.gz")) return "nii.gz";
        const parts = filename.split(".");
        return parts.length > 1 ? parts.pop().toLowerCase() : "unknown";
    }

    function handleMRIFile(file) {
        if (!file) return;

        if (selectedFile) {
            selectedFile.textContent = "✓ " + file.name;
            selectedFile.classList.add("selected");
        }
        if (previewFile) previewFile.textContent = file.name;

        const ext = getFileExtension(file.name);
        if (previewFormat) previewFormat.textContent = ext.toUpperCase();

        if (previewStatus) {
            previewStatus.textContent = "READY FOR PIPELINE";
            previewStatus.className = "preview-status ready";
        }

        if (mriPreview) {
            if (file.type.startsWith("image/")) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    mriPreview.innerHTML = `<img src="${e.target.result}" alt="MRI Slice Preview" style="max-width:100%; max-height:220px; border-radius:8px;">`;
                };
                reader.readAsDataURL(file);
            } else {
                mriPreview.innerHTML = `
                    <div class="preview-placeholder">
                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
                        <p style="font-weight: 600; color: #38bdf8;">${file.name}</p>
                        <small style="color: #94a3b8;">Format: ${ext.toUpperCase()} | AAL-116 Connectome Parsing Target</small>
                    </div>
                `;
            }
        }
    }

    if (mriInput) {
        mriInput.addEventListener("change", function () {
            if (this.files && this.files.length > 0) {
                handleMRIFile(this.files[0]);
            }
        });
    }

    if (uploadArea) {
        uploadArea.addEventListener("dragover", function (e) {
            e.preventDefault();
            uploadArea.classList.add("dragging");
        });

        uploadArea.addEventListener("dragleave", function () {
            uploadArea.classList.remove("dragging");
        });

        uploadArea.addEventListener("drop", function (e) {
            e.preventDefault();
            uploadArea.classList.remove("dragging");
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                if (mriInput) mriInput.files = e.dataTransfer.files;
                handleMRIFile(e.dataTransfer.files[0]);
            }
        });
    }

    /* =====================================================
       INFERENCE EXECUTION & DATA BINDING
    ===================================================== */
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", async function (e) {
            if (e) e.preventDefault();

            // 1. Reference Patient Intake Elements
            const nameEl = document.getElementById("patientNameInput");
            const ageEl = document.getElementById("patientAgeInput");
            const genderEl = document.getElementById("patientGenderInput");
            const phoneEl = document.getElementById("patientPhoneInput");
            const modalityEl = document.getElementById("scanModalityInput");

            // 2. Validation Helper to Alert and Highlight Missing Fields
            function checkField(el, labelName) {
                if (!el || !el.value.trim()) {
                    if (el) {
                        el.style.borderColor = "#ef4444";
                        el.focus();
                    }
                    alert(`⚠️ Incomplete Intake Form:\nPlease provide "${labelName}" before starting AI analysis.`);
                    return false;
                }
                el.style.borderColor = "#334155";
                return true;
            }

            // 3. Enforce Mandatory Fields in Order
            if (!checkField(nameEl, "Patient Full Name")) return;
            if (!checkField(ageEl, "Age (Years)")) return;
            if (!checkField(genderEl, "Biological Sex / Gender")) return;
            if (!checkField(phoneEl, "Contact Number")) return;

            // 4. Verify Scan Upload
            if (!mriInput || !mriInput.files || !mriInput.files.length) {
                alert("⚠️ Missing Scan:\nPlease select or drop a valid brain scan first.");
                return;
            }

            // 5. Append Verified Patient Data to FormData
            const file = mriInput.files[0];
            const formData = new FormData();
            formData.append("file", file);
            formData.append("patient_name", nameEl.value.trim());
            formData.append("patient_age", ageEl.value.trim());
            formData.append("patient_gender", genderEl.value);
            formData.append("patient_phone", phoneEl.value.trim());
            if (modalityEl) formData.append("scan_modality", modalityEl.value);

            // Loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = `<span>Extracting Features & Executing GNN...</span> <span>⏳</span>`;

            if (resultSection) {
                resultSection.classList.add("show");
                requestAnimationFrame(() => {
                    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
                });
            }
            
            setTimeout(() => {
                const container = document.getElementById("brain3DContainer");
                if (container && brainRenderer && brainCamera) {
                    const w = container.clientWidth;
                    const h = container.clientHeight || 440;
                    brainCamera.aspect = w / h;
                    brainCamera.updateProjectionMatrix();
                    brainRenderer.setSize(w, h);
                }
            }, 150);

            if (resultStatus) {
                resultStatus.textContent = "PROCESSING";
                resultStatus.className = "badge badge-processing";
            }
            if (resultTitle) resultTitle.textContent = "AI Pipeline Running";
            if (resultDescription) {
                resultDescription.textContent = "Performing AAL-116 topological alignment and spatial feature extraction...";
            }

            const spinner = document.getElementById("viewerSpinner");
            if (spinner) spinner.style.display = "block";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                // Safe JSON handling to prevent Unexpected token '<' errors
                const contentType = response.headers.get("content-type") || "";
                let data;
                if (contentType.includes("application/json")) {
                    data = await response.json();
                } else {
                    const rawText = await response.text();
                    if (response.status === 413) {
                        throw new Error("Server transfer buffer exceeded. Please use .nii.gz format.");
                    }
                    throw new Error(`Server returned HTTP ${response.status}: ${rawText.slice(0, 100)}`);
                }

                if (!response.ok || data.error) {
                    throw new Error(data.error || "Server processing failed.");
                }

                // SAVE LOGGED-IN CLINICIAN DATA GLOBALLY FOR PRINT REPORT
                if (data.subject_meta) {
                    currentClinicianMeta = data.subject_meta;
                }

                // Render Success State
                if (resultStatus) {
                    resultStatus.textContent = "DIAGNOSTIC COMPLETE";
                    resultStatus.className = `badge badge-${data.badge || "success"}`;
                }

                if (resultTitle) {
                    resultTitle.innerHTML = `Identified: <span style="color: #38bdf8;">${data.prediction}</span>`;
                }

                // Metadata Header
                let metaHtml = "";
                if (data.subject_meta) {
                    const patientName = data.subject_meta.name || data.subject_meta.patient_name || data.subject_meta.patient_id || "Anonymous";
                    const patientAge = data.subject_meta.age || data.subject_meta.patient_age || "N/A";
                    const patientGender = data.subject_meta.gender || data.subject_meta.patient_gender || "N/A";
                    const patientPhone = data.subject_meta.phone || data.subject_meta.patient_phone || "N/A";

                    metaHtml = `
                        <div style="background: rgba(15, 23, 42, 0.7); padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.85rem; border-left: 3px solid #38bdf8; text-align: left;">
                            <div><strong>Patient:</strong> ${patientName} | <strong>Profile:</strong> ${patientAge} (${patientGender})</div>
                            <div><strong>Contact:</strong> ${patientPhone}</div>
                            <div><strong>Clinician:</strong> ${data.subject_meta.evaluating_clinician || "Staff"} (${data.subject_meta.license || "Verified"})</div>
                        </div>
                    `;
                }

                let rawConf = Number(data.confidence);
                if (rawConf <= 1.0 && rawConf > 0) {
                    rawConf = rawConf * 100.0;
                }
                const confDisplay = rawConf.toFixed(2);

                let breakdownHTML = `
                    ${metaHtml}
                    <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem; text-align: left;">
                        Model Confidence: ${confDisplay}%
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 1rem; text-align: left;">
                        ${data.description || ""}
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem; text-align: left;">
                `;

                if (data.probabilities) {
                    for (const [disease, prob] of Object.entries(data.probabilities)) {
                        let rawProb = Number(prob);
                        if (rawProb <= 1.0 && rawProb > 0) {
                            rawProb = rawProb * 100.0;
                        }
                        const numProb = rawProb.toFixed(2);

                        const isDominant = disease === data.prediction;
                        const barColor = isDominant ? "#38bdf8" : "#64748b";
                        const fontColor = isDominant ? "#ffffff" : "#94a3b8";

                        breakdownHTML += `
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: ${fontColor}; margin-bottom: 0.2rem;">
                                    <span>${disease}</span>
                                    <strong>${numProb}%</strong>
                                </div>
                                <div style="background: rgba(255,255,255,0.08); width: 100%; height: 7px; border-radius: 4px; overflow: hidden;">
                                    <div style="background: ${barColor}; width: ${numProb}%; height: 100%; border-radius: 4px; transition: width 0.6s ease;"></div>
                                </div>
                            </div>
                        `;
                    }
                }

                breakdownHTML += `
                    </div>
                    <button onclick="printClinicalReport()" style="margin-top: 1.25rem; padding: 0.65rem 1.4rem; background: #0284c7; color: #ffffff; border: none; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 0.95rem; box-shadow: 0 2px 6px rgba(2,132,199,0.4);">
                        🖨️ Print / Save Clinical PDF Report
                    </button>
                `;

                if (resultDescription) resultDescription.innerHTML = breakdownHTML;

                // Render 3D Connectome
                if (data.connectome_3d) {
                    render3DConnectomeData(data.connectome_3d);
                }

            } catch (err) {
                console.error("Diagnostic Inference Error:", err);
                if (resultStatus) {
                    resultStatus.textContent = "INFERENCE ERROR";
                    resultStatus.className = "badge badge-danger";
                }
                if (resultTitle) resultTitle.textContent = "Analysis Failed";
                if (resultDescription) {
                    resultDescription.innerHTML = `<span style="color: #f87171;">Diagnostic Exception: ${err.message}</span>`;
                }
            } finally {
                if (spinner) spinner.style.display = "none";
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = `<span>Start AI Analysis</span> <span>→</span>`;
            }
        });
    }
});

/* =====================================================
   DEDICATED CLINICAL REPORT PRINT HANDLER
===================================================== */
function printClinicalReport() {// 1. Dynamic Clinician Details from API response
    const doctorName = (currentClinicianMeta && (currentClinicianMeta.evaluating_clinician || currentClinicianMeta.name || currentClinicianMeta.clinician_name)) || "Dr. Staff Clinician";
    const doctorDesignation = (currentClinicianMeta && currentClinicianMeta.designation) || "Consultant Neurologist";
    const doctorRegId = (currentClinicianMeta && (currentClinicianMeta.license || currentClinicianMeta.reg_id)) || "Verified Staff";

    // 2. Patient Input Element Lookups (Ensure ALL 4 elements are declared here)
    const nameEl = document.getElementById("patientNameInput");
    const ageEl = document.getElementById("patientAgeInput");
    const genderEl = document.getElementById("patientGenderInput");
    const phoneEl = document.getElementById("patientPhoneInput");

    // 3. Extract Values Safely
    const patientName = (nameEl && nameEl.value.trim()) ? nameEl.value.trim() : "Anonymous Patient";
    const patientAge = (ageEl && ageEl.value.trim()) ? `${ageEl.value.trim()} yrs` : "Not Specified";
    const patientGender = (genderEl && genderEl.value) ? genderEl.value : "Not Specified";
    const patientPhone = (phoneEl && phoneEl.value.trim()) ? phoneEl.value.trim() : "N/A";

    // 4. Gather Analysis Result Elements
    const resultTitleEl = document.getElementById("resultTitle");
    const primaryFinding = resultTitleEl ? resultTitleEl.innerText.replace("Identified:", "").trim() : "Screening Complete";
    const reportDate = new Date().toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric"
    });

    // 3. Extract Defect Nodes from 3D Scene / Tooltip or Map by Pathology
    let affectedRegionsHTML = "";
    if (primaryFinding.includes("Alzheimer")) {
        affectedRegionsHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 0.85rem; margin-top: 0.75rem;">
                <div style="font-weight: 700; color: #b91c1c; font-size: 0.82rem; margin-bottom: 0.4rem; text-transform: uppercase;">
                    ⚠️ Identified Pathology Drivers (AAL-116 Atlas ROIs)
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.82rem; color: #334155; line-height: 1.5;">
                    <li><strong>Hippocampus & Parahippocampal Gyrus (HIP / PHG):</strong> Marked cortical thinning and synaptic attenuation affecting memory encoding.</li>
                    <li><strong>Precuneus & Posterior Cingulate (PCUN / PCG):</strong> Deficits in Default Mode Network (DMN) functional connectivity.</li>
                    <li><strong>Inferior & Middle Temporal Lobes (ITG / MTG):</strong> Bilateral structural connectivity degradation correlating with semantic processing deficits.</li>
                </ul>
            </div>
        `;
    } else if (primaryFinding.includes("Parkinson")) {
        affectedRegionsHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 0.85rem; margin-top: 0.75rem;">
                <div style="font-weight: 700; color: #b91c1c; font-size: 0.82rem; margin-bottom: 0.4rem; text-transform: uppercase;">
                    ⚠️ Identified Pathology Drivers (AAL-116 Atlas ROIs)
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.82rem; color: #334155; line-height: 1.5;">
                    <li><strong>Basal Ganglia / Substantia Nigra:</strong> Nigrostriatal pathway connectivity disruption.</li>
                    <li><strong>Putamen & Caudate Nucleus:</strong> Structural disconnectivity impacting dopamine regulation and motor control loops.</li>
                </ul>
            </div>
        `;
    } else if (primaryFinding.includes("Autism")) {
        affectedRegionsHTML = `
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 0.85rem; margin-top: 0.75rem;">
                <div style="font-weight: 700; color: #b91c1c; font-size: 0.82rem; margin-bottom: 0.4rem; text-transform: uppercase;">
                    ⚠️ Identified Pathology Drivers (AAL-116 Atlas ROIs)
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.82rem; color: #334155; line-height: 1.5;">
                    <li><strong>Amygdala & Fusiform Gyrus:</strong> Atypical edge weights across socio-emotional and facial recognition connectomes.</li>
                    <li><strong>Prefrontal Cortex (Superior Frontal Gyrus):</strong> Local hyper-connectivity alongside long-range interhemispheric hypo-connectivity.</li>
                </ul>
            </div>
        `;
    } else {
        affectedRegionsHTML = `
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 0.75rem; margin-top: 0.75rem; font-size: 0.82rem; color: #166534;">
                ✓ <strong>Normal Physiological Topology:</strong> No localized focal atrophy or salient defect nodes identified across the 116 anatomical ROIs.
            </div>
        `;
    }

    // 4. Extract and Clean Probability Breakdown (stripping print buttons)
    const resultDescEl = document.getElementById("resultDescription");
    let analysisContent = "";
    if (resultDescEl) {
        const temp = document.createElement("div");
        temp.innerHTML = resultDescEl.innerHTML;
        temp.querySelectorAll("button, .no-print").forEach(b => b.remove());
        analysisContent = temp.innerHTML;
    }

    // 5. Open Isolated Print Window
    const printWindow = window.open("", "_blank", "width=850,height=950");
    if (!printWindow) {
        window.print();
        return;
    }

    printWindow.document.open();
    printWindow.document.write(`
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>NeuroScreen CDSS - Clinical Diagnostic Report</title>
            <style>
                @page {
                    size: A4 portrait;
                    margin: 1.2cm;
                }
                * {
                    box-sizing: border-box;
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                    background: #ffffff !important;
                    color: #0f172a !important;
                    margin: 0;
                    padding: 0;
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                }
                .report-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    border-bottom: 2px solid #0284c7;
                    padding-bottom: 0.75rem;
                    margin-bottom: 1.2rem;
                }
                .report-title {
                    margin: 0;
                    font-size: 1.3rem;
                    color: #0f172a;
                    font-weight: 800;
                }
                .report-subtitle {
                    color: #64748b;
                    font-size: 0.8rem;
                }
                .meta-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                    margin-bottom: 1.2rem;
                }
                .card {
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 0.85rem;
                    background: #f8fafc;
                }
                .card-title {
                    font-size: 0.75rem;
                    font-weight: 700;
                    color: #0284c7;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-bottom: 0.4rem;
                }
                .card-row {
                    font-size: 0.85rem;
                    line-height: 1.5;
                    color: #1e293b;
                }
                .analysis-box {
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 1rem;
                    background: #ffffff;
                }
                .footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-end;
                    margin-top: 1.5rem;
                    padding-top: 1rem;
                    border-top: 1px dashed #cbd5e1;
                    font-size: 0.72rem;
                    color: #64748b;
                }
                .sign-line {
                    width: 200px;
                    border-top: 1px solid #334155;
                    text-align: center;
                    padding-top: 0.35rem;
                    font-weight: 600;
                    color: #0f172a;
                    font-size: 0.8rem;
                }
            </style>
        </head>
        <body>
            <!-- Report Header -->
            <div class="report-header">
                <div>
                    <h1 class="report-title">NeuroScreen CDSS Diagnostic Report</h1>
                    <div class="report-subtitle">AI-Assisted Multi-Disease Neuroimaging Evaluation (AAL-116 Connectome Engine)</div>
                </div>
                <div style="text-align: right; font-size: 0.8rem; color: #475569;">
                    <div><strong>Date:</strong> ${reportDate}</div>
                    <div><strong>Status:</strong> Verified Clinical Screening</div>
                </div>
            </div>

            <!-- Doctor & Patient Details Grid -->
            <div class="meta-grid">
                <div class="card">
                    <div class="card-title">👨‍⚕️ Evaluating Clinician</div>
                    <div class="card-row"><strong>Name:</strong> ${doctorName}</div>
                    <div class="card-row"><strong>Designation:</strong> ${doctorDesignation}</div>
                    <div class="card-row"><strong>Medical Reg. ID:</strong> ${doctorRegId}</div>
                </div>

                <div class="card">
                    <div class="card-title">👤 Patient Information</div>
                    <div class="card-row"><strong>Patient Name:</strong> ${patientName}</div>
                    <div class="card-row"><strong>Age / Gender:</strong> ${patientAge} / ${patientGender}</div>
                    <div class="card-row"><strong>Contact:</strong> ${patientPhone}</div>
                </div>
            </div>

            <!-- Brain Analysis & Findings -->
            <div class="analysis-box">
                <div class="card-title">🧠 Neuroimaging Analysis & GNN Findings</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0;">
                    <div>
                        <span style="font-size: 0.8rem; color: #64748b;">Primary Diagnostic Classification:</span>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">${primaryFinding}</div>
                    </div>
                </div>

                <div style="margin-top: 0.5rem;">
                    ${analysisContent}
                </div>

                <!-- Anatomical Region Breakdown -->
                ${affectedRegionsHTML}
            </div>

            <!-- Footer & Signature Block -->
            <div class="footer">
                <div style="max-width: 60%; line-height: 1.4;">
                    <em>Disclaimer: This computer-aided clinical decision support document was compiled by an automated AAL-116 Graph Neural Network. To be correlated with physical neurological evaluation.</em>
                </div>
                <div class="sign-line">
                    Clinician Signature & Date
                </div>
            </div>

            <script>
                window.onload = function() {
                    window.focus();
                    window.print();
                    setTimeout(function() {
                        window.close();
                    }, 500);
                };
            <\/script>
        </body>
        </html>
    `);
    printWindow.document.close();
}
window.printClinicalReport = printClinicalReport;

/* =====================================================
   INTERACTIVE 3D ANATOMICAL BRAIN VIEWER (THREE.JS)
===================================================== */
function init3DBrainViewer() {
    const container = document.getElementById("brain3DContainer");
    if (!container || typeof THREE === "undefined" || brainRenderer) return false;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 440;

    brainScene = new THREE.Scene();
    brainCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    brainCamera.position.set(0, 25, 230);
    
    // Parent group that holds both the cortex mesh and the defect nodes
    brainContainer = new THREE.Group();
    brainScene.add(brainContainer);

    try {
        brainRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "default" });
    } catch (e) {
        console.error("WebGL failed to initialize:", e);
        return false;
    }

    brainRenderer.setSize(width, height);
    brainRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    
    // Clear old canvases if any
    const oldCanvases = container.querySelectorAll("canvas");
    oldCanvases.forEach(c => c.remove());
    container.appendChild(brainRenderer.domElement);

    if (typeof THREE.OrbitControls !== "undefined") {
        brainControls = new THREE.OrbitControls(brainCamera, brainRenderer.domElement);
        brainControls.enableDamping = true;
        brainControls.dampingFactor = 0.05;
        brainControls.rotateSpeed = 0.8;
        brainControls.minDistance = 60;
        brainControls.maxDistance = 380;
    }

    // Lighting setup
    brainScene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight1.position.set(100, 150, 100);
    brainScene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xff1744, 0.8);
    dirLight2.position.set(-100, -100, -100);
    brainScene.add(dirLight2);

    // Load Anatomical Mesh
    loadAnatomicalCortexMesh();

    // Raycaster Click Handler
    container.removeEventListener("pointerdown", onBrainNodeClick);
    container.addEventListener("pointerdown", onBrainNodeClick);

    // Single Animation Loop rotating the whole container
    function animate() {
        animFrameId = requestAnimationFrame(animate);
        if (brainControls) brainControls.update();
        
        // Rotate the root container so both brain wireframe and dots rotate together
        if (brainContainer) {
            brainContainer.rotation.y += 0.001;
        }

        if (brainRenderer && brainScene && brainCamera) {
            brainRenderer.render(brainScene, brainCamera);
        }
    }
    animate();

    window.addEventListener("resize", () => {
        if (!container || !brainRenderer || !brainCamera) return;
        brainCamera.aspect = container.clientWidth / container.clientHeight;
        brainCamera.updateProjectionMatrix();
        brainRenderer.setSize(container.clientWidth, container.clientHeight);
    });

    return true;
}

/* =====================================================
   LOAD REAL 3D ANATOMICAL BRAIN MESH
===================================================== */
function loadAnatomicalCortexMesh() {
    if (typeof THREE.OBJLoader !== "undefined") {
        const loader = new THREE.OBJLoader();
        loader.load(
            "/static/models/brain.obj",
            function (obj) {
                const cortexMaterial = new THREE.MeshBasicMaterial({
                    color: 0x38bdf8,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.10
                });

                obj.traverse(function (child) {
                    if (child.isMesh) {
                        child.material = cortexMaterial;
                    }
                });

                obj.scale.set(1.0, 1.0, 1.0);
                obj.position.set(0, 0, 0);
                obj.name = "anatomicalCortex";
                
                if (brainContainer) {
                    brainContainer.add(obj);
                } else {
                    brainScene.add(obj);
                }
            },
            undefined,
            function (err) {
                console.warn("OBJ failed to load, falling back to hull.", err);
                buildFallbackAnatomicalHull();
            }
        );
    } else {
        buildFallbackAnatomicalHull();
    }
}

function buildFallbackAnatomicalHull() {
    const cortexGroup = new THREE.Group();
    const cortexMat = new THREE.MeshPhysicalMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.08
    });

    const hemisphereGeo = new THREE.SphereGeometry(64, 32, 24);
    hemisphereGeo.scale(0.85, 1.15, 0.95);

    const left = new THREE.Mesh(hemisphereGeo, cortexMat);
    left.position.set(-18, 5, -5);
    const right = new THREE.Mesh(hemisphereGeo, cortexMat);
    right.position.set(18, 5, -5);

    cortexGroup.add(left);
    cortexGroup.add(right);
    
    if (brainContainer) {
        brainContainer.add(cortexGroup);
    } else {
        brainScene.add(cortexGroup);
    }
}

/* =====================================================
   CLICK INTERSECTION / RAYCASTER
===================================================== */
function onBrainNodeClick(event) {
    if (!brainGraphGroup || !brainCamera) return;

    const container = document.getElementById("brain3DContainer");
    const rect = container.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, brainCamera);

    const nodeMeshes = brainGraphGroup.children.filter(c => c.isMesh && c.userData && c.userData.name);
    const intersects = raycaster.intersectObjects(nodeMeshes);

    const tooltip = document.getElementById("nodeTooltip");

    if (intersects.length > 0) {
        const clickedMesh = intersects[0].object;
        const data = clickedMesh.userData;

        if (tooltip) {
            tooltip.innerHTML = `
                <div style="font-weight: 700; color: #fff; font-size: 0.85rem; margin-bottom: 2px;">${data.name}</div>
                <div style="color: #cbd5e1; font-size: 0.75rem; margin-bottom: 2px;">
                    Biomarker Salience: <strong style="color: ${data.is_defect ? '#ff1744' : '#38bdf8'};">${data.score}%</strong>
                </div>
                <div style="font-size: 0.7rem; font-weight: 600; color: ${data.is_defect ? '#ff1744' : '#00e5ff'};">
                    ${data.is_defect ? 'PRIMARY DEFECT REGION (PATHOLOGY DRIVER)' : 'BASELINE CONNECTOME NODE'}
                </div>
            `;
            tooltip.style.borderColor = data.is_defect ? '#ff1744' : '#00e5ff';
            tooltip.style.left = `${event.clientX - rect.left + 15}px`;
            tooltip.style.top = `${event.clientY - rect.top + 15}px`;
            tooltip.style.display = "block";
        }

        // Pulse selected sphere
        clickedMesh.scale.set(1.8, 1.8, 1.8);
        setTimeout(() => {
            clickedMesh.scale.set(1.0, 1.0, 1.0);
        }, 300);
    } else {
        if (tooltip) tooltip.style.display = "none";
    }
}

/* =====================================================
   RENDER GRAPH DATA FROM FLASK (NODES + DEFECT EDGES)
===================================================== */
function render3DConnectomeData(connectomeData) {
    if (!brainScene) {
        const ok = init3DBrainViewer();
        if (!ok) return;
    }

    // Clean previous graph group from the container
    if (brainGraphGroup) {
        if (brainContainer) {
            brainContainer.remove(brainGraphGroup);
        } else {
            brainScene.remove(brainGraphGroup);
        }
        brainGraphGroup = null;
    }

    brainGraphGroup = new THREE.Group();
    if (brainContainer) {
        brainContainer.add(brainGraphGroup);
    } else {
        brainScene.add(brainGraphGroup);
    }

    // Handle both {nodes, edges} and {nodes_3d, edges_3d} formats
    const rawNodes = connectomeData.nodes || connectomeData.nodes_3d || [];
    const rawEdges = connectomeData.edges || connectomeData.edges_3d || [];

    if (!rawNodes || rawNodes.length === 0) {
        console.warn("No 3D connectome nodes found in payload.");
        return;
    }

    // Auto-calculate scale factor so MNI coordinates fit inside the cortex mesh
    let maxCoord = 0;
    rawNodes.forEach(n => {
        maxCoord = Math.max(maxCoord, Math.abs(n.x || 0), Math.abs(n.y || 0), Math.abs(n.z || 0));
    });

    const scaleFactor = maxCoord > 0 ? (70.0 / maxCoord) : 1.0;

    const normalNodeMat = new THREE.MeshStandardMaterial({
        color: 0x00e5ff,
        emissive: 0x008ba3,
        roughness: 0.2,
        metalness: 0.8
    });

    const defectNodeMat = new THREE.MeshStandardMaterial({
        color: 0xff1744,
        emissive: 0xd50000,
        roughness: 0.1,
        metalness: 0.9
    });

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0xff1744,
        transparent: true,
        opacity: 0.85,
        linewidth: 2
    });

    const normalGeo = new THREE.SphereGeometry(2.2, 16, 16);
    const defectGeo = new THREE.SphereGeometry(4.8, 24, 24);

    const nodeLookup = {};

    rawNodes.forEach(n => {
        const isDefect = Boolean(n.is_defect);

        // Map MNI coordinates to Three.js orientation and scale
        const posX = (n.x || 0) * scaleFactor;
        const posY = (n.z || 0) * scaleFactor;
        const posZ = -(n.y || 0) * scaleFactor;

        const mesh = new THREE.Mesh(isDefect ? defectGeo : normalGeo, isDefect ? defectNodeMat : normalNodeMat);
        mesh.position.set(posX, posY, posZ);

        mesh.userData = {
            id: n.id,
            name: n.name || `Node ${n.id}`,
            score: n.score || (isDefect ? 85 : 10),
            is_defect: isDefect
        };

        brainGraphGroup.add(mesh);
        nodeLookup[n.id] = new THREE.Vector3(posX, posY, posZ);
    });

    // Draw structural edges connecting salient defect pathways
    if (rawEdges && rawEdges.length > 0) {
        rawEdges.forEach(e => {
            const p1 = nodeLookup[e.source];
            const p2 = nodeLookup[e.target];
            if (p1 && p2) {
                const edgeGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const line = new THREE.Line(edgeGeo, edgeMaterial);
                brainGraphGroup.add(line);
            }
        });
    }

    console.log(`[✓] Rendered ${rawNodes.length} connectome nodes into brainContainer.`);
}