/* =====================================================
   GLOBAL 3D BRAIN VIEWER VARIABLES & RAYCASTER
===================================================== */
var brainScene = null;
var brainCamera = null;
var brainRenderer = null;
var brainControls = null;
var brainGraphGroup = null;
var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var animFrameId = null;

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
        analyzeBtn.addEventListener("click", async function () {
            if (!mriInput || !mriInput.files || !mriInput.files.length) {
                alert("Please select or drop a valid brain scan first.");
                return;
            }

            const file = mriInput.files[0];
            const formData = new FormData();
            formData.append("file", file);

            // Patient intake attributes
            const nameEl = document.getElementById("patientNameInput");
            const ageEl = document.getElementById("patientAgeInput");
            const genderEl = document.getElementById("patientGenderInput");
            const phoneEl = document.getElementById("patientPhoneInput");
            const modalityEl = document.getElementById("scanModalityInput");

            if (nameEl) formData.append("patient_name", nameEl.value.trim());
            if (ageEl) formData.append("patient_age", ageEl.value.trim());
            if (genderEl) formData.append("patient_gender", genderEl.value);
            if (phoneEl) formData.append("patient_phone", phoneEl.value.trim());
            if (modalityEl) formData.append("scan_modality", modalityEl.value);

            // Loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = `<span>Extracting Features & Executing GNN...</span> <span>⏳</span>`;

            if (resultSection) {
                resultSection.classList.add("show");
                // Smoothly scroll to results
                requestAnimationFrame(() => {
                    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
                });
            }
            // Force Three.js canvas to adapt to the visible container width
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
                        throw new Error("File exceeds upload limit (500MB). Please compress the scan.");
                    }
                    throw new Error(`Server returned HTTP ${response.status}: ${rawText.slice(0, 100)}`);
                }

                if (!response.ok || data.error) {
                    throw new Error(data.error || "Server processing failed.");
                }

                // Render Success State
                if (resultStatus) {
                    resultStatus.textContent = "DIAGNOSTIC COMPLETE";
                    resultStatus.className = `badge badge-${data.badge || "success"}`;
                }

                if (resultTitle) {
                    resultTitle.innerHTML = `Identified: <span style="color: #38bdf8;">${data.prediction}</span>`;
                }

                // Metadata Header (supports both keys)
                let metaHtml = "";
                if (data.subject_meta) {
                    const patientName = data.subject_meta.name || data.subject_meta.patient_name || data.subject_meta.patient_id || "Anonymous";
                    const patientAge = data.subject_meta.age || data.subject_meta.patient_age || "N/A";
                    const patientGender = data.subject_meta.gender || data.subject_meta.patient_gender || "N/A";
                    const patientPhone = data.subject_meta.phone || data.subject_meta.patient_phone || "N/A";

                    metaHtml = `
                        <div style="background: rgba(15, 23, 42, 0.7); padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.85rem; border-left: 3px solid #38bdf8; text-align: left;">
                            <div><strong>Patient:</strong> ${patientName} | <strong>Profile:</strong> ${patientAge} (${patientGender})</div>
                            <div><strong>Contact:</strong> ${patientPhone} | <strong>Modality:</strong> ${data.subject_meta.modality || "MRI"}</div>
                            <div><strong>Clinician:</strong> ${data.subject_meta.evaluating_clinician || "Staff"} (${data.subject_meta.license || "Verified"})</div>
                        </div>
                    `;
                }

                let breakdownHTML = `
                    ${metaHtml}
                    <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem; text-align: left;">
                        Model Confidence: ${Number(data.confidence).toFixed(2)}%
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 1rem; text-align: left;">
                        ${data.description || ""}
                    </p>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem; text-align: left;">
                `;

                if (data.probabilities) {
                    for (const [disease, prob] of Object.entries(data.probabilities)) {
                        const numProb = Number(prob).toFixed(2);
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
                    <button onclick="window.print()" style="margin-top: 1.25rem; padding: 0.5rem 1rem; background: #0284c7; border: none; color: #fff; font-weight: 600; border-radius: 6px; cursor: pointer; font-size: 0.85rem;">
                        🖨️ Export / Print Diagnostic Summary
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

    // Single Animation Loop
    function animate() {
        animFrameId = requestAnimationFrame(animate);
        if (brainControls) brainControls.update();
        if (brainGraphGroup) {
            brainGraphGroup.rotation.y += 0.001;
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
                // Use MeshBasicMaterial: fast, lightweight, and will NOT crash the GPU
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
                brainScene.add(obj);
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
    brainScene.add(cortexGroup);
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
   RENDER GRAPH DATA FROM FLASK
===================================================== */
function render3DConnectomeData(connectomeData) {
    if (!brainScene) {
        const ok = init3DBrainViewer();
        if (!ok) return;
    }

    if (brainGraphGroup && brainScene) {
        brainScene.remove(brainGraphGroup);
        brainGraphGroup = null;
    }

    brainGraphGroup = new THREE.Group();
    const { nodes, edges } = connectomeData;
    const nodeLookup = {};

    const normalNodeMat = new THREE.MeshStandardMaterial({
        color: 0x00e5ff,
        emissive: 0x00404d,
        roughness: 0.3,
        metalness: 0.5
    });

    const defectNodeMat = new THREE.MeshStandardMaterial({
        color: 0xff1744,
        emissive: 0x880016,
        roughness: 0.2,
        metalness: 0.7
    });

    const normalGeo = new THREE.SphereGeometry(1.6, 16, 16);
    const defectGeo = new THREE.SphereGeometry(3.8, 24, 24);

    nodes.forEach(n => {
        // MNI to Three.js coordinates
        const posX = n.x;
        const posY = n.z;
        const posZ = -n.y;

        const mesh = new THREE.Mesh(n.is_defect ? defectGeo : normalGeo, n.is_defect ? defectNodeMat : normalNodeMat);
        mesh.position.set(posX, posY, posZ);

        mesh.userData = {
            id: n.id,
            name: n.name,
            score: n.score,
            is_defect: n.is_defect
        };

        brainGraphGroup.add(mesh);
        nodeLookup[n.id] = new THREE.Vector3(posX, posY, posZ);
    });

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: 0xff1744,
        transparent: true,
        opacity: 0.85
    });

    if (edges && edges.length > 0) {
        edges.forEach(e => {
            const p1 = nodeLookup[e.source];
            const p2 = nodeLookup[e.target];
            if (p1 && p2) {
                const edgeGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const line = new THREE.Line(edgeGeo, edgeMaterial);
                brainGraphGroup.add(line);
            }
        });
    }

    brainScene.add(brainGraphGroup);
}