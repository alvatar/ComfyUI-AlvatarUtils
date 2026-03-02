const { app } = window.comfyAPI.app;

const TARGET_NODE_NAMES = new Set(["LoadGLBToTrimesh", "LoadGLBFromPath"]);
const ALLOWED_EXTENSIONS = new Set([".glb", ".gltf", ".obj", ".ply", ".stl"]);

function getExtension(filename) {
    const idx = filename.lastIndexOf(".");
    if (idx < 0) return "";
    return filename.slice(idx).toLowerCase();
}

function normalizeRelPath(path) {
    return String(path || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

function parseUploadResponse(data, fallbackName) {
    if (!data || typeof data !== "object") {
        return normalizeRelPath(fallbackName);
    }

    if (typeof data.path === "string" && data.path.length > 0) {
        return normalizeRelPath(data.path);
    }
    if (typeof data.filepath === "string" && data.filepath.length > 0) {
        return normalizeRelPath(data.filepath);
    }

    const name = data.name || data.filename || fallbackName;
    const subfolder = data.subfolder || "";

    if (subfolder) {
        return normalizeRelPath(`${subfolder}/${name}`);
    }
    return normalizeRelPath(name);
}

async function postForm(path, formData) {
    const api = window.comfyAPI?.api;
    if (api && typeof api.fetchApi === "function") {
        return api.fetchApi(path, {
            method: "POST",
            body: formData,
        });
    }

    return fetch(path, {
        method: "POST",
        body: formData,
        credentials: "same-origin",
    });
}

async function uploadMeshFile(file) {
    const candidates = [
        { path: "/upload/file", field: "file" },
        { path: "/upload/image", field: "image" },
        // Fallbacks for deployments where routes are mounted under /api.
        { path: "/api/upload/file", field: "file" },
        { path: "/api/upload/image", field: "image" },
    ];

    const errors = [];

    for (const candidate of candidates) {
        const fd = new FormData();
        fd.append(candidate.field, file, file.name);
        fd.append("type", "input");
        fd.append("subfolder", "");
        fd.append("overwrite", "true");

        try {
            const response = await postForm(candidate.path, fd);
            let data = null;
            let bodyText = "";

            try {
                data = await response.json();
            } catch (_jsonErr) {
                try {
                    bodyText = await response.text();
                } catch (_textErr) {
                    bodyText = "";
                }
            }

            if (!response.ok) {
                const detail = (data && (data.error || data.message)) || bodyText || `HTTP ${response.status}`;
                errors.push(`${candidate.path}: ${detail}`);
                continue;
            }

            return parseUploadResponse(data, file.name);
        } catch (err) {
            errors.push(`${candidate.path}: ${err?.message || String(err)}`);
        }
    }

    throw new Error(`Mesh upload failed. Tried endpoints:\n- ${errors.join("\n- ")}`);
}

function pickMeshFile() {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".glb,.gltf,.obj,.ply,.stl";
        input.style.display = "none";

        input.onchange = () => {
            const file = input.files && input.files.length ? input.files[0] : null;
            input.remove();
            if (!file) {
                reject(new Error("No file selected"));
                return;
            }
            resolve(file);
        };

        document.body.appendChild(input);
        input.click();
    });
}

function setWidgetValue(node, widgetName, value) {
    const widget = node.widgets?.find(w => w.name === widgetName);
    if (!widget) return false;

    if (widget.options && Array.isArray(widget.options.values)) {
        if (!widget.options.values.includes(value)) {
            widget.options.values.push(value);
            widget.options.values.sort((a, b) => String(a).localeCompare(String(b)));
        }
    }

    widget.value = value;

    if (widget.inputEl) {
        widget.inputEl.value = value;
    }

    if (typeof widget.callback === "function") {
        try {
            widget.callback(value, app.canvas, node);
        } catch (e) {
            console.warn("[Alvatar.MeshUpload] widget callback failed", e);
        }
    }

    return true;
}

function ensureUploadWidget(node, nodeDataName) {
    if (node.widgets?.some(w => w.name === "upload_mesh")) {
        return;
    }

    const button = node.addWidget("button", "upload_mesh", "Upload mesh", async () => {
        const oldLabel = button.value;

        try {
            const file = await pickMeshFile();
            const ext = getExtension(file.name);
            if (!ALLOWED_EXTENSIONS.has(ext)) {
                throw new Error(`Unsupported extension '${ext}'. Allowed: ${Array.from(ALLOWED_EXTENSIONS).join(", ")}`);
            }

            button.value = "Uploading...";
            node.setDirtyCanvas(true, true);

            const relPath = await uploadMeshFile(file);
            console.log(`[Alvatar.MeshUpload] Uploaded ${file.name} -> ${relPath}`);

            if (nodeDataName === "LoadGLBToTrimesh") {
                setWidgetValue(node, "mesh_file", relPath);
            } else if (nodeDataName === "LoadGLBFromPath") {
                setWidgetValue(node, "file_path", relPath);
                setWidgetValue(node, "mesh_file", relPath);
            }
        } catch (err) {
            console.error("[Alvatar.MeshUpload]", err);
            window.alert(`Mesh upload failed:\n${err?.message || String(err)}`);
        } finally {
            button.value = oldLabel;
            node.setDirtyCanvas(true, true);
        }
    });

    button.serialize = false;
}

app.registerExtension({
    name: "Alvatar.MeshUpload",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            ensureUploadWidget(this, nodeData.name);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (origOnConfigure) {
                origOnConfigure.apply(this, arguments);
            }
            // Ensure widget exists on workflow restore.
            setTimeout(() => {
                try {
                    ensureUploadWidget(this, nodeData.name);
                } catch (e) {
                    console.warn("[Alvatar.MeshUpload] restore failed", e);
                }
            }, 0);
        };
    },
});
