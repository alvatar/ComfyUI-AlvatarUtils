const { app } = window.comfyAPI.app;

// Extension to display debug text in DebugAny node
app.registerExtension({
    name: "Alvatar.DebugAny",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DebugAny") {
            // Store original onExecuted
            const origOnExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                // Call original if exists
                if (origOnExecuted) {
                    origOnExecuted.apply(this, arguments);
                }

                // Get text from message
                if (message && message.text) {
                    const text = Array.isArray(message.text) ? message.text.join("\n") : message.text;

                    // Find or create text widget
                    let textWidget = this.widgets?.find(w => w.name === "debug_output");

                    if (!textWidget) {
                        // Create a text display widget
                        textWidget = this.addWidget("text", "debug_output", "", () => {}, {
                            multiline: true,
                            serialize: false,
                        });
                        textWidget.inputEl = document.createElement("textarea");
                        textWidget.inputEl.readOnly = true;
                        textWidget.inputEl.style.cssText = `
                            width: 100%;
                            height: 120px;
                            background: #1a1a1a;
                            color: #00ff00;
                            border: 1px solid #333;
                            border-radius: 4px;
                            font-family: monospace;
                            font-size: 11px;
                            padding: 8px;
                            resize: none;
                            overflow: auto;
                        `;
                        textWidget.computeSize = () => [this.size[0] - 20, 130];
                        textWidget.draw = function(ctx, node, width, y) {
                            // Position the textarea
                            const margin = 10;
                            this.inputEl.style.width = (width - margin * 2) + "px";
                        };
                    }

                    // Update text content
                    textWidget.value = text;
                    if (textWidget.inputEl) {
                        textWidget.inputEl.value = text;
                    }

                    // Resize node to fit content
                    this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], 200)]);
                    this.setDirtyCanvas(true);
                }
            };

            // Override onNodeCreated to setup initial widget
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }

                // Set minimum size
                this.setSize([300, 180]);
            };
        }
    }
});
