const { app } = window.comfyAPI.app;

/**
 * Dynamic inputs/outputs for Continue node.
 *
 * Behavior:
 * - Starts with 2 inputs + 2 outputs.
 * - When highest connected input is N, keep inputs up to N+1 (show next slot).
 * - Outputs track connected range only (2..N) to avoid trailing empty passthrough outputs.
 */
app.registerExtension({
    name: "Alvatar.ContinueDynamicSlots",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Continue") {
            return;
        }

        // Ensure base schema in UI (backend handles dynamic validation/execution).
        nodeData.input = ["*", "*"];
        nodeData.input_name = ["input1", "input2"];
        nodeData.output = ["*", "*"];
        nodeData.output_name = ["output1", "output2"];

        const syncSlots = function() {
            if (!this.inputs || !this.outputs) {
                return;
            }

            // Highest connected input slot index (0-based).
            let lastConnected = -1;
            for (let i = 0; i < this.inputs.length; i++) {
                if (this.inputs[i]?.link != null) {
                    lastConnected = i;
                }
            }

            // Keep one trailing input slot beyond the highest connected slot.
            const desiredInputs = Math.max(2, lastConnected + 2);
            const inputType = this.inputs[0]?.type || "*";

            while (this.inputs.length < desiredInputs) {
                this.addInput(`input${this.inputs.length + 1}`, inputType);
            }
            while (this.inputs.length > desiredInputs) {
                this.removeInput(this.inputs.length - 1);
            }
            for (let i = 0; i < this.inputs.length; i++) {
                this.inputs[i].name = `input${i + 1}`;
            }

            // Outputs mirror connected input range (minimum 2).
            const desiredOutputs = Math.max(2, lastConnected + 1);
            const outputType = this.outputs[0]?.type || "*";

            while (this.outputs.length < desiredOutputs) {
                this.addOutput(`output${this.outputs.length + 1}`, outputType);
            }
            while (this.outputs.length > desiredOutputs) {
                this.removeOutput(this.outputs.length - 1);
            }
            for (let i = 0; i < this.outputs.length; i++) {
                this.outputs[i].name = `output${i + 1}`;
            }
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            syncSlots.call(this);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (origOnConfigure) {
                origOnConfigure.apply(this, arguments);
            }
            // Run after links/widgets are restored.
            setTimeout(() => {
                try {
                    syncSlots.call(this);
                } catch (e) {
                    console.warn("[Alvatar.ContinueDynamicSlots] sync on configure failed", e);
                }
            }, 0);
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }

            // type: 1=input, 2=output. We only care about input socket changes.
            if (type !== 1) {
                return;
            }

            try {
                syncSlots.call(this);
            } catch (e) {
                console.warn("[Alvatar.ContinueDynamicSlots] sync on connection change failed", e);
            }
        };
    },
});
