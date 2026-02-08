const { app } = window.comfyAPI.app;

/**
 * Alert Collector - Intercepts 3D preview alerts and collects them in a minimal UI.
 * Shows a small "X alerts" badge that expands to show details.
 */

const SETTING_ID_3D = "Alvatar.Collect3DAlerts";
const SETTING_ID_ALL = "Alvatar.CollectAllAlerts";

app.registerExtension({
    name: "Alvatar.AlertCollector",

    async setup() {
        // Register settings - match KJNodes pattern exactly (no category field)
        app.ui.settings.addSetting({
            id: SETTING_ID_3D,
            name: "Alvatar: Collect 3D preview alerts in badge",
            type: "boolean",
            defaultValue: true,
        });

        app.ui.settings.addSetting({
            id: SETTING_ID_ALL,
            name: "Alvatar: Collect ALL notifications in badge",
            type: "boolean",
            defaultValue: false,
        });

        // Initialize
        this.alerts = [];
        this.ui = null;

        // Create UI
        this.createUI();

        // Hook toast store (needs to wait for Vue app to be ready)
        this.waitForVueAndHook();

        console.log("[Alvatar.AlertCollector] Extension loaded");
    },

    waitForVueAndHook() {
        const tryHook = () => {
            const vueApp = document.querySelector('#vue-app')?.__vue_app__ ||
                           document.querySelector('#app')?.__vue_app__ ||
                           document.querySelector('[data-v-app]')?.__vue_app__;

            const pinia = vueApp?.config?.globalProperties?.$pinia;
            const toastStore = pinia?._s?.get('toast');

            if (toastStore && !toastStore._alvatarCollectorHooked) {
                this.hookToastStore(toastStore);
                // Clear any alerts that slipped through before hook
                this.clearExistingToasts(toastStore);
            } else if (!toastStore) {
                // Retry quickly - we need to hook ASAP
                setTimeout(tryHook, 50);
            }
        };

        // Start immediately, no delay
        tryHook();
    },

    clearExistingToasts(toastStore) {
        // Clear any toasts that appeared before we hooked
        setTimeout(() => {
            toastStore.removeAll();
            // Also clear via PrimeVue if available
            const vueApp = document.querySelector('#vue-app')?.__vue_app__ ||
                           document.querySelector('#app')?.__vue_app__ ||
                           document.querySelector('[data-v-app]')?.__vue_app__;
            const primeToast = vueApp?.config?.globalProperties?.$toast;
            if (primeToast) {
                primeToast.removeAllGroups();
            }
            console.log("[AlertCollector] Cleared pre-hook toasts");
        }, 100);
    },

    hookToastStore(toastStore) {
        const self = this;
        const originalAddAlert = toastStore.addAlert.bind(toastStore);
        const originalAdd = toastStore.add.bind(toastStore);

        toastStore.addAlert = function(message) {
            const collect3D = app.ui.settings.getSettingValue(SETTING_ID_3D, true);
            const collectAll = app.ui.settings.getSettingValue(SETTING_ID_ALL, false);

            // Check if this is a 3D-related error
            const msg = (message || '').toLowerCase();
            const is3DError = msg.includes('error loading model') ||
                              msg.includes('model not found') ||
                              msg.includes('could not load') ||
                              msg.includes('failed to load');

            // Collect based on settings
            if (collectAll || (collect3D && is3DError)) {
                self.addAlert(message, 'alert');
                console.log("[AlertCollector] Collected:", message);
                return null; // Suppress the popup
            }

            return originalAddAlert(message);
        };

        toastStore.add = function(options) {
            const collect3D = app.ui.settings.getSettingValue(SETTING_ID_3D, true);
            const collectAll = app.ui.settings.getSettingValue(SETTING_ID_ALL, false);

            // Check severity and message
            const severity = options?.severity || 'info';
            const detail = (options?.detail || options?.summary || '').toLowerCase();
            const is3DError = detail.includes('error loading model') ||
                              detail.includes('model not found') ||
                              detail.includes('could not load') ||
                              detail.includes('failed to load');
            const isError = severity === 'error' || severity === 'warn';

            // Collect based on settings
            if (collectAll || (collect3D && isError && is3DError)) {
                const message = options?.detail || options?.summary || 'Unknown notification';
                self.addAlert(message, severity);
                console.log("[AlertCollector] Collected:", options);
                return null;
            }

            return originalAdd(options);
        };

        toastStore._alvatarCollectorHooked = true;
        console.log("[AlertCollector] Toast store hooked successfully");
    },

    addAlert(message, severity = 'alert') {
        this.alerts.push({
            message: message,
            severity: severity,
            timestamp: new Date()
        });
        this.updateUI();
    },

    clearAlerts() {
        this.alerts = [];
        this.updateUI();
    },

    createUI() {
        // Container
        const container = document.createElement('div');
        container.id = 'alvatar-alert-collector';
        container.innerHTML = `
            <style>
                #alvatar-alert-collector {
                    position: fixed;
                    top: 115px;
                    right: 20px;
                    z-index: 10000;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    font-size: 13px;
                }

                #alvatar-alert-badge {
                    background: #3a3a3a;
                    color: #f0a030;
                    padding: 6px 12px;
                    border-radius: 16px;
                    cursor: pointer;
                    display: none;
                    align-items: center;
                    gap: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    border: 1px solid #555;
                    transition: all 0.2s ease;
                }

                #alvatar-alert-badge:hover {
                    background: #454545;
                    transform: scale(1.02);
                }

                #alvatar-alert-badge.has-alerts {
                    display: flex;
                }

                #alvatar-alert-badge .icon {
                    font-size: 14px;
                }

                #alvatar-alert-panel {
                    display: none;
                    background: #2a2a2a;
                    border: 1px solid #555;
                    border-radius: 8px;
                    width: 350px;
                    max-height: 300px;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
                    overflow: hidden;
                }

                #alvatar-alert-panel.expanded {
                    display: block;
                }

                #alvatar-alert-header {
                    background: #3a3a3a;
                    padding: 10px 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 1px solid #555;
                }

                #alvatar-alert-header .title {
                    color: #f0a030;
                    font-weight: 500;
                }

                #alvatar-alert-header .actions {
                    display: flex;
                    gap: 8px;
                }

                #alvatar-alert-header button {
                    background: #555;
                    border: none;
                    color: #ddd;
                    padding: 4px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                }

                #alvatar-alert-header button:hover {
                    background: #666;
                }

                #alvatar-alert-header button.clear {
                    background: #a04040;
                }

                #alvatar-alert-header button.clear:hover {
                    background: #b05050;
                }

                #alvatar-alert-list {
                    max-height: 240px;
                    overflow-y: auto;
                    padding: 8px;
                }

                .alvatar-alert-item {
                    background: #353535;
                    padding: 8px 10px;
                    margin-bottom: 6px;
                    border-radius: 4px;
                    border-left: 3px solid #f0a030;
                }

                .alvatar-alert-item:last-child {
                    margin-bottom: 0;
                }

                .alvatar-alert-item.severity-error {
                    border-left-color: #e05050;
                }

                .alvatar-alert-item.severity-warn {
                    border-left-color: #f0a030;
                }

                .alvatar-alert-item.severity-info {
                    border-left-color: #5090e0;
                }

                .alvatar-alert-item.severity-success {
                    border-left-color: #50b050;
                }

                .alvatar-alert-item .message {
                    color: #ddd;
                    word-break: break-word;
                }

                .alvatar-alert-item .time {
                    color: #888;
                    font-size: 11px;
                    margin-top: 4px;
                }

                .alvatar-alert-item .severity-tag {
                    font-size: 10px;
                    text-transform: uppercase;
                    opacity: 0.7;
                    margin-left: 8px;
                }
            </style>

            <div id="alvatar-alert-badge">
                <span class="icon">⚠️</span>
                <span class="count">0 alerts</span>
            </div>

            <div id="alvatar-alert-panel">
                <div id="alvatar-alert-header">
                    <span class="title">⚠️ Collected Alerts</span>
                    <div class="actions">
                        <button class="clear">Clear All</button>
                        <button class="close">✕</button>
                    </div>
                </div>
                <div id="alvatar-alert-list"></div>
            </div>
        `;

        document.body.appendChild(container);

        // Store references
        this.ui = {
            container,
            badge: container.querySelector('#alvatar-alert-badge'),
            panel: container.querySelector('#alvatar-alert-panel'),
            list: container.querySelector('#alvatar-alert-list'),
            count: container.querySelector('.count')
        };

        // Event listeners
        this.ui.badge.addEventListener('click', () => this.togglePanel());
        container.querySelector('.close').addEventListener('click', () => this.togglePanel(false));
        container.querySelector('.clear').addEventListener('click', () => this.clearAlerts());

        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            if (!container.contains(e.target)) {
                this.togglePanel(false);
            }
        });
    },

    togglePanel(show) {
        if (show === undefined) {
            show = !this.ui.panel.classList.contains('expanded');
        }

        if (show) {
            this.ui.panel.classList.add('expanded');
            this.ui.badge.style.display = 'none';
        } else {
            this.ui.panel.classList.remove('expanded');
            if (this.alerts.length > 0) {
                this.ui.badge.style.display = 'flex';
            }
        }
    },

    updateUI() {
        const count = this.alerts.length;

        // Update badge
        this.ui.count.textContent = `${count} alert${count !== 1 ? 's' : ''}`;

        if (count > 0) {
            this.ui.badge.classList.add('has-alerts');
        } else {
            this.ui.badge.classList.remove('has-alerts');
            this.togglePanel(false);
        }

        // Update list
        this.ui.list.innerHTML = this.alerts.map((alert, i) => `
            <div class="alvatar-alert-item severity-${alert.severity || 'warn'}">
                <div class="message">${this.escapeHtml(alert.message)}</div>
                <div class="time">${alert.timestamp.toLocaleTimeString()}<span class="severity-tag">${alert.severity || 'alert'}</span></div>
            </div>
        `).join('');
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
