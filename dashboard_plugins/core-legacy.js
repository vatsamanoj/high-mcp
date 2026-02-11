const API_BASE = 'http://localhost:8004/api';
        let autofixConfig = {};
        let pendingAttachments = []; // Stores {type, mime_type, data, name, content, previewUrl}
        let lastModels = [];
        let currentSort = { column: 'tier', dir: 'asc' };
        let currentPatches = null;
        let currentPlan = null; // Store the latest generated plan

        // --- Navigation ---
        function switchView(viewName) {
            // Update Nav Buttons
            document.querySelectorAll('nav button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active'); 

            // Update Views
            document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
            document.getElementById(`view-${viewName}`).classList.add('active');
            
            if (viewName === 'chat') {
                fetchChatModels();
            } else if (viewName === 'templates') {
                fetchTemplates();
            } else if (viewName === 'coder') {
                fetchCoderModels();
            } else if (viewName === 'plugins') {
                fetchComponents();
            }
        }

        // --- Data Fetching ---
        async function fetchComponents() {
            try {
                const res = await fetch(`${API_BASE}/components`);
                const data = await res.json();
                const tbody = document.querySelector('#plugins-table tbody');
                tbody.innerHTML = '';
                data.components.forEach(c => {
                    const tr = document.createElement('tr');
                    // Highlight if it's a plugin (in plugins/ dir)
                    const isPlugin = c.file_path.includes('plugins');
                    const nameHtml = isPlugin ? `<span style="color:var(--accent-color)">${c.name}</span> <span class="badge badge-std">Plugin</span>` : c.name;
                    
                    tr.innerHTML = `
                        <td>${nameHtml}</td>
                        <td style="font-size:12px; color:var(--text-secondary)">${c.file_path}</td>
                        <td>${c.routes}</td>
                        <td>${c.services.join(', ')}</td>
                        <td>${c.mtime ? new Date(c.mtime * 1000).toLocaleString() : '-'}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (e) { console.error("Fetch components error", e); }
        }

        async function fetchStatus() {
            try {
                const response = await fetch(`${API_BASE}/status`);
                const data = await response.json();
                lastModels = data.models;
                updateDashboard(data);
                updateModelsTable(data.models);
            } catch (e) { console.error("Fetch status error", e); }
        }

        function updateDashboard(data) {
            // Speed Toggle
            const toggle = document.getElementById('speed-toggle');
            const label = document.getElementById('speed-label');
            toggle.checked = data.speed_override;
            label.textContent = data.speed_override ? "🚀 SPEED PRIORITY" : "Standard Mode";
            label.style.color = data.speed_override ? "var(--accent-color)" : "var(--text-primary)";

            // Stats
            const active = data.models.filter(m => m.available).length;
            const blocked = data.models.length - active;
            document.getElementById('stat-active-models').textContent = active;
            document.getElementById('stat-blocked-models').textContent = blocked;
        }

        function sortTable(column) {
            if (currentSort.column === column) {
                currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
            } else {
                currentSort.column = column;
                currentSort.dir = 'asc';
            }
            updateModelsTable(lastModels);
        }

        function updateModelsTable(models) {
            const showActiveOnly = document.getElementById('show-active-only')?.checked;
            const tbody = document.querySelector('#models-table tbody');
            tbody.innerHTML = '';
            
            // Sort
            models.sort((a, b) => {
                let valA = a[currentSort.column];
                let valB = b[currentSort.column];

                if (currentSort.column === 'tier') {
                    const tierOrder = { "free": 1, "economical": 2, "standard": 3, "premium": 4 };
                    valA = tierOrder[valA] || 3;
                    valB = tierOrder[valB] || 3;
                } else if (currentSort.column === 'providers') {
                     valA = (valA || []).join(', ').toLowerCase();
                     valB = (valB || []).join(', ').toLowerCase();
                } else if (currentSort.column === 'status') {
                     valA = a.available ? 1 : 0;
                     valB = b.available ? 1 : 0;
                } else if (typeof valA === 'string') {
                    valA = valA.toLowerCase();
                    valB = valB.toLowerCase();
                }

                if (valA < valB) return currentSort.dir === 'asc' ? -1 : 1;
                if (valA > valB) return currentSort.dir === 'asc' ? 1 : -1;
                return 0;
            });

            models.forEach(model => {
                if (showActiveOnly && !model.available) return;

                const tr = document.createElement('tr');
                const isAvail = model.available;
                const providersStr = (model.providers || []).join(', ');
                
                tr.innerHTML = `
                    <td style="font-weight:500; display:flex; align-items:center;">
                        <span class="status-dot ${isAvail ? 'status-ok' : 'status-blocked'}"></span>
                        ${model.model}
                    </td>
                    <td><span class="badge badge-${(model.tier || 'std').toLowerCase().substring(0,4)}">${model.tier}</span></td>
                    <td>${model.category}</td>
                    <td>${providersStr}</td>
                    <td style="color:${isAvail ? 'var(--success-color)' : 'var(--danger-color)'}">${isAvail ? 'Active' : 'Blocked'}</td>
                    <td>${model.rpm_used.toFixed(1)} / ${formatLimit(model.rpm_limit)}</td>
                    <td>${model.rpd_used.toFixed(1)} / ${formatLimit(model.rpd_limit)}</td>
                `;
                tbody.appendChild(tr);
            });
            
            // Update header arrows
            document.querySelectorAll('#models-table th').forEach(th => {
                const col = th.getAttribute('data-col');
                if (col) {
                    th.textContent = th.textContent.replace(/ [↑↓↕]/, '');
                    if (col === currentSort.column) {
                        th.textContent += currentSort.dir === 'asc' ? ' ↑' : ' ↓';
                    } else {
                        th.textContent += ' ↕';
                    }
                }
            });
        }

        // --- Speed Toggle ---
        async function toggleSpeed() {
            const toggle = document.getElementById('speed-toggle');
            try {
                await fetch(`${API_BASE}/toggle_speed`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: toggle.checked })
                });
                fetchStatus();
            } catch (e) { console.error("Toggle speed error", e); toggle.checked = !toggle.checked; }
        }

        // --- Template Functions ---
        async function fetchTemplates() {
            try {
                const response = await fetch(`${API_BASE}/templates`);
                const data = await response.json();
                const list = document.getElementById('templates-list');
                list.innerHTML = '';

                if (!data.templates || data.templates.length === 0) {
                    list.innerHTML = '<p style="text-align:center; color:var(--text-secondary)">No templates found.</p>';
                    return;
                }

                data.templates.forEach(t => {
                    const item = document.createElement('div');
                    item.className = 'log-item';
                    item.innerHTML = `
                        <div class="log-header">
                            <span style="font-weight:600; color:var(--accent-color);">${t.description || 'No Description'}</span>
                            <span style="font-size:11px; color:var(--text-secondary);">ID: ${t.id}</span>
                        </div>
                        <div style="font-family:monospace; background:rgba(0,0,0,0.3); padding:8px; border-radius:4px; margin-bottom:5px; font-size:12px; overflow-x:auto;">
                            <div style="color:var(--warning-color); margin-bottom:2px;">Pattern: ${t.pattern.replace(/</g, '&lt;')}</div>
                            <div style="color:var(--success-color);">Format: ${t.format.substring(0, 100).replace(/</g, '&lt;')}${t.format.length > 100 ? '...' : ''}</div>
                        </div>
                    `;
                    list.appendChild(item);
                });
            } catch (e) { console.error("Fetch templates error", e); }
        }

        async function saveTemplate() {
            const pattern = document.getElementById('tmpl-pattern').value;
            const format = document.getElementById('tmpl-format').value;
            const desc = document.getElementById('tmpl-desc').value;

            if (!pattern || !format) {
                alert("Pattern and Format are required.");
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/templates`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pattern: pattern, format_str: format, description: desc })
                });
                
                if (response.ok) {
                    alert("Template saved!");
                    document.getElementById('tmpl-pattern').value = '';
                    document.getElementById('tmpl-format').value = '';
                    document.getElementById('tmpl-desc').value = '';
                    fetchTemplates();
                } else {
                    alert("Error saving template");
                }
            } catch (e) { console.error("Save template error", e); alert("Error saving template"); }
        }

        // --- Chat Functions ---
        async function fetchChatModels() {
            const select = document.getElementById('chat-model-select');
            if (select.options.length > 1) return; // Already loaded

            try {
                const response = await fetch(`${API_BASE}/chat/models`);
                const data = await response.json();
                
                select.innerHTML = '';
                const models = data.models.filter(m => m.available).sort((a,b) => a.model.localeCompare(b.model));
                
                if (models.length === 0) {
                    const opt = document.createElement('option');
                    opt.text = "No active models";
                    select.add(opt);
                    return;
                }

                models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.model;
                    opt.text = `${m.model} (${m.category || 'General'})`;
                    select.add(opt);
                });
            } catch (e) { console.error("Fetch chat models error", e); }
        }

        function handleChatUpload(input) {
            const files = input.files;
            if (!files.length) return;

            Array.from(files).forEach(file => {
                // 1. Images & PDFs (Binary for backend)
                // Note: For Gemini, PDF is treated as inline_data like images
                if (file.type.startsWith('image/') || file.type === 'application/pdf') {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64Data = e.target.result.split(',')[1];
                        pendingAttachments.push({ 
                            type: file.type === 'application/pdf' ? 'pdf' : 'image',
                            mime_type: file.type, 
                            data: base64Data,
                            name: file.name,
                            previewUrl: file.type === 'application/pdf' ? null : e.target.result
                        });
                        renderAttachments();
                    };
                    reader.readAsDataURL(file);
                } 
                // 2. Everything else -> Treat as Text
                else {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        pendingAttachments.push({
                            type: 'text',
                            name: file.name,
                            content: e.target.result,
                            mime_type: 'text/plain' // Backend sees this as text content
                        });
                        renderAttachments();
                    };
                    // Try to read as text. If it fails (binary), we might need a fallback, 
                    // but for "code files" and "txt", this works.
                    // If the user uploads a binary file (exe, zip), this might produce garbage,
                    // but we will attempt to send it as text content for analysis.
                    reader.readAsText(file);
                }
            });
            input.value = ''; // Reset
        }

        function renderAttachments() {
            const container = document.getElementById('image-preview');
            container.innerHTML = '';
            
            pendingAttachments.forEach((att, idx) => {
                const div = document.createElement('div');
                div.className = 'preview-item';
                
                if (att.type === 'image') {
                    div.style.backgroundImage = `url(${att.previewUrl})`;
                } else {
                    // Text or PDF icon
                    div.style.display = 'flex';
                    div.style.alignItems = 'center';
                    div.style.justifyContent = 'center';
                    div.style.backgroundColor = 'rgba(255,255,255,0.1)';
                    div.style.flexDirection = 'column';
                    div.style.fontSize = '10px';
                    div.style.overflow = 'hidden';
                    
                    let icon = '📄';
                    if (att.type === 'text') icon = '📝';
                    if (att.type === 'pdf') icon = '📕';
                    
                    div.innerHTML = `<div style="font-size:20px">${icon}</div><div style="width:100%;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 2px;">${att.name}</div>`;
                }

                const removeBtn = document.createElement('div');
                removeBtn.className = 'preview-remove';
                removeBtn.innerHTML = '×';
                removeBtn.onclick = () => {
                    pendingAttachments.splice(idx, 1);
                    renderAttachments();
                };
                
                div.appendChild(removeBtn);
                container.appendChild(div);
            });
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            let message = input.value.trim();
            const modelSelect = document.getElementById('chat-model-select');
            const model = modelSelect.value;
            
            // Allow empty message to check cache/history
            // if (!message && pendingAttachments.length === 0) return;
            
            if (!model) { alert("Please select a model first."); return; }

            // Process Attachments
            const textAttachments = pendingAttachments.filter(a => a.type === 'text');
            const binaryAttachments = pendingAttachments.filter(a => a.type === 'image' || a.type === 'pdf');

            // Append text files to message
            if (textAttachments.length > 0) {
                const fileContext = textAttachments.map(f => `\n--- File: ${f.name} ---\n${f.content}\n--- End of File ---`).join('\n');
                if (message) message += "\n";
                message += fileContext;
            }

            // Prepare images/media for API
            const apiImages = binaryAttachments.map(a => ({
                mime_type: a.mime_type,
                data: a.data
            }));

            // 1. Render User Message
            addMessageToChat('user', input.value.trim(), [...pendingAttachments]); 

            // Clear Input
            input.value = '';
            pendingAttachments = [];
            renderAttachments();

            // 2. Render Loading
            const loadingHtml = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
            const loadingId = addMessageToChat('ai', loadingHtml);

            // 3. Send API Request
            try {
                const loadingMsg = document.querySelector(`[data-id="${loadingId}"] .message-content`);
                
                // Add status updates for long waits
                const longWaitTimer = setTimeout(() => {
                    if (loadingMsg) loadingMsg.innerHTML += '<br><br><small style="color:var(--text-secondary)">Still working... performing deep reasoning or trying multiple models.</small>';
                }, 15000);
                
                const veryLongWaitTimer = setTimeout(() => {
                    if (loadingMsg) loadingMsg.innerHTML += '<br><small style="color:var(--warning-color)">Taking longer than usual (retrying providers)... Please wait up to 2 minutes.</small>';
                }, 45000);

                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        message: message,
                        images: apiImages.length > 0 ? apiImages : []
                    })
                });

                clearTimeout(longWaitTimer);
                clearTimeout(veryLongWaitTimer);

                const data = await response.json();
                
                // Replace loading with response
                // const loadingMsg = document.querySelector(`[data-id="${loadingId}"] .message-content`); // Already defined above
                if (data.response) {
                    loadingMsg.innerHTML = data.response.replace(/\n/g, '<br>');
                } else if (data.detail) {
                    let errorMsg = data.detail;
                    if (typeof data.detail === 'object') {
                        errorMsg = JSON.stringify(data.detail, null, 2);
                    }
                    loadingMsg.textContent = "Error: " + errorMsg;
                    loadingMsg.style.color = "var(--danger-color)";
                    // Also log to console for debugging
                    console.error("Chat API Error:", data);
                }
            } catch (e) {
                const loadingMsg = document.querySelector(`[data-id="${loadingId}"] .message-content`);
                loadingMsg.textContent = "Network Error: " + e.message;
                loadingMsg.style.color = "var(--danger-color)";
            }
        }

        function addMessageToChat(role, text, images = []) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            const id = Date.now();
            div.setAttribute('data-id', id);
            
            let avatar = role === 'user' ? '👤' : '🤖';
            
            let contentHtml = text.replace(/\n/g, '<br>');
            if (images && images.length > 0) {
                 const counts = images.reduce((acc, curr) => {
                     const type = curr.type === 'image' ? 'Image' : (curr.type === 'pdf' ? 'PDF' : 'File');
                     acc[type] = (acc[type] || 0) + 1;
                     return acc;
                 }, {});
                 const summary = Object.entries(counts).map(([k,v]) => `${v} ${k}(s)`).join(', ');
                 contentHtml += `<br><small style="opacity:0.7"><i>[Attached: ${summary}]</i></small>`;
            }

            div.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">${contentHtml}</div>
            `;
            
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return id;
        }

        async function reloadConfig() {
            if(!confirm("Reload configuration?")) return;
            try {
                await fetch(`${API_BASE}/reload_config`, { method: 'POST' });
                alert("Reloaded!");
                fetchStatus();
            } catch (e) { alert("Error reloading"); }
        }

        function formatLimit(val) { return val < 0 ? '∞' : val; }

        // --- Auto-Fix ---
        async function fetchAutoFixData() {
            try {
                // Config
                const cRes = await fetch(`${API_BASE}/autofix/config`);
                autofixConfig = await cRes.json();
                
                // Update UI elements
                const afToggle = document.getElementById('autofix-toggle');
                const afLabel = document.getElementById('autofix-status');
                const afInterval = document.getElementById('af-interval');
                const afConfidence = document.getElementById('af-confidence');

                afToggle.checked = autofixConfig.auto_fix_enabled;
                afInterval.value = autofixConfig.schedule_interval_minutes;
                afConfidence.value = autofixConfig.auto_apply_confidence_threshold;
                
                afLabel.textContent = autofixConfig.auto_fix_enabled ? "Auto-Fix: ON" : "Auto-Fix: OFF";
                afLabel.style.color = autofixConfig.auto_fix_enabled ? "var(--success-color)" : "var(--text-secondary)";

                // Errors
                const eRes = await fetch(`${API_BASE}/errors`);
                const errors = await eRes.json();
                renderErrors(errors);

                // Patches
                const pRes = await fetch(`${API_BASE}/patches`);
                const patches = await pRes.json();
                renderPatches(patches);

            } catch (e) { console.error("Fetch autofix error", e); }
        }

        async function updateAutoFixConfig() {
            const afToggle = document.getElementById('autofix-toggle');
            const afInterval = document.getElementById('af-interval');
            const afConfidence = document.getElementById('af-confidence');
            
            const newConfig = {
                auto_fix_enabled: afToggle.checked,
                schedule_interval_minutes: parseInt(afInterval.value),
                auto_apply_confidence_threshold: parseFloat(afConfidence.value)
            };
            
            try {
                await fetch(`${API_BASE}/autofix/config`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(newConfig)
                });
                fetchAutoFixData(); // Refresh UI
            } catch (e) { console.error("Update config error", e); }
        }

        function renderErrors(errors) {
            const list = document.getElementById('errors-list');
            if(errors.length === 0) { list.innerHTML = '<p style="text-align:center; color:var(--text-secondary)">No active errors.</p>'; return; }
            
            list.innerHTML = errors.reverse().slice(0, 10).map(e => `
                <div class="log-item">
                    <div class="log-header">
                        <strong style="color:var(--danger-color)">${e.status.toUpperCase()}</strong>
                        <span style="color:var(--text-secondary)">${new Date(e.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="log-msg">${e.message}</div>
                    ${e.analysis ? `<div style="margin-top:5px; color:var(--accent-color); font-size:12px;">💡 ${e.analysis}</div>` : ''}
                </div>
            `).join('');
        }

        function renderPatches(patches) {
            const list = document.getElementById('patches-list');
            const pending = patches.filter(p => p.status === 'pending');
            if(pending.length === 0) { list.innerHTML = '<p style="text-align:center; color:var(--text-secondary)">No pending patches.</p>'; return; }
            
            list.innerHTML = pending.reverse().map(p => {
                const analysis = p.analysis || {};
                const what = analysis.what || p.suggestion;
                const why = analysis.why || "Analysis not available";
                const remedy = Array.isArray(analysis.remedy) ? analysis.remedy.join('</li><li>') : "Manual steps";
                const simResult = p.simulation_result ? 
                    (p.simulation_result.success ? 
                        `<span style="color:var(--success-color)">✅ Verified: ${p.simulation_result.message}</span>` : 
                        `<span style="color:var(--danger-color)">❌ Failed: ${p.simulation_result.message}</span>`) 
                    : '<span style="color:var(--text-secondary)">Not simulated yet</span>';

                return `
                <div class="log-item">
                    <div class="log-header">
                        <strong style="color:var(--success-color)">${(p.confidence*100).toFixed(0)}% Confidence</strong>
                        <div style="display:flex; gap:5px;">
                            <button class="primary-btn" style="padding:4px 8px; font-size:11px; background:var(--accent-color);" onclick="simulatePatch('${p.id}')">🧪 Simulate & Verify</button>
                            <button class="primary-btn" style="padding:4px 8px; font-size:11px;" onclick="applyPatch('${p.id}')">Apply</button>
                        </div>
                    </div>
                    
                    <div style="margin-top:10px; font-size:13px;">
                        <div style="font-weight:bold; color:var(--text-primary); margin-bottom:4px;">What Happened:</div>
                        <div style="color:var(--text-secondary); margin-bottom:10px;">${what}</div>
                        
                        <div style="font-weight:bold; color:var(--text-primary); margin-bottom:4px;">Why it Happened:</div>
                        <div style="color:var(--text-secondary); margin-bottom:10px;">${why}</div>

                        <div style="font-weight:bold; color:var(--text-primary); margin-bottom:4px;">Remedy Plan:</div>
                        <ul style="color:var(--text-secondary); margin-bottom:10px; padding-left:20px;">
                            <li>${remedy}</li>
                        </ul>

                        <div style="background:rgba(0,0,0,0.2); padding:8px; border-radius:4px; margin-top:10px;">
                            <strong>Simulation Status:</strong> ${simResult}
                        </div>
                    </div>
                </div>
            `}).join('');
        }

        async function simulatePatch(id) {
            const btn = event.target;
            btn.textContent = "Simulating...";
            btn.disabled = true;
            try {
                const res = await fetch(`${API_BASE}/patches/${id}/simulate`, { method: 'POST' });
                const result = await res.json();
                if(!result.success) alert("Simulation Failed: " + result.message);
                fetchAutoFixData();
            } catch(e) { alert("Error: " + e); }
        }

        async function applyPatch(id) {
            if(!confirm("Apply patch? This will modify server code.")) return;
            try {
                const res = await fetch(`${API_BASE}/patches/${id}/apply`, { method: 'POST' });
                const result = await res.json();
                if(result.success) {
                    alert("Patch Applied Successfully!");
                } else {
                    alert("Patch Failed: " + result.message);
                }
                fetchAutoFixData();
            } catch(e) { alert("Error applying patch: " + e); }
        }

        // --- Versions & Trust ---
        async function fetchVersions() {
            try {
                const res = await fetch(`${API_BASE}/versions`);
                const data = await res.json();
                renderVersions(data);
            } catch(e) { console.error("Fetch versions error", e); }
        }

        function renderVersions(data) {
            document.getElementById('current-version-display').textContent = data.current_version || "Unknown";
            
            const tbody = document.querySelector('#versions-table tbody');
            tbody.innerHTML = '';
            
            if(!data.versions || data.versions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">No snapshots found</td></tr>';
                return;
            }

            // Show newest first
            const sorted = [...data.versions].reverse();
            
            sorted.forEach(v => {
                const tr = document.createElement('tr');
                const isCurrent = v.id === data.current_version;
                tr.innerHTML = `
                    <td style="font-family:monospace;">${v.id}</td>
                    <td>${new Date(v.timestamp).toLocaleString()}</td>
                    <td>${v.label}</td>
                    <td><span class="badge ${isCurrent ? 'badge-fast' : 'badge-std'}">${isCurrent ? 'Active' : 'Backup'}</span></td>
                    <td>
                        ${!isCurrent ? `<button class="action-btn" onclick="rollbackSystem('${v.id}')">Rollback To This</button>` : '<span style="color:var(--text-secondary); font-size:12px;">Current</span>'}
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function rollbackSystem(id) {
            if(!confirm(`WARNING: Are you sure you want to rollback to version ${id}? This will revert code changes.`)) return;
            try {
                const res = await fetch(`${API_BASE}/rollback`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ version_id: id })
                });
                const result = await res.json();
                if(result.success) {
                    alert("Rollback Successful! The server state has been restored.");
                    fetchVersions();
                } else {
                    alert("Rollback Failed: " + result.detail);
                }
            } catch(e) { alert("Error requesting rollback: " + e); }
        }

        // --- File Management ---
        async function fetchFiles() {
            try {
                const res = await fetch(`${API_BASE}/quotas`);
                const data = await res.json();
                const list = document.getElementById('file-list');
                list.innerHTML = data.files.map(f => `
                    <li style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid var(--border-color); color:var(--text-primary);">
                        <span>${f}</span>
                        <button class="action-btn" style="color:var(--danger-color); border-color:var(--danger-color); padding:2px 8px;" onclick="deleteFile('${f}')">Delete</button>
                    </li>
                `).join('');
            } catch(e) {}
        }

        async function deleteFile(f) {
            if(!confirm(`Delete ${f}?`)) return;
            await fetch(`${API_BASE}/quotas/${f}`, { method: 'DELETE' });
            fetchFiles();
        }

        // Drag & Drop
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        dropZone.onclick = () => fileInput.click();
        fileInput.onchange = (e) => { if(e.target.files.length) uploadFile(e.target.files[0]); };
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => {
            e.preventDefault(); dropZone.classList.remove('dragover');
            if(e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
        };

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            await fetch(`${API_BASE}/quotas/upload`, { method: 'POST', body: formData });
            alert("Uploaded!");
            fetchFiles();
        }

        // --- Superpowers ---
        function switchSPTab(tab) {
            document.querySelectorAll('.sp-tab').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.sp-content').forEach(el => el.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(`sp-${tab}`).classList.add('active');
        }

        async function runSuperpower(skill) {
            const out = document.getElementById(`sp-${skill}-out`);
            out.innerHTML = '<span class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span> Processing...';
            
            let body = {};
            if (skill === 'brainstorm') {
                body = { 
                    problem: document.getElementById('sp-brainstorm-input').value,
                    context: document.getElementById('sp-brainstorm-context').value
                };
            } else if (skill === 'plan') {
                body = { design_doc: document.getElementById('sp-plan-input').value };
            } else if (skill === 'tdd') {
                body = { feature_spec: document.getElementById('sp-tdd-input').value };
            } else if (skill === 'review') {
                body = { code: document.getElementById('sp-review-input').value };
            } else if (skill === 'debug') {
                body = { error_log: document.getElementById('sp-debug-input').value };
            }

            try {
                const res = await fetch(`${API_BASE}/superpowers/${skill}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                
                let content = null;
                if (skill === 'brainstorm') content = data.result;
                else if (skill === 'plan') content = data.plan;
                else if (skill === 'review') content = data.review;
                else if (skill === 'tdd') content = data.test_code;
                else if (skill === 'debug') content = data.analysis;
                
                if (!content && data) content = data;

                if (skill === 'plan' && Array.isArray(content)) {
                    // Store plan for "Send to Coder" feature
                    currentPlan = content;
                    
                    out.innerHTML = `
                        <div style="margin-bottom:15px;">
                            <button class="primary-btn" style="background:var(--accent-color)" onclick="sendPlanToCoder()">🚀 Implement Plan with AI Coder</button>
                        </div>
                    ` + content.map(step => `
                        <div style="margin-bottom:15px; border-left:3px solid var(--accent-color); padding-left:10px;">
                            <div style="font-weight:600; color:var(--text-primary)">${step.step}. ${step.title}</div>
                            <div style="color:var(--text-secondary); margin-bottom:5px;">${step.description}</div>
                            <div style="font-size:12px; color:var(--text-secondary)">Files: ${(step.files||[]).join(', ')}</div>
                        </div>
                    `).join('');
                } else {
                    let text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
                    out.innerHTML = text.replace(/\n/g, '<br>').replace(/```/g, ''); 
                }
            } catch (e) {
                out.innerHTML = `<span style="color:var(--danger-color)">Error: ${e.message}</span>`;
            }
        }

        // --- Init moved to dashboard plugin bootstrap ---

async function fetchCoderModels() {
            try {
                const response = await fetch(`${API_BASE}/chat/models`);
                const data = await response.json();
                const select = document.getElementById('coder-model-select');
                select.innerHTML = '';
                
                // Prioritize "Code" or "Premium" models
                const models = data.models.sort((a, b) => {
                    const tierOrder = { "premium": 1, "standard": 2, "economical": 3, "free": 4 };
                    return (tierOrder[a.tier] || 5) - (tierOrder[b.tier] || 5);
                });

                if (!models.length) {
                    const opt = document.createElement('option');
                    opt.text = "No active models";
                    select.add(opt);
                    return;
                }

                models.forEach(m => {
                    if (!m.available) return;
                    const opt = document.createElement('option');
                    opt.value = m.model;
                    opt.text = `${m.model} (${m.tier} - ${m.category})`;
                    if (m.model.toLowerCase().includes('claude') || m.model.toLowerCase().includes('gpt-4')) {
                        opt.selected = true; // Auto-select capable models
                    }
                    select.add(opt);
                });
            } catch (e) { console.error("Fetch coder models error", e); }
        }

        function sendPlanToCoder() {
            if (!currentPlan) return alert("No plan available to send.");
            
            // Format plan into a prompt
            const planText = currentPlan.map(s => `${s.step}. ${s.title}\n   ${s.description} (Files: ${(s.files||[]).join(', ')})`).join('\n\n');
            const fullPrompt = `Implement the following plan:\n\n${planText}\n\nStrictly follow the steps and file paths provided.`;
            
            // Switch view
            switchView('coder');
            
            // Populate prompt
            document.getElementById('coder-prompt').value = fullPrompt;
            
            // Optional: Auto-scroll or focus
            document.getElementById('coder-prompt').focus();
        }

        function toggleCoderMode() {
            const isClaude = document.getElementById('use-claude-cli').checked;
            const select = document.getElementById('coder-model-select');
            const btn = document.getElementById('btn-gen-patch');
            const settings = document.getElementById('claude-settings');
            
            if (isClaude) {
                select.disabled = true;
                btn.textContent = "Run Claude Code";
                settings.style.display = 'block';
            } else {
                select.disabled = false;
                btn.textContent = "Generate Patch";
                settings.style.display = 'none';
            }
        }

        async function generatePatch() {
            const prompt = document.getElementById('coder-prompt').value;
            const model = document.getElementById('coder-model-select').value;
            const useClaude = document.getElementById('use-claude-cli').checked;
            const apiBase = document.getElementById('claude-api-base').value;
            const apiKey = document.getElementById('claude-api-key').value;
            
            if (!prompt) return alert("Please enter a prompt.");
            
            const btn = document.getElementById('btn-gen-patch');
            const preview = document.getElementById('coder-preview');
            
            btn.disabled = true;
            btn.textContent = useClaude ? "Running Claude..." : "Generating...";
            
            // Clear previous results but prepare layout
            preview.innerHTML = '';
            
            // Console Output Container
            const consoleDiv = document.createElement('div');
            consoleDiv.id = 'coder-console';
            consoleDiv.style.backgroundColor = '#1e1e1e';
            consoleDiv.style.color = '#0f0';
            consoleDiv.style.fontFamily = 'monospace';
            consoleDiv.style.padding = '15px';
            consoleDiv.style.borderRadius = '8px';
            consoleDiv.style.marginBottom = '20px';
            consoleDiv.style.maxHeight = '300px';
            consoleDiv.style.overflowY = 'auto';
            consoleDiv.style.border = '1px solid var(--border-color)';
            consoleDiv.innerHTML = '<div style="color:#888; margin-bottom:10px;">// AI Agent Logs</div>';
            preview.appendChild(consoleDiv);
            
            function log(msg) {
                const line = document.createElement('div');
                line.style.marginTop = '4px';
                line.style.fontSize = '12px';
                line.style.borderBottom = '1px solid #333';
                line.style.paddingBottom = '2px';
                
                // Add timestamp
                const time = new Date().toLocaleTimeString();
                line.innerHTML = `<span style="color:#666">[${time}]</span> ${msg}`;
                
                consoleDiv.appendChild(line);
                consoleDiv.scrollTop = consoleDiv.scrollHeight;
            }

            try {
                const endpoint = useClaude ? `${API_BASE}/claude/run` : `${API_BASE}/coder/generate_stream`;
                const body = useClaude 
                    ? { prompt, api_base: apiBase || undefined, api_key: apiKey || undefined } 
                    : { prompt, model, api_base: apiBase || undefined };
                
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const parts = buffer.split('\n\n');
                    buffer = parts.pop(); // Keep incomplete part
                    
                    for (const part of parts) {
                        if (part.startsWith('data: ')) {
                            const jsonStr = part.substring(6);
                            try {
                                const event = JSON.parse(jsonStr);
                                
                                if (event.type === 'log') {
                                    log(event.message);
                                } else if (event.type === 'result') {
                                    const data = event.data;
                                    log("✨ Processing complete.");
                                    
                                    // Render results below console
                                    const resultContainer = document.createElement('div');
                                    resultContainer.id = 'coder-results';
                                    resultContainer.style.animation = 'fadeIn 0.5s';
                                    preview.appendChild(resultContainer);
                                    
                                    if (data.error) {
                                        resultContainer.innerHTML = `<div style="color:var(--danger-color); margin-top:20px;">Error: ${data.error}</div>`;
                                        if (data.raw) resultContainer.innerHTML += `<hr><pre>${data.raw.replace(/</g, '&lt;')}</pre>`;
                                    } else if (data.patches) {
                                        currentPatches = data.patches;
                                        renderPreview(data, resultContainer);
                                        document.getElementById('btn-apply-patch').style.display = 'block';
                                        document.getElementById('patch-status').textContent = `${data.patches.length} file(s) to modify.`;
                                    } else if (data.success) {
                                        resultContainer.innerHTML = `<div style="color:var(--success-color); margin-top:20px; font-size:16px;">✅ ${data.message}</div>`;
                                        resultContainer.innerHTML += `<div style="color:var(--text-secondary); margin-top:5px;">Changes have been applied directly to the filesystem.</div>`;
                                        fetchFiles();
                                        fetchVersions();
                                    } else {
                                         resultContainer.innerHTML = `<div style="color:var(--warning-color)">Unexpected response format.</div><pre>${JSON.stringify(data, null, 2)}</pre>`;
                                    }
                                    
                                } else if (event.type === 'error') {
                                    log(`❌ Error: ${event.message}`);
                                    preview.innerHTML += `<div style="color:var(--danger-color); margin-top:10px;">Stream Error: ${event.message}</div>`;
                                }
                            } catch (e) {
                                console.error("Parse error", e);
                            }
                        }
                    }
                }
                
            } catch (e) {
                preview.innerHTML += `<div style="color:var(--danger-color)">Network Error: ${e.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = useClaude ? "Run Claude Code" : "Generate Patch";
            }
        }

        function renderPreview(data, targetContainer = null) {
            const container = targetContainer || document.getElementById('coder-preview');
            if (!targetContainer) container.innerHTML = '';
            
            // 1. Thought Section
            if (data.thought) {
                const thoughtDiv = document.createElement('div');
                thoughtDiv.style.marginBottom = '20px';
                thoughtDiv.style.padding = '15px';
                thoughtDiv.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                thoughtDiv.style.borderLeft = '4px solid var(--accent-color)';
                thoughtDiv.style.borderRadius = '4px';
                thoughtDiv.innerHTML = `<strong style="color:var(--accent-color)">🧠 Thought:</strong><br><div style="margin-top:5px; white-space:pre-wrap;">${data.thought}</div>`;
                container.appendChild(thoughtDiv);
            }

            // 2. Plan Section
            if (data.plan && data.plan.length > 0) {
                const planDiv = document.createElement('div');
                planDiv.style.marginBottom = '20px';
                planDiv.innerHTML = `<strong style="color:var(--success-color)">📋 Plan:</strong><ul style="margin-top:5px; padding-left:20px;">${data.plan.map(step => `<li>${step}</li>`).join('')}</ul>`;
                container.appendChild(planDiv);
            }

            // 3. Shell Commands (Scripts)
            if (data.shell_commands && data.shell_commands.length > 0) {
                const scriptDiv = document.createElement('div');
                scriptDiv.style.marginBottom = '20px';
                scriptDiv.innerHTML = `<strong style="color:var(--warning-color)">💻 Shell Scripts:</strong>
                    <div style="background:black; color:#0f0; padding:10px; border-radius:5px; margin-top:5px; font-family:monospace;">
                        ${data.shell_commands.map(cmd => `<div>$ ${cmd}</div>`).join('')}
                    </div>`;
                container.appendChild(scriptDiv);
            }

            // 4. Patches
            if (data.patches && data.patches.length > 0) {
                const patchesHeader = document.createElement('h3');
                patchesHeader.textContent = "Proposed File Changes";
                patchesHeader.style.fontSize = "16px";
                patchesHeader.style.marginTop = "20px";
                patchesHeader.style.borderBottom = "1px solid var(--border-color)";
                container.appendChild(patchesHeader);

                data.patches.forEach(p => {
                    const item = document.createElement('div');
                    item.style.marginBottom = '15px';
                    item.style.borderBottom = '1px solid var(--border-color)';
                    item.style.paddingBottom = '10px';
                    
                    let color = 'var(--text-primary)';
                    let icon = '📄';
                    if (p.action === 'create') { color = 'var(--success-color)'; icon = '✨'; }
                    if (p.action === 'delete') { color = 'var(--danger-color)'; icon = '🗑️'; }
                    if (p.action === 'replace') { color = 'var(--warning-color)'; icon = '📝'; }
                    
                    const contentPreview = p.content ? p.content.substring(0, 200).replace(/</g, '&lt;') + (p.content.length > 200 ? '...' : '') : '(Deletion)';
                    
                    item.innerHTML = `
                        <div style="font-weight:600; color:${color}; margin-bottom:5px;">${icon} ${p.action.toUpperCase()}: ${p.file}</div>
                        <div style="font-size:11px; background:rgba(0,0,0,0.3); padding:5px; overflow-x:auto;">${contentPreview}</div>
                    `;
                    container.appendChild(item);
                });
            }
        }

        async function applyPatch() {
            if (!currentPatches) return;
            
            if (!confirm("Are you sure you want to apply these changes to the codebase?")) return;
            
            const btn = document.getElementById('btn-apply-patch');
            btn.disabled = true;
            btn.textContent = "Applying...";
            
            try {
                const response = await fetch(`${API_BASE}/coder/apply`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ patches: currentPatches })
                });
                const data = await response.json();
                
                let msg = "Patch Applied!\n\n";
                data.results.forEach(r => {
                    msg += `${r.status === 'success' ? '✅' : '❌'} ${r.file}: ${r.status}\n`;
                });
                alert(msg);
                
                document.getElementById('coder-preview').innerHTML = '<p style="text-align:center; margin-top:100px;">Patch Applied Successfully.</p>';
                document.getElementById('btn-apply-patch').style.display = 'none';
                currentPatches = null;
                
            } catch (e) {
                alert("Error applying patch: " + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = "Apply Patch";
            }
        }
