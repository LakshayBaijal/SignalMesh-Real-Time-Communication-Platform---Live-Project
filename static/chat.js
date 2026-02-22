// static/chat.js (UPDATED — handles {"type":"suggestions"} properly)
(() => {
  const host = location.host;

  // choose ws or wss based on page protocol
  const protocol = (location.protocol === "https:") ? "wss://" : "ws://";
  const anonWsUrl = protocol + host + "/ws";

  const usersListEl = document.getElementById("users-list");
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const typingIndicator = document.getElementById("typing-indicator");
  const headerUser = document.getElementById("header-username");

  // counters
  const userCountEl = document.getElementById("user-count");
  const viewerCountEl = document.getElementById("viewer-count");

  // file + emoji UI
  const fileInput = document.getElementById("file-input");
  const emojiBtn = document.getElementById("emoji-btn");
  const emojiPickerDiv = document.getElementById("emoji-picker");

  // auth modal elements
  const authModal = document.getElementById("auth-modal");
  const authUsername = document.getElementById("auth-username");
  const authPassword = document.getElementById("auth-password");
  const authSubmit = document.getElementById("auth-submit");
  const toggleLogin = document.getElementById("toggle-login");
  const toggleSignup = document.getElementById("toggle-signup");
  const continueGuest = document.getElementById("continue-guest");
  const authError = document.getElementById("auth-error");
  const authTitle = document.getElementById("auth-title");

  // guest popup
  const mustAuthPopup = document.getElementById("must-auth-popup");
  const popupSignin = document.getElementById("popup-signin");

  let anonSocket = null;
  let authSocket = null;
  let username = null;
  let token = null;
  let color = null;
  let mode = "login"; // or signup
  let users = {}; // name -> {name,color,lastSeen}

  function nowTime() {
    const d = new Date();
    return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  }
  function randomColorFromName(name){
    const palette = ["#ef4444","#f97316","#f59e0b","#10b981","#06b6d4","#3b82f6","#7c3aed","#ec4899"];
    let h=0;
    for(let i=0;i<name.length;i++) h = (h*31 + name.charCodeAt(i))|0;
    return palette[Math.abs(h) % palette.length];
  }

  function addRawMessage(name, text, mine=false){
    const row = document.createElement("div");
    row.className = "msg-row " + (mine ? "me" : "other");

    const bubble = document.createElement("div");
    bubble.className = "msg " + (mine ? "me" : "other");

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${name} • ${nowTime()}`;

    const content = document.createElement("div");
    content.className = "content";
    content.innerHTML = text || "";

    bubble.appendChild(meta);
    bubble.appendChild(content);

    row.appendChild(bubble);

    chatBox.appendChild(row);

    scrollToBottom();
  }

  function addSystemNotice(text){
    const wrap = document.createElement("div");
    wrap.style.textAlign = "center";
    wrap.style.color = "var(--muted)";
    wrap.style.fontSize = "13px";
    wrap.textContent = text;
    chatBox.appendChild(wrap);
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'auto' });
  }

  function renderUsers(){
    usersListEl.innerHTML = "";
    Object.values(users).forEach(u => {
      const li = document.createElement("li");
      li.innerHTML = `<div class="avatar" style="background:${u.color}">${(u.name||"U").slice(0,2).toUpperCase()}</div>
                      <div>
                        <div style="font-weight:600">${u.name}</div>
                        <div style="font-size:12px;color:var(--muted)">${u.lastSeen||"online"}</div>
                      </div>`;
      usersListEl.appendChild(li);
    });
  }

  // escape to avoid HTML injection for user messages
  function escapeHtml(unsafe) {
    return String(unsafe || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ===== NEW: showSuggestions =====
  function showSuggestions(list){
    // ensure list is array
    if(!Array.isArray(list) || list.length === 0) {
      // clear suggestions container if present
      const existing = document.getElementById("suggestions");
      if(existing) existing.innerHTML = "";
      return;
    }

    let container = document.getElementById("suggestions");

    // create container if not exists (insert above composer)
    if(!container){
      container = document.createElement("div");
      container.id = "suggestions";
      container.style.padding = "8px";
      container.style.display = "flex";
      container.style.gap = "8px";
      container.style.flexWrap = "wrap";
      container.style.alignItems = "center";
      container.style.justifyContent = "flex-start";

      const chatBox = document.getElementById("chat-box");
      chatBox.appendChild(container);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Render buttons
    container.innerHTML = "";
    list.forEach(text => {
      const btn = document.createElement("button");
      btn.className = "suggestion";
      btn.textContent = text;
      // basic inline styles (keeps same look regardless of CSS)
      btn.style.padding = "8px 12px";
      btn.style.borderRadius = "999px";
      btn.style.border = "none";
      btn.style.cursor = "pointer";
      btn.style.background = "#7c3aed";
      btn.style.color = "white";
      btn.style.fontSize = "13px";

      btn.onclick = function(){
        // default behaviour: fill input, focus it
        messageInput.value = text;
        messageInput.focus();
      };

      container.appendChild(btn);
    });
  }

  // common message handler for messages from server (both anon and auth)
  function handleServerData(data){
    let obj = null;
    try { obj = JSON.parse(data); } catch(e){ obj = null; }

    // NEW: intercept suggestions BEFORE any rendering
    if(obj && obj.type === "suggestions"){
      // show suggestions as buttons and DO NOT render as server message
      showSuggestions(obj.suggestions || []);
      return;
    }

    if(obj && obj.type === "user_count"){
      userCountEl.textContent = `Online: ${obj.auth} users`;
      viewerCountEl.textContent = `Viewers: ${obj.total} total`;
      return;
    }

    if(obj && obj.type){
      if(obj.type === "join"){
        users[obj.name] = {name: obj.name, color: obj.color || randomColorFromName(obj.name), lastSeen: "online"};
        renderUsers();
        addSystemNotice(`${obj.name} joined the world`);
      } else if(obj.type === "leave"){
        if(users[obj.name]) users[obj.name].lastSeen = "left";
        renderUsers();
        addSystemNotice(`${obj.name} left`);
      } else if(obj.type === "typing"){
        typingIndicator.textContent = `${obj.name} is typing…`;
        clearTimeout(window._typingTimeout);
        window._typingTimeout = setTimeout(()=> { typingIndicator.textContent = ""; }, 1400);
      } else if(obj.type === "message"){
        const mine = username && obj.name === username;
        addRawMessage(obj.name, escapeHtml(obj.text), mine);
        // Clear suggestions after a new real message is rendered (user likely consumed them)
        const sug = document.getElementById("suggestions");
        if(sug) sug.innerHTML = "";
      } else if(obj.type === "file"){
        const mine = username && obj.name === username;
        const safeUrl = escapeHtml(obj.url || "");
        const safeFilename = escapeHtml(obj.filename || "file");
        const linkHtml = `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">📎 ${safeFilename}</a>`;
        addRawMessage(obj.name, linkHtml, mine);
        const sug = document.getElementById("suggestions");
        if(sug) sug.innerHTML = "";
      } else {
        addRawMessage("server", escapeHtml(JSON.stringify(obj)));
      }
    } else {
      const s = String(data);
      const sep = s.indexOf(":");
      if(sep > 0){
        const n = s.slice(0, sep).trim();
        const t = s.slice(sep+1).trim();
        addRawMessage(n, escapeHtml(t), false);
      } else {
        addRawMessage("server", escapeHtml(s), false);
      }
    }
  }

  // ---------------- anonymous connection ----------------
  function connectAnon(){
    anonSocket = new WebSocket(anonWsUrl);
    anonSocket.addEventListener("open", () => {
      console.log("connected anon websocket");
      headerUser.textContent = "Viewing as guest";
    });
    anonSocket.addEventListener("message", (ev) => {
      handleServerData(ev.data);
    });
    anonSocket.addEventListener("close", () => {
      console.log("anon socket closed");
    });
  }

  // ---------------- authenticated connection ----------------
  function connectAuth(tkn, name, c){
    token = tkn;
    username = name;
    color = c || randomColorFromName(name);
    const url = protocol + host + "/ws/" + token;
    authSocket = new WebSocket(url);

    authSocket.addEventListener("open", () => {
      console.log("auth socket open");
      headerUser.textContent = `You: ${username}`;
      setTimeout(scrollToBottom, 100);
      // server will announce join
    });

    authSocket.addEventListener("message", (ev) => {
      handleServerData(ev.data);
      scrollToBottom();
    });

    authSocket.addEventListener("close", () => {
      headerUser.textContent = "Disconnected — viewing as guest";
      authSocket = null;
      if(!anonSocket || anonSocket.readyState !== WebSocket.OPEN) connectAnon();
    });
  }

  // ---------------- auth form logic ----------------
  toggleLogin.onclick = () => { mode = "login"; authTitle.textContent = "Login to World Chat"; authError.textContent = ""; }
  toggleSignup.onclick = () => { mode = "signup"; authTitle.textContent = "Create an account"; authError.textContent = ""; }

  async function doAuth(){
    const u = authUsername.value.trim();
    const p = authPassword.value.trim();
    if(!u || !p){ authError.textContent = "username and password required"; return; }
    authError.textContent = "";
    try {
      if(mode === "signup"){
        const res = await fetch("/signup", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({username:u,password:p})});
        if(!res.ok){
          const err = await res.json(); authError.textContent = err.detail || "Signup failed"; return;
        }
      }
      const res2 = await fetch("/login", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({username:u,password:p})});
      if(!res2.ok){
        const err = await res2.json(); authError.textContent = err.detail || "Login failed"; return;
      }
      const body = await res2.json();
      const tok = body.token;
      try { if(anonSocket){ anonSocket.close(); anonSocket = null; } } catch(e){}
      connectAuth(tok, u, randomColorFromName(u));
      authModal.style.display = "none";
    } catch(e){
      authError.textContent = "Network error";
      console.error(e);
    }
  }

  authSubmit.onclick = doAuth;
  authPassword.addEventListener("keydown", (e) => { if(e.key === "Enter") doAuth(); });

  continueGuest.onclick = () => {
    authModal.style.display = "none";
    headerUser.textContent = "Viewing as guest";
  };

  popupSignin.onclick = () => {
    mustAuthPopup.style.display = "none";
    authModal.style.display = "flex";
  };

  // ---------------- file upload ----------------
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if(!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/upload", {
        method: "POST",
        body: formData
      });
      if(!res.ok){
        console.error("upload failed");
        return;
      }
      const data = await res.json();

      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        authSocket.send(JSON.stringify({
          type: "file",
          name: username,
          url: data.url,
          filename: data.filename
        }));
      } else {
        mustAuthPopup.style.display = "flex";
        setTimeout(()=> { mustAuthPopup.style.display = "none"; }, 3500);
      }
    } catch(e){
      console.error("upload error", e);
    } finally {
      fileInput.value = "";
    }
  });

  // ---------------- emoji picker ----------------
  let pickerOpen = false;
  emojiBtn.onclick = () => {

    if(pickerOpen){
      emojiPickerDiv.style.display = "none";
      pickerOpen = false;
      return;
    }

    emojiPickerDiv.innerHTML = "";

    try {
      const picker = new EmojiMart.Picker({
        onEmojiSelect: (emoji) => {
          messageInput.value += (emoji.native || emoji.colons || emoji.native);
          messageInput.focus();
        },
        theme: "dark",
        perLine: 8,
        showPreview: false,
        showSkinTones: false
      });
      emojiPickerDiv.appendChild(picker);
      emojiPickerDiv.style.display = "block";
      pickerOpen = true;
    } catch (err) {
      console.error("emoji picker load error", err);
      messageInput.value += "😊";
    }
  };

  // ---------------- send flow ----------------
  function sendMessage(){
    const text = messageInput.value.trim();
    if(!text) return;

    if(text.length > 5000){
      alert("Message is too long (limit 5000 characters). Please shorten it.");
      return;
    }

    if(authSocket && authSocket.readyState === WebSocket.OPEN){
      const msg = {type:"message", name: username, text};
      authSocket.send(JSON.stringify(msg));
      messageInput.value = "";
      // clear suggestions once user sends
      const sug = document.getElementById("suggestions");
      if(sug) sug.innerHTML = "";
    } else {
      mustAuthPopup.style.display = "flex";
      setTimeout(()=> { mustAuthPopup.style.display = "none"; }, 4000);
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  messageInput.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    } else {
      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        try { authSocket.send(JSON.stringify({type:"typing", name: username})); } catch(e){}
      }
    }
  });

  window.addEventListener("beforeunload", () => {
    try {
      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        authSocket.close();
      }
    } catch(e){}
  });
  
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatBox.scrollTop = chatBox.scrollHeight;
  });
}
  // init
  connectAnon();
  authModal.style.display = "flex";
  authUsername.focus();
})();


