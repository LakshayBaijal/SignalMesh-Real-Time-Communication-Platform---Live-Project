// static/chat.js (final - suggestion buttons send immediately and persist)
(() => {
  const host = location.host;
  const protocol = (location.protocol === "https:") ? "wss://" : "ws://";
  const anonWsUrl = protocol + host + "/ws";

  const usersListEl = document.getElementById("users-list");
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const headerUser = document.getElementById("header-username");

  // auth modal elements
  const authModal = document.getElementById("auth-modal");
  const authUsername = document.getElementById("auth-username");
  const authPassword = document.getElementById("auth-password");
  const authSubmit = document.getElementById("auth-submit");
  const continueGuest = document.getElementById("continue-guest");
  const mustAuthPopup = document.getElementById("must-auth-popup");

  let anonSocket = null;
  let authSocket = null;
  let username = null;
  let token = null;
  let users = {};

  function nowTime() {
    const d = new Date();
    return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  }

  function escapeHtml(unsafe) {
    return String(unsafe || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
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
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function addSystemNotice(text){
    const wrap = document.createElement("div");
    wrap.style.textAlign = "center";
    wrap.style.color = "var(--muted)";
    wrap.style.fontSize = "13px";
    wrap.style.margin = "6px 0";
    wrap.textContent = text;
    chatBox.appendChild(wrap);
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'auto' });
  }

  function renderUsers(){
    usersListEl.innerHTML = "";
    Object.values(users).forEach(u => {
      const li = document.createElement("li");
      li.innerHTML = `<div class="avatar">${(u.name||"U").slice(0,2).toUpperCase()}</div>
                      <div style="margin-left:8px">
                        <div style="font-weight:600">${u.name}</div>
                        <div style="font-size:12px;color:var(--muted)">online</div>
                      </div>`;
      usersListEl.appendChild(li);
    });
  }

  function showSuggestions(list){
    if(!Array.isArray(list) || list.length === 0) {
      const existing = document.getElementById("suggestions");
      if(existing) existing.innerHTML = "";
      return;
    }

    let container = document.getElementById("suggestions");
    if(!container){
      container = document.createElement("div");
      container.id = "suggestions";
      container.style.padding = "8px";
      container.style.display = "flex";
      container.style.gap = "8px";
      container.style.flexWrap = "wrap";
      const composer = document.querySelector(".composer");
      if(composer && composer.parentNode){
        composer.parentNode.insertBefore(container, composer);
      } else {
        document.body.appendChild(container);
      }
    }

    container.innerHTML = "";
    list.forEach(text => {
      const btn = document.createElement("button");
      btn.className = "suggestion";
      btn.textContent = text;
      btn.style.padding = "8px 12px";
      btn.style.borderRadius = "999px";
      btn.style.border = "none";
      btn.style.cursor = "pointer";
      btn.style.background = "#7c3aed";
      btn.style.color = "white";
      btn.style.fontSize = "13px";

      btn.onclick = function(){
        // send immediately as a message (if authenticated)
        if(authSocket && authSocket.readyState === WebSocket.OPEN){
          authSocket.send(JSON.stringify({type: "message", name: username, text: text}));
        } else {
          // otherwise, fill input and show auth prompt
          messageInput.value = text;
          mustAuthPopup.style.display = "flex";
          setTimeout(()=> { mustAuthPopup.style.display = "none"; }, 3000);
        }
      };

      container.appendChild(btn);
    });
  }

  // server message parser
  function handleServerData(data){
    let obj = null;
    try { obj = JSON.parse(data); } catch(e){ obj = null; }

    if(obj && obj.type === "suggestions"){
      showSuggestions(obj.suggestions || []);
      return;
    }

    if(obj && obj.type){
      if(obj.type === "join"){
        users[obj.name] = {name: obj.name, color: obj.color || "#777"};
        renderUsers();
        addSystemNotice(`${obj.name} joined the world`);
      } else if(obj.type === "leave"){
        if(users[obj.name]) delete users[obj.name];
        renderUsers();
        addSystemNotice(`${obj.name} left`);
      } else if(obj.type === "message"){
        const mine = username && obj.name === username;
        addRawMessage(obj.name, escapeHtml(obj.text), mine);
      } else if(obj.type === "file"){
        const mine = username && obj.name === username;
        const safeUrl = escapeHtml(obj.url || "");
        const safeFilename = escapeHtml(obj.filename || "file");
        const linkHtml = `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">📎 ${safeFilename}</a>`;
        addRawMessage(obj.name, linkHtml, mine);
      } else if(obj.type === "user_count"){
        // optionally display counts somewhere
      } else {
        addRawMessage("server", escapeHtml(JSON.stringify(obj)));
      }
    } else {
      // plain text fallback
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
    try {
      anonSocket = new WebSocket(anonWsUrl);
    } catch(e){
      console.error("Anon WS connect error", e);
      return;
    }
    anonSocket.addEventListener("open", () => {
      headerUser.textContent = "Viewing as guest";
    });
    anonSocket.addEventListener("message", (ev) => {
      handleServerData(ev.data);
    });
    anonSocket.addEventListener("close", () => { console.log("anon socket closed"); });
  }

  // ---------------- authenticated connection ----------------
  function connectAuth(tkn, name){
    token = tkn;
    username = name;
    const url = protocol + host + "/ws/" + token;
    try {
      authSocket = new WebSocket(url);
    } catch(e){
      console.error("Auth WS connect error", e);
      return;
    }
    authSocket.addEventListener("open", () => {
      headerUser.textContent = `You: ${username}`;
    });
    authSocket.addEventListener("message", (ev) => {
      handleServerData(ev.data);
    });
    authSocket.addEventListener("close", () => {
      headerUser.textContent = "Disconnected — viewing as guest";
      authSocket = null;
      if(!anonSocket || anonSocket.readyState !== WebSocket.OPEN) connectAnon();
    });
  }

  // ---------------- auth logic ----------------
  authSubmit.onclick = async function(){
    const u = authUsername.value.trim();
    const p = authPassword.value.trim();
    if(!u || !p) return alert("username and password required");
    // login
    try {
      const res = await fetch("/login", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({username:u,password:p})});
      if(!res.ok){ const j = await res.json(); return alert(j.detail || "login failed"); }
      const body = await res.json();
      // close anon socket
      try { if(anonSocket) anonSocket.close(); } catch(e){}
      connectAuth(body.token, u);
      authModal.style.display = "none";
    } catch(e){
      console.error("login error", e);
      alert("Network error");
    }
  };
  continueGuest.onclick = () => { authModal.style.display = "none"; headerUser.textContent = "Viewing as guest"; };

  // ---------------- send flow ----------------
  sendBtn.addEventListener("click", () => {
    const text = messageInput.value.trim();
    if(!text) return;
    if(authSocket && authSocket.readyState === WebSocket.OPEN){
      authSocket.send(JSON.stringify({type:"message", name: username, text: text}));
      messageInput.value = "";
    } else {
      mustAuthPopup.style.display = "flex";
      setTimeout(()=> { mustAuthPopup.style.display = "none"; }, 3500);
    }
  });
  messageInput.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey){
      e.preventDefault();
      sendBtn.click();
    }
  });

  window.addEventListener("beforeunload", () => {
    try { if(authSocket && authSocket.readyState === WebSocket.OPEN) authSocket.close(); } catch(e){}
  });

  // init
  connectAnon();
  authModal.style.display = "flex";
  authUsername.focus();
})();
