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

  const authModal = document.getElementById("auth-modal");
  const authMode = document.getElementById("auth-mode");
  const authUsername = document.getElementById("auth-username");
  const authPassword = document.getElementById("auth-password");
  const authSubmit = document.getElementById("auth-submit");
  const authSwitch = document.getElementById("auth-switch");
  const authError = document.getElementById("auth-error");
  const mustAuthPopup = document.getElementById("must-auth");

  const fileInput = document.getElementById("file-input");

  let anonSocket = null;
  let authSocket = null;
  let username = null;
  let token = null;
  let color = null;

  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, (m) => {
      return { "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[m];
    });
  }

  function randomColorFromName(n){
    let sum = 0;
    for(let i=0;i<n.length;i++) sum += n.charCodeAt(i);
    const colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"];
    return colors[sum % colors.length];
  }

  function addRawMessage(name, text, mine){
    const el = document.createElement("div");
    el.className = "msg " + (mine ? "mine" : "theirs");
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = name;
    const body = document.createElement("div");
    body.className = "body";
    body.innerHTML = text;
    el.appendChild(who);
    el.appendChild(body);
    chatBox.appendChild(el);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function showSuggestions(list){
    const sug = document.getElementById("suggestions");
    if(!sug) return;
    sug.innerHTML = "";
    list.forEach(s => {
      const b = document.createElement("button");
      b.className = "sugg";
      b.textContent = s;
      b.onclick = () => {
        messageInput.value = s;
        messageInput.focus();
      };
      sug.appendChild(b);
    });
  }

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

    if(obj && typeof obj === "object"){
      if(obj.type === "typing"){
        typingIndicator.textContent = `${obj.name} is typing...`;
        window._typingTimeout && clearTimeout(window._typingTimeout);
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
      // server will announce join
    });

    authSocket.addEventListener("message", (ev) => {
      handleServerData(ev.data);
    });

    authSocket.addEventListener("close", () => {
      console.log("auth socket closed");
      try {
        // fallback to anonymous socket
        if(!anonSocket || anonSocket.readyState !== WebSocket.OPEN) connectAnon();
      } catch(e){}
    });
  }

  // ---------------- UI / auth ----------------
  async function doAuth(){
    const mode = authMode.textContent.trim();
    const u = authUsername.value.trim();
    const p = authPassword.value;
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
  authPassword.addEventListener("keyup", (e) => { if(e.key === "Enter") doAuth(); });
  authSwitch.onclick = () => {
    if(authMode.textContent.trim() === "signup"){
      authMode.textContent = "login";
      authSwitch.textContent = "create account";
    } else {
      authMode.textContent = "signup";
      authSwitch.textContent = "signin";
    }
  };

  // ---------------- sending messages ----------------
  sendBtn.onclick = async () => {
    const txt = messageInput.value.trim();
    if(!txt) return;
    messageInput.value = "";
    if(authSocket && authSocket.readyState === WebSocket.OPEN){
      authSocket.send(JSON.stringify({type:"message", text: txt}));
    } else {
      // if not authenticated, send to anon socket (server will treat as guest)
      if(anonSocket && anonSocket.readyState === WebSocket.OPEN){
        anonSocket.send(txt);
      } else {
        alert("Not connected");
      }
    }
  };

  messageInput.addEventListener("input", () => {
    if(authSocket && authSocket.readyState === WebSocket.OPEN){
      authSocket.send(JSON.stringify({type:"typing"}));
    }
  });

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
      console.error(e);
    }
  });

  // beforeunload cleanup
  window.addEventListener("beforeunload", () => {
    try {
      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        authSocket.close();
      }
    } catch(e){}
  });

  // init
  connectAnon();
  authModal.style.display = "flex";
  authUsername.focus();
})();
