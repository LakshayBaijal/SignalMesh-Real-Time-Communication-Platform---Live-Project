// robust mobile scrolling: pointer-event fallback + fixed composer without transform
(() => {
  const host = location.host;
  const protocol = (location.protocol === "https:") ? "wss://" : "ws://";
  const anonWsUrl = protocol + host + "/ws";

  // DOM refs
  const usersListEl = document.getElementById("users-list");
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const typingIndicator = document.getElementById("typing-indicator");
  const headerUser = document.getElementById("header-username");
  const fileInput = document.getElementById("file-input");
  const attachBtn = document.getElementById("attach-btn");
  const emojiBtn = document.getElementById("emoji-btn");
  const emojiPickerDiv = document.getElementById("emoji-picker");
  const usersToggle = document.getElementById("users-toggle");
  const sidebar = document.getElementById("sidebar");

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

  // state
  let anonSocket = null;
  let authSocket = null;
  let username = null;
  let token = null;
  let mode = "login";
  let users = {};

  // helpers
  function nowTime(){ const d = new Date(); return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }
  function randomColorFromName(name){ const palette = ["#ef4444","#f97316","#f59e0b","#10b981","#06b6d4","#3b82f6","#8b5cf6","#ec4899"]; let h=0; for(let i=0;i<name.length;i++) h=(h*31+name.charCodeAt(i))|0; return palette[Math.abs(h)%palette.length]; }
  function escapeHtml(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;"); }

  // autosize textarea
  function autosizeTextarea(el){
    el.style.height = "auto";
    const max = 180;
    el.style.height = Math.min(max, el.scrollHeight) + "px";
  }

  // reliable scroll to bottom (small delay to allow layout)
  function scrollToBottom(behavior='auto'){
    setTimeout(()=> {
      try { chatBox.scrollTo({ top: chatBox.scrollHeight, behavior }); }
      catch(e) { chatBox.scrollTop = chatBox.scrollHeight; }
    }, 60);
  }

  // render users
  function renderUsers(){
    usersListEl.innerHTML = "";
    Object.values(users).forEach(u=>{
      const li = document.createElement("li");
      li.innerHTML = `<div class="avatar" style="background:${u.color}">${(u.name||"U").slice(0,2).toUpperCase()}</div>
                      <div><div style="font-weight:600">${u.name}</div><div style="font-size:12px;color:var(--muted)">${u.lastSeen||"online"}</div></div>`;
      usersListEl.appendChild(li);
    });
  }

  // append message
  function addRawMessage(name, text, mine=false){
    const row = document.createElement("div");
    row.className = "msg-row " + (mine ? "me" : "other");

    const bubble = document.createElement("div");
    bubble.className = "msg " + (mine ? "me" : "other");

    const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = `${name} • ${nowTime()}`;
    const content = document.createElement("div"); content.className = "content"; content.innerHTML = text || "";

    bubble.appendChild(meta); bubble.appendChild(content); row.appendChild(bubble);
    chatBox.appendChild(row);

    scrollToBottom('smooth');
  }

  function addSystemNotice(text){
    const e = document.createElement("div"); e.style.textAlign="center"; e.style.color="var(--muted)"; e.style.fontSize="13px"; e.textContent=text;
    chatBox.appendChild(e); scrollToBottom('auto');
  }

  // parse server messages
  function handleServerData(data){
    let obj = null;
    try { obj = JSON.parse(data); } catch(e){ obj = null; }

    if(obj && obj.type === "user_count"){
      document.getElementById("user-count").textContent = `Online: ${obj.auth} users`;
      document.getElementById("viewer-count").textContent = `Viewers: ${obj.total} total`;
      return;
    }

    if(obj && obj.type){
      if(obj.type === "join"){
        users[obj.name] = {name: obj.name, color: obj.color || randomColorFromName(obj.name), lastSeen: "online"};
        renderUsers(); addSystemNotice(`${obj.name} joined`);
      } else if(obj.type === "leave"){
        if(users[obj.name]) users[obj.name].lastSeen = "left";
        renderUsers(); addSystemNotice(`${obj.name} left`);
      } else if(obj.type === "typing"){
        typingIndicator.textContent = `${obj.name} is typing…`;
        clearTimeout(window._typingTimeout); window._typingTimeout = setTimeout(()=> typingIndicator.textContent = "", 1400);
      } else if(obj.type === "message"){
        const mine = username && obj.name === username;
        addRawMessage(obj.name, escapeHtml(obj.text), mine);
      } else if(obj.type === "file"){
        const mine = username && obj.name === username;
        const safeUrl = escapeHtml(obj.url || "");
        const safeFilename = escapeHtml(obj.filename || "file");
        const linkHtml = `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">📎 ${safeFilename}</a>`;
        addRawMessage(obj.name, linkHtml, mine);
      } else {
        addRawMessage("server", escapeHtml(JSON.stringify(obj)));
      }
    } else {
      const s = String(data); const sep = s.indexOf(":");
      if(sep > 0){ const n = s.slice(0,sep).trim(); const t = s.slice(sep+1).trim(); addRawMessage(n, escapeHtml(t), false); }
      else addRawMessage("server", escapeHtml(s), false);
    }
  }

  // connect anon socket
  function connectAnon(){
    try { anonSocket = new WebSocket(anonWsUrl); } catch(e){ console.error("WS error", e); return; }
    anonSocket.addEventListener("open", ()=>{ headerUser.textContent = "Viewing as guest"; });
    anonSocket.addEventListener("message", ev => handleServerData(ev.data));
    anonSocket.addEventListener("close", ()=>{ console.log("anon socket closed"); });
  }

  // connect auth socket
  function connectAuth(tkn, name){
    token = tkn; username = name;
    const url = protocol + host + "/ws/" + token;
    try { authSocket = new WebSocket(url); } catch(e){ console.error(e); return; }

    authSocket.addEventListener("open", ()=>{ headerUser.textContent = `You: ${username}`; });
    authSocket.addEventListener("message", ev => handleServerData(ev.data));
    authSocket.addEventListener("close", ()=>{ headerUser.textContent = "Disconnected — viewing as guest"; authSocket = null; if(!anonSocket || anonSocket.readyState !== WebSocket.OPEN) connectAnon(); });
  }

  // auth UI
  function setMode(m){
    mode = m;
    toggleLogin.classList.toggle('active', m==='login');
    toggleSignup.classList.toggle('active', m==='signup');
    authTitle.textContent = m==='login' ? "Login to World Chat" : "Create an account";
    authError.textContent = "";
  }
  toggleLogin.onclick = ()=> setMode('login');
  toggleSignup.onclick = ()=> setMode('signup');

  async function doAuth(){
    const u = authUsername.value.trim(); const p = authPassword.value.trim();
    if(!u||!p){ authError.textContent="username and password required"; return; }
    authError.textContent = "";
    try {
      if(mode === "signup"){
        const res = await fetch("/signup", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({username:u,password:p})});
        if(!res.ok){ const err = await res.json(); authError.textContent = err.detail || "Signup failed"; return; }
      }
      const res2 = await fetch("/login", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({username:u,password:p})});
      if(!res2.ok){ const err = await res2.json(); authError.textContent = err.detail || "Login failed"; return; }
      const body = await res2.json(); const tok = body.token;
      try { if(anonSocket){ anonSocket.close(); anonSocket = null; } } catch(e){}
      connectAuth(tok, u);
      authModal.style.display = "none";
    } catch(e){ authError.textContent="Network error"; console.error(e); }
  }

  authSubmit.onclick = doAuth;
  authPassword.addEventListener("keydown", (e)=>{ if(e.key==="Enter") doAuth(); });
  continueGuest.onclick = ()=>{ authModal.style.display = "none"; headerUser.textContent = "Viewing as guest"; }

  // file upload/attach
  attachBtn.addEventListener("click", ()=> fileInput.click());
  fileInput.addEventListener("change", async ()=>{
    const file = fileInput.files[0]; if(!file) return;
    const form = new FormData(); form.append("file", file);
    try {
      const res = await fetch("/upload", {method:"POST", body: form});
      if(!res.ok){ console.error("upload failed"); alert("Upload failed"); return; }
      const data = await res.json();
      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        authSocket.send(JSON.stringify({type:"file", name: username, url: data.url, filename: data.filename}));
      } else {
        alert("Sign in to send attachments");
      }
    } catch(e){ console.error("upload error", e); alert("Upload error"); }
    finally { fileInput.value = ""; }
  });

  // emoji picker
  let pickerOpen = false;
  emojiBtn.onclick = ()=>{
    if(pickerOpen){ emojiPickerDiv.style.display = "none"; pickerOpen = false; return; }
    emojiPickerDiv.innerHTML = "";
    try {
      const picker = new EmojiMart.Picker({
        onEmojiSelect: (emoji)=>{ messageInput.value += (emoji.native || emoji.colons || emoji.native); autosizeTextarea(messageInput); messageInput.focus(); },
        theme: "auto", perLine: 8, showPreview: false, showSkinTones: false
      });
      emojiPickerDiv.appendChild(picker); emojiPickerDiv.style.display = "block"; pickerOpen = true;
    } catch(err){ messageInput.value += "😊"; autosizeTextarea(messageInput); }
  };

  // send message
  function sendMessage(){
    const text = messageInput.value.trim();
    if(!text) return;
    if(text.length > 5000){ alert("Message too long (limit 5000)"); return; }
    if(authSocket && authSocket.readyState === WebSocket.OPEN){
      authSocket.send(JSON.stringify({type:"message", name: username, text}));
      messageInput.value = ""; autosizeTextarea(messageInput);
    } else {
      alert("Sign in to send messages");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  messageInput.addEventListener("input", ()=> autosizeTextarea(messageInput));
  messageInput.addEventListener("keydown", (e)=>{
    if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); }
    else {
      if(authSocket && authSocket.readyState === WebSocket.OPEN){
        try { authSocket.send(JSON.stringify({type:"typing", name: username})); } catch(e){}
      }
    }
  });

  // mobile sidebar toggle
  usersToggle.addEventListener("click", ()=> sidebar.classList.toggle("open"));
  document.addEventListener("click", (ev)=>{
    if(window.innerWidth <= 880){
      if(!sidebar.contains(ev.target) && !usersToggle.contains(ev.target)){
        sidebar.classList.remove("open");
      }
    }
  });

  // adjust for visual viewport
  function adjustForVisualViewport(){
    if(window.visualViewport){
      const vv = window.visualViewport;
      vv.addEventListener('resize', ()=> { setTimeout(()=> scrollToBottom('auto'), 80); });
      vv.addEventListener('scroll', ()=> { setTimeout(()=> scrollToBottom('auto'), 80); });
    } else {
      messageInput.addEventListener('focus', ()=> setTimeout(()=> scrollToBottom('smooth'), 250));
    }
  }
  adjustForVisualViewport();

  // Pointer/touch fallback for dragging scroll inside chatBox
  (function enablePointerDragScroll(){
    // prefer Pointer Events
    if(window.PointerEvent){
      let active = false, startY = 0, startScroll = 0, activeId = null;
      chatBox.addEventListener('pointerdown', function(e){
        if(e.isPrimary === false) return;
        active = true; activeId = e.pointerId;
        startY = e.clientY; startScroll = chatBox.scrollTop;
        try { chatBox.setPointerCapture(activeId); } catch(_){}
      }, {passive:false});

      chatBox.addEventListener('pointermove', function(e){
        if(!active || e.pointerId !== activeId) return;
        const dy = e.clientY - startY;
        chatBox.scrollTop = startScroll - dy;
        e.preventDefault();
      }, {passive:false});

      chatBox.addEventListener('pointerup', function(e){
        if(e.pointerId !== activeId) return;
        active = false; try { chatBox.releasePointerCapture(e.pointerId); } catch(_) {}
      }, {passive:true});

      chatBox.addEventListener('pointercancel', function(e){
        active = false; try { chatBox.releasePointerCapture(e.pointerId); } catch(_) {}
      }, {passive:true});
    } else {
      // touch fallback
      let touching = false, startY = 0, startScroll = 0;
      chatBox.addEventListener('touchstart', (e) => {
        if(e.touches.length !== 1) return;
        touching = true; startY = e.touches[0].clientY; startScroll = chatBox.scrollTop;
      }, {passive:true});

      chatBox.addEventListener('touchmove', (e) => {
        if(!touching || e.touches.length !== 1) return;
        const dy = e.touches[0].clientY - startY;
        chatBox.scrollTop = startScroll - dy;
      }, {passive:true});

      chatBox.addEventListener('touchend', () => { touching = false; }, {passive:true});
      chatBox.addEventListener('touchcancel', () => { touching = false; }, {passive:true});
    }
  })();

  // ===== Additional mobile fixes requested (keyboard/resize handling) =====
  // Ensure chat auto-scrolls when the viewport resizes (keyboard open/close) and on input focus.
  window.addEventListener("resize", () => {
    setTimeout(() => {
      try { chatBox.scrollTop = chatBox.scrollHeight; } catch(e){}
    }, 200);
  });

  messageInput.addEventListener("focus", () => {
    setTimeout(() => {
      try { chatBox.scrollTop = chatBox.scrollHeight; } catch(e){}
    }, 300);
  });
  // ======================================================================

  // close sockets on unload
  window.addEventListener("beforeunload", ()=> { try { if(authSocket && authSocket.readyState === WebSocket.OPEN) authSocket.close(); } catch(e){} });

  // init
  connectAnon();
  authModal.style.display = "flex";
  authUsername.focus();
  autosizeTextarea(messageInput);

  // expose for debug
  window.signalmesh = { addRawMessage, scrollToBottom };
})();
