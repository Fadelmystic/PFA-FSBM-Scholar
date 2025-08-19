let mode = 'rag';
const RAG_URL = 'http://localhost:8000/chat';
const KG_URL  = 'http://localhost:8001/chat';

const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const btnRag = document.getElementById('btnRag');
const btnKg = document.getElementById('btnKg');

btnRag.onclick = () => switchMode('rag');
btnKg.onclick = () => switchMode('kg');

function switchMode(next){
  mode = next;
  btnRag.classList.toggle('active', mode==='rag');
  btnKg.classList.toggle('active', mode==='kg');
  addBot(`Mode: ${mode.toUpperCase()} sélectionné.`);
}

function addUser(text){
  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight;
}

function addBot(text){
  const el = document.createElement('div');
  el.className = 'message bot';
  el.innerHTML = `<div class="avatar">FS</div><div class="bubble">${linkify(text)}</div>`;
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight;
}

sendBtn.onclick = async () => {
  const text = input.value.trim();
  if (!text) return;
  addUser(text); input.value = ''; sendBtn.disabled = true;
  try{
    const url = mode==='rag' ? RAG_URL : KG_URL;
    const res = await fetch(url, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: text })
    });
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const response = data.response || JSON.stringify(data);
    addBot(response);
  }catch(e){
    addBot(`Erreur lors de l'appel ${mode.toUpperCase()}. Vérifiez que l'API est démarrée.`);
  }finally{
    sendBtn.disabled = false; input.focus();
  }
}

input.addEventListener('keydown', (e)=>{
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault(); sendBtn.click();
  }
});

function escapeHtml(s){
  return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function linkify(s){
  return escapeHtml(s).replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank">$1</a>');
}


