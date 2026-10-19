"""WhatsApp-style streaming chat for the original torchtune / Llama 3 8B AICLONE.

Run from the LLM_testing project root:
    python chat_app.py --config config/llama3-8B/inference_config.yaml

The server listens on loopback only. For a remote machine, connect with an SSH
port-forward rather than exposing the unauthenticated service publicly.

Cache-rebuild fallback for inconsistent/legacy torchtune builds.
Compatible with the legacy torchtune API used by the original chat_app.py:
    torchtune.utils.generate_next_token(model, input_pos, x, temperature, top_k)
"""

import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import uvicorn
from fastapi import Cookie, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from torchtune import config, utils
from torchtune.data._types import Message

PORT = 7860
HOST = "127.0.0.1"  # SSH-tunnel-friendly; do not expose without authentication
SESSION_COOKIE = "aiclone_session"


@dataclass
class ChatSession:
    history: List[Message] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel: threading.Event = field(default_factory=threading.Event)
    busy: bool = False


class ChatModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = utils.get_device(device=cfg.device)
        self.dtype = utils.get_dtype(dtype=cfg.dtype)
        utils.set_seed(seed=cfg.seed)

        ckpt = config.instantiate(cfg.checkpointer).load_checkpoint()
        with utils.set_default_dtype(self.dtype), self.device:
            self.model = config.instantiate(cfg.model)
        self.model.load_state_dict(ckpt[utils.MODEL_KEY])
        with self.device:
            self.model.setup_caches(max_batch_size=1, dtype=self.dtype)
        self.model.eval()
        self.tokenizer = config.instantiate(cfg.tokenizer)
        print(f"[AICLONE] Model ready; torchtune streaming API: "
              f"generate_next_token={hasattr(utils, 'generate_next_token')}; "
              f"reset_caches={hasattr(self.model, 'reset_caches')}", flush=True)

        with open(cfg.system_prompt_file, encoding="utf-8") as handle:
            self.system_prompt = handle.read().strip()
        self.gpu_lock = threading.Lock()  # a single model/cache, one generation at a time
        self.sessions: Dict[str, ChatSession] = {}
        self.sessions_lock = threading.Lock()

    def get_session(self, session_id: str) -> ChatSession:
        with self.sessions_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatSession(
                    history=[Message(role="system", content=self.system_prompt)]
                )
            return self.sessions[session_id]

    def _prepare_prompt(self, messages: List[Message]):
        # Reserve room for the response; drop the oldest *complete* chat turns.
        limit = min(int(self.cfg.get("max_seq_len", 8192)), self.model.max_seq_len)
        output_limit = int(self.cfg.max_new_tokens)
        budget = limit - output_limit
        if budget < 64:
            raise ValueError("max_seq_len must exceed max_new_tokens by at least 64 tokens")
        current = list(messages)
        while True:
            tokens, _ = self.tokenizer.tokenize_messages(current, max_seq_len=limit)
            if tokens and tokens[-1] == 128001:  # legacy Llama 3 end-of-text
                tokens = tokens[:-1]
            if len(tokens) <= budget:
                return torch.tensor(tokens, dtype=torch.int, device=self.device)
            # Keep system + current user; remove oldest user/assistant exchange.
            if len(current) <= 2:
                raise ValueError("Message is too long for the context window")
            current = [current[0]] + current[3:]

    @staticmethod
    def _visible_text(raw: str) -> str:
        # Older torchtune tokenizers may decode Llama role markers literally.
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in raw:
            raw = raw.rsplit(marker, 1)[1]
        elif raw.startswith("<|start_header_id|>") and "<|end_header_id|>" not in raw:
            return ""
        raw = re.sub(r"<\|(?:eot_id|end_of_text|end_header_id|start_header_id)\|>", "", raw)
        raw = re.sub(r"<\|[^>]*$", "", raw)  # hide incomplete special token
        return raw.lstrip("\n ")

    @torch.inference_mode()
    def _next_token(self, input_pos, x, temperature: float):
        return utils.generate_next_token(
            self.model,
            input_pos=input_pos,
            x=x,
            temperature=temperature,
            top_k=self.cfg.top_k,
        )

    def stream_reply(self, messages: List[Message], temperature: float, cancel: threading.Event):
        """Yield the full decoded response as it changes, one or a few tokens at a time."""
        prompt = self._prepare_prompt(messages)
        limit = min(int(self.cfg.get("max_seq_len", 8192)), self.model.max_seq_len)
        max_output = min(int(self.cfg.max_new_tokens), limit - prompt.numel())
        if max_output < 1:
            raise ValueError("There is no room left to generate a reply")

        # Recreate KV caches for this turn using the *same* API that successfully
        # initialized the model. Legacy/custom TransformerDecoders may expose
        # setup_caches() but lack caches_are_enabled() / reset_caches().
        # This is slower than an in-place reset but avoids assumptions about
        # cache implementation and ensures each request starts from empty state.
        with self.device:
            self.model.setup_caches(max_batch_size=1, dtype=self.dtype)

        input_pos = torch.arange(self.model.max_seq_len, device=self.device)
        prompt_length = prompt.numel()
        x = prompt.unsqueeze(0)
        out_tokens: List[int] = []
        last_text = ""
        stop_ids = set(self.tokenizer.stop_tokens or [])
        # setup_caches() was called at model initialization. Not every
        # TransformerDecoder version exposes caches_are_enabled().
        # Matches torchtune.utils.generate(): prompt [0..L-1] at prefill,
        # then feed each newly sampled token at L, L+1, L+2... respectively.
        for step in range(max_output):
            if cancel.is_set():
                break
            if step == 0:
                pos = input_pos[:prompt_length]
            else:
                new_token_position = prompt_length + step - 1
                pos = input_pos[new_token_position:new_token_position + 1]
            token = self._next_token(pos, x, temperature)
            if isinstance(token, tuple):  # Some builds return (token, logits).
                token = token[0]
            token_id = int(token.reshape(-1)[0].item())
            if token_id in stop_ids:
                break
            out_tokens.append(token_id)

            # Decode entire generated prefix to handle UTF-8 and subword joins.
            decoded = self.tokenizer.decode(out_tokens, truncate_at_eos=False)
            visible = self._visible_text(decoded)
            if visible != last_text:
                last_text = visible
                yield visible

            x = token.reshape(1, 1)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=20000)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    regenerate: bool = False


def as_event(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def build_app(model: ChatModel) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def lookup(session_id: str):
        if not session_id or session_id not in model.sessions:
            raise HTTPException(status_code=401, detail="Open the chat page again")
        return model.get_session(session_id)

    @app.get("/", response_class=HTMLResponse)
    def index(session_id: str = Cookie(default=None, alias=SESSION_COOKIE)):
        if not session_id or session_id not in model.sessions:
            session_id = secrets.token_urlsafe(32)
            model.get_session(session_id)
        result = HTMLResponse(PAGE)
        result.set_cookie(
            SESSION_COOKIE, session_id, httponly=True, samesite="strict", secure=False
        )
        return result

    @app.get("/history")
    def history(session_id: str = Cookie(default=None, alias=SESSION_COOKIE)):
        session = lookup(session_id)
        with session.lock:
            return {
                "messages": [
                    {"role": item.role, "content": item.content}
                    for item in session.history if item.role != "system"
                ],
                "busy": session.busy,
            }

    @app.post("/chat")
    def chat(req: ChatRequest, session_id: str = Cookie(default=None, alias=SESSION_COOKIE)):
        session = lookup(session_id)
        with session.lock:
            if session.busy:
                return JSONResponse({"error": "A reply is already being generated"}, status_code=409)
            if req.regenerate:
                if len(session.history) < 3 or session.history[-1].role != "assistant":
                    return JSONResponse({"error": "No reply to regenerate"}, status_code=400)
                user_message = session.history[-2].content
                prefix = list(session.history[:-2])
            else:
                user_message = req.message.strip()
                if not user_message:
                    return JSONResponse({"error": "Message is empty"}, status_code=400)
                prefix = list(session.history)
            session.busy = True
            session.cancel.clear()

        def generate_events():
            latest = ""
            try:
                # Only one request can use the shared model / cache at a time.
                yield as_event({"type": "status", "value": "typing"})
                with model.gpu_lock:
                    for latest in model.stream_reply(
                        prefix + [Message(role="user", content=user_message)],
                        req.temperature, session.cancel,
                    ):
                        yield as_event({"type": "snapshot", "text": latest})
                stopped = session.cancel.is_set()
                if latest:
                    with session.lock:
                        session.history = prefix + [
                            Message(role="user", content=user_message),
                            Message(role="assistant", content=latest),
                        ]
                yield as_event({"type": "done", "stopped": stopped, "text": latest})
            except Exception as exc:
                import traceback
                traceback.print_exc()  # visible in the terminal for diagnosis
                yield as_event({"type": "error", "message": str(exc)})
            finally:
                with session.lock:
                    session.busy = False
                    session.cancel.clear()

        return StreamingResponse(
            generate_events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/stop")
    def stop(session_id: str = Cookie(default=None, alias=SESSION_COOKIE)):
        session = lookup(session_id)
        session.cancel.set()
        return {"ok": True}

    @app.post("/reset")
    def reset(session_id: str = Cookie(default=None, alias=SESSION_COOKIE)):
        session = lookup(session_id)
        with session.lock:
            if session.busy:
                return JSONResponse({"error": "Stop the response before starting a new chat"}, status_code=409)
            session.history = [Message(role="system", content=model.system_prompt)]
            session.cancel.clear()
        return {"ok": True}

    return app


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AICLONE · Pierre</title>
<style>
:root { --page:#dbe3e6; --header:#f0f2f5; --paper:#e9e4dc; --bubble-me:#d9fdd3;
  --bubble-them:#fff; --ink:#111b21; --muted:#667781; --green:#00a884;
  --line:#d1d7db; --composer:#f0f2f5; --field:#fff; --hover:#e4e9ec; }
:root[data-theme="dark"] { --page:#0c1317; --header:#202c33; --paper:#0b141a;
  --bubble-me:#005c4b; --bubble-them:#202c33; --ink:#e9edef; --muted:#8696a0;
  --green:#00a884; --line:#2a3942; --composer:#202c33; --field:#2a3942; --hover:#34434b; }
* {box-sizing:border-box} [hidden]{display:none!important} html,body{height:100%; margin:0}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--page); display:flex; align-items:center; justify-content:center}
button,textarea,input{font:inherit}button{cursor:pointer}button:disabled{opacity:.42;cursor:not-allowed}
.app{width:min(820px,100vw); height:min(94dvh,960px); display:flex; flex-direction:column;
  overflow:hidden; border-radius:13px; background:var(--paper); box-shadow:0 12px 60px #0003}
.header{flex-shrink:0; min-height:70px; background:var(--header); display:flex;
  align-items:center; padding:10px 18px; gap:12px; border-bottom:1px solid var(--line)}
.avatar{display:grid;place-items:center; width:43px;height:43px;border-radius:50%;
  background:linear-gradient(135deg,#1c9a81,#075e54); color:white;font-size:19px;font-weight:700}
.who{min-width:0;flex:1}.who strong{font-size:16px;font-weight:650}
.status{display:flex; align-items:center;gap:6px;font-size:12px;color:var(--muted);margin-top:2px}
.status-dot{width:7px;height:7px;border-radius:50%;background:#20ba75}
.icon-btn{width:37px;height:37px;border:0; border-radius:50%; background:transparent;
  color:var(--muted); display:grid;place-items:center;font-size:19px}
.icon-btn:hover{background:var(--hover)}.icon-btn.small{width:29px;height:29px;font-size:15px}
.tooltip{position:relative}.tooltip[data-title]:hover::after{content:attr(data-title);position:absolute;
  top:100%;left:50%;transform:translateX(-50%);white-space:nowrap;background:#263238;
  color:white;padding:5px 8px;border-radius:6px;font-size:11px;z-index:10}
.chat{flex:1;min-height:0; overflow-y:auto;padding:16px 8% 22px;scroll-behavior:smooth;
  background-color:var(--paper);
  background-image:radial-gradient(circle at 20px 15px,#7f8e8b0b 1.2px,transparent 1.2px),
   radial-gradient(circle at 70px 70px,#6f8c820d 1px,transparent 1px);
  background-size:110px 110px}
.day{width:max-content;max-width:95%;margin:1px auto 24px;padding:5px 12px;
  border-radius:8px;background:var(--header);color:var(--muted);font-size:11px;box-shadow:0 1px 1px #0001}
.empty{min-height:60%;display:flex;align-items:center;justify-content:center;flex-direction:column;
  text-align:center;gap:8px;color:var(--muted)}.empty .big{font-size:29px}
.row{display:flex;margin:0 0 8px;animation:pop .16s ease-out}
.row.me{justify-content:flex-end}.row.them{justify-content:flex-start}
@keyframes pop{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.bubble{position:relative;max-width:min(78%,590px);min-width:65px;
  border-radius:10px; padding:9px 10px 6px;background:var(--bubble-them);
  box-shadow:0 1px 1px #0002;overflow-wrap:anywhere}
.row.me .bubble{background:var(--bubble-me);border-top-right-radius:2px}
.row.them .bubble{border-top-left-radius:2px}
.content{white-space:pre-wrap;word-break:break-word;font-size:14.5px;line-height:1.44}
.meta{display:flex;justify-content:flex-end;align-items:center;gap:5px;color:var(--muted);
  font-size:10px;margin-top:3px;user-select:none}.check{font-size:12px}
.streaming .content::after{content:'▎';color:var(--green);animation:blink .75s infinite}
@keyframes blink{50%{opacity:0}}.typing-bubble{padding:12px 16px}
.dots{display:flex;gap:5px;align-items:center;min-height:15px}.dots i{width:7px;height:7px;
  border-radius:50%;background:var(--muted);animation:bounce 1.1s infinite}
.dots i:nth-child(2){animation-delay:.15s}.dots i:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,55%,100%{transform:translateY(0);opacity:.4}30%{transform:translateY(-4px);opacity:1}}
.msg-tools{position:absolute;right:0;top:-27px;display:flex;gap:2px;justify-content:flex-end;
  padding:2px 4px;border-radius:6px;background:var(--header);box-shadow:0 1px 5px #0002;
  opacity:0;pointer-events:none;transition:opacity .15s}
.row.them:hover .msg-tools,.row.them:focus-within .msg-tools{opacity:1;pointer-events:auto}
.msg-tools button{border:0;background:transparent;color:var(--muted);font-size:11px;padding:4px 6px;border-radius:5px}
.msg-tools button:hover{background:var(--hover)}
.composer{flex-shrink:0;background:var(--composer);border-top:1px solid var(--line);
  display:flex;align-items:flex-end;gap:10px;padding:11px 14px max(11px,env(safe-area-inset-bottom))}
.input-wrap{flex:1; min-width:0;background:var(--field);border-radius:23px;display:flex;align-items:center;
  padding:4px 14px;min-height:46px}
textarea{width:100%;background:transparent;color:var(--ink);border:0;outline:0;
  resize:none;min-height:25px;max-height:140px;line-height:1.45;font-size:14.5px;
  padding:6px 1px;overflow-y:auto}
textarea::placeholder{color:var(--muted)}
.circle{flex-shrink:0;display:grid;place-items:center;width:45px;height:45px;
  border:0;border-radius:50%;background:var(--green);color:white;font-size:19px}
.circle:hover{filter:brightness(.93)}.circle.stop{background:#db5656}
.settings{display:flex;align-items:center;gap:7px;background:var(--header);
  color:var(--muted);padding:7px 18px;font-size:11px;border-bottom:1px solid var(--line)}
.settings input{accent-color:var(--green);width:110px}.settings output{min-width:23px}
.right{margin-left:auto}.hint{color:var(--muted);font-size:11px}
.toast{position:absolute;bottom:95px;left:50%;transform:translateX(-50%);
  background:#263238;color:white;padding:9px 14px;border-radius:8px;font-size:12px;display:none;z-index:20}
@media(max-width:600px){body{display:block}.app{height:100dvh;border-radius:0;width:100vw;box-shadow:none}
 .header{padding:9px 11px;gap:9px}.chat{padding:13px 4%}.bubble{max-width:88%}
 .settings{padding:6px 12px}.hint{display:none}.composer{padding:8px 9px max(9px,env(safe-area-inset-bottom))}}
@media(hover:none){.msg-tools{opacity:1}}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="avatar">P</div>
    <div class="who"><strong>Pierre</strong><div class="status"><span class="status-dot"></span><span id="status">online</span></div></div>
    <button class="icon-btn tooltip" data-title="Regenerate last reply" id="regenerate" aria-label="Regenerate last reply">↻</button>
    <button class="icon-btn tooltip" data-title="Save conversation" id="export" aria-label="Save conversation">⇩</button>
    <button class="icon-btn tooltip" data-title="Toggle dark mode" id="theme" aria-label="Toggle dark mode">☾</button>
    <button class="icon-btn tooltip" data-title="New conversation" id="reset" aria-label="New conversation">✎</button>
  </header>
  <div class="settings"><span>Creativity</span><input id="temp" type="range" min="0.1" max="1.5" step="0.1" value="0.8">
    <output id="temp-val">0.8</output><span class="right hint">Enter to send · Shift+Enter for new line</span></div>
  <main class="chat" id="chat" aria-live="polite"><div class="day">Today</div>
    <div class="empty" id="empty"><span class="big">💬</span><strong>Start a conversation</strong><span>Message Pierre to get started.</span></div>
  </main>
  <footer class="composer"><div class="input-wrap"><textarea id="input" rows="1" placeholder="Message Pierre" aria-label="Message"></textarea></div>
    <button id="send" class="circle" aria-label="Send message">➤</button>
    <button id="stop" class="circle stop" aria-label="Stop generating" hidden>■</button>
  </footer>
</div>
<div class="toast" id="toast"></div>
<script>
'use strict';
const $ = id => document.getElementById(id);
const chat = $('chat'), field = $('input');
let busy = false, messages = [], activeRow = null, typingRow = null, toastTimer;
const savedTheme = localStorage.getItem('aiclone-theme');
if(savedTheme === 'dark') document.documentElement.dataset.theme = 'dark';

function clock(){return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});}
function scrollBottom(){chat.scrollTop = chat.scrollHeight;}
function toast(msg){const t=$('toast');t.textContent=msg;t.style.display='block';clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>t.style.display='none',2600);}
function status(msg){$('status').textContent=msg;}
function setBusy(value){busy=value;$('send').hidden=value;$('stop').hidden=!value;
  $('regenerate').disabled=value;$('reset').disabled=value;field.disabled=value;
  if(!value){status('online');field.focus();}}
function emptyHint(){const e=$('empty');if(e)e.remove();}
function addBubble(text,role,time=clock()){
  emptyHint();
  const row=document.createElement('div');row.className='row '+(role==='user'?'me':'them');
  const bubble=document.createElement('div');bubble.className='bubble';
  const content=document.createElement('div');content.className='content';content.textContent=text;
  const meta=document.createElement('div');meta.className='meta';
  const stamp=document.createElement('span');stamp.textContent=time;meta.appendChild(stamp);
  if(role==='user'){const tick=document.createElement('span');tick.className='check';tick.textContent='✓';meta.appendChild(tick);}
  bubble.append(content,meta);
  if(role==='assistant'){
    const tools=document.createElement('div');tools.className='msg-tools';
    const copy=document.createElement('button');copy.textContent='Copy';copy.title='Copy this reply';
    copy.onclick=()=>navigator.clipboard.writeText(content.textContent).then(()=>toast('Copied')).catch(()=>toast('Copy unavailable'));
    tools.appendChild(copy);bubble.appendChild(tools);
  }
  row.appendChild(bubble);chat.appendChild(row);scrollBottom();
  return {row,bubble,content};
}
function startTyping(){
  typingRow=document.createElement('div');typingRow.className='row them';
  const bubble=document.createElement('div');bubble.className='bubble typing-bubble';
  bubble.innerHTML='<div class="dots"><i></i><i></i><i></i></div>';
  typingRow.appendChild(bubble);chat.appendChild(typingRow);scrollBottom();
}
function removeTyping(){if(typingRow){typingRow.remove();typingRow=null;}}
function updateSnapshot(text){
  if(!activeRow){removeTyping();activeRow=addBubble('','assistant');activeRow.bubble.classList.add('streaming');}
  activeRow.content.textContent=text;scrollBottom();
}
function resize(){field.style.height='auto';field.style.height=Math.min(field.scrollHeight,140)+'px';}
field.addEventListener('input',resize);
field.addEventListener('keydown', e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
$('send').onclick=()=>send();
$('stop').onclick=async()=>{try{await fetch('/stop',{method:'POST'});status('stopping…');}catch(e){toast('Could not stop response');}};
$('temp').oninput=()=>{$('temp-val').textContent=$('temp').value;};
$('theme').onclick=()=>{const root=document.documentElement;root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';
  localStorage.setItem('aiclone-theme',root.dataset.theme);};

async function readEvents(response){
  if(!response.ok){const err=await response.json().catch(()=>({}));throw Error(err.error||err.detail||`HTTP ${response.status}`);}
  if(!response.body) throw Error('Streaming is unavailable in this browser');
  const reader=response.body.getReader(),decoder=new TextDecoder();let pending='',finished=null;
  while(true){const {value,done}=await reader.read();if(done)break;pending+=decoder.decode(value,{stream:true});
    let boundary;while((boundary=pending.indexOf('\n'))!==-1){const line=pending.slice(0,boundary);pending=pending.slice(boundary+1);
      if(!line.trim())continue;const event=JSON.parse(line);
      if(event.type==='status')status('typing…');
      if(event.type==='snapshot')updateSnapshot(event.text);
      if(event.type==='error')throw Error(event.message);
      if(event.type==='done'){
        removeTyping();if(!activeRow && event.text)updateSnapshot(event.text);
        if(activeRow)activeRow.bubble.classList.remove('streaming');
        if(event.stopped)toast('Generation stopped');
        finished=event;
      }
    }
  }
  if(finished)return finished;
  throw Error('Connection ended before the reply was complete');
}
async function send(regenerate=false){
  if(busy)return;const text=field.value.trim();if(!regenerate&&!text)return;
  let removedReply=null, userRow=null;
  if(regenerate){if(messages.length<2||messages.at(-1).role!=='assistant'){toast('Nothing to regenerate');return;}
    removedReply=messages.pop();const rows=chat.querySelectorAll('.row.them');rows[rows.length-1]?.remove();
  } else {userRow=addBubble(text,'user');messages.push({role:'user',content:text});field.value='';resize();}
  setBusy(true);status('typing…');startTyping();activeRow=null;
  try{
    const response=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:regenerate?'':text,regenerate,temperature:Number($('temp').value)})});
    const result=await readEvents(response);
    if(result.text)messages.push({role:'assistant',content:result.text});
    else {removeTyping();if(activeRow){activeRow.row.remove();activeRow=null;}
      if(!regenerate && userRow){userRow.row.remove();messages.pop();}}
  } catch(e){removeTyping();if(activeRow){activeRow.bubble.classList.remove('streaming');}
    if(regenerate&&removedReply){messages.push(removedReply);if(activeRow)activeRow.row.remove();addBubble(removedReply.content,'assistant');}
    if(!regenerate && userRow){userRow.row.remove();messages.pop();}
    toast('Error: '+e.message);
  } finally{removeTyping();if(activeRow)activeRow.bubble.classList.remove('streaming');setBusy(false);scrollBottom();}
}
$('regenerate').onclick=()=>send(true);
$('reset').onclick=async()=>{
  if(busy)return;
  try{const res=await fetch('/reset',{method:'POST'});if(!res.ok)throw Error('Reset failed');
    messages=[];chat.innerHTML='<div class="day">Today</div><div class="empty" id="empty"><span class="big">💬</span><strong>Start a conversation</strong><span>Message Pierre to get started.</span></div>';
    field.focus();toast('New conversation');
  }catch(e){toast(e.message);}
};
$('export').onclick=()=>{
  if(!messages.length){toast('No messages yet');return;}
  const text=messages.map(m=>(m.role==='user'?'You':'Pierre')+': '+m.content).join('\n\n');
  const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='aiclone-chat.txt';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
(async()=>{
  try{const res=await fetch('/history');if(!res.ok)throw Error();const data=await res.json();
    messages=data.messages||[];for(const m of messages)addBubble(m.content,m.role);
    if(data.busy)toast('Another reply is in progress. Reload when it finishes.');
  }catch(e){toast('Could not restore session');}
  field.focus();
})();
</script>
</body>
</html>'''


@config.parse
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    config.log_config(recipe_name="ChatAppStreaming", cfg=cfg)
    print("[AICLONE] Loading checkpoint...")
    model = ChatModel(cfg)
    print(f"[AICLONE] Ready: http://{HOST}:{PORT}", flush=True)
    print("[AICLONE] From your laptop, SSH forward port 7860 or use VS Code > Ports.", flush=True)
    uvicorn.run(build_app(model), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
