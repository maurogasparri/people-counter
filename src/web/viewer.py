"""Viewer HTTP en vivo del pipeline runtime.

Streamea un composite de 3 paneles (left annotated | right raw | depth
colormap) como MJPEG sobre HTTP. Hecho para debug on-site: el operador abre
la IP del dispositivo en un celular o laptop, camina bajo las cámaras y ve
el overlay de detección / tracking / conteo en vivo.

Diseñado para ser seguro en el hot path del runtime:

- ``push`` es no-bloqueante. La queue de encode descarta el más viejo si el
  cliente es lento; el pipeline nunca se traba esperando al viewer.
- El encoding JPEG corre en su propio thread, no en el loop del pipeline.
- Las fallas de bind (puerto en uso, CAP_NET_BIND_SERVICE faltante en el
  puerto 80) se loggean pero no matan el pipeline — el counter sigue corriendo.

Puerto default 80 porque el operador on-site no carga una lista de puertos
custom. ``--web-viewer-port 0`` deshabilita el viewer entero.
"""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import threading
import time
import urllib.parse
from collections import deque
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import cv2
import numpy as np

from src.web.admin_auth import (
    ADMIN_SECRET_PATH,
    MIN_PASSWORD_LEN,
    read_secret,
    verify_password,
    write_secret,
)

logger = logging.getLogger(__name__)

# Backoff del login: tras N fallos consecutivos, cada verificación duerme
# este delay (dentro del lock → todo el flood espera en cola). Baja el
# brute-force online a ~1 guess/s sin molestar al operador legítimo (que
# rara vez falla 5 veces; un login exitoso resetea el contador).
_LOGIN_BACKOFF_AFTER = 5
_LOGIN_BACKOFF_S = 1.0

# Vida de una sesión admin server-side. Mantener en sync con el Max-Age de
# la cookie pc_session (el server es la fuente de verdad; el Max-Age solo
# le ahorra al browser mandar una cookie que igual sería rechazada).
_SESSION_TTL_S = 86400.0


_HTML = """<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <link rel='icon' href='data:,'>
  <title>People Counter — Live</title>
  <style>
    :root{--bg:#0b0d10;--panel:#15181e;--panel2:#1c2128;--line:#2a2f38;
      --txt:#e6e9ee;--muted:#8b94a3;--in:#3ddc84;--out:#ff8c42;--accent:#5b9dff;}
    *{box-sizing:border-box;}
    body{margin:0;background:var(--bg);color:var(--txt);font-size:14px;
      font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;}
    .wrap{max-width:1180px;margin:0 auto;padding:14px;}
    header{display:flex;align-items:center;justify-content:space-between;gap:16px;
      flex-wrap:wrap;padding:12px 16px;background:var(--panel);
      border:1px solid var(--line);border-radius:12px;}
    .brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;}
    .dot{width:9px;height:9px;border-radius:50%;background:var(--in);
      box-shadow:0 0 8px var(--in);animation:pulse 2s infinite;}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
    .site{text-align:right;line-height:1.35;}
    .site .name{font-weight:600;}
    .site .dev{color:var(--muted);font-family:ui-monospace,monospace;font-size:12px;}
    .meta{display:flex;gap:18px;align-items:center;padding:9px 6px 0;
      color:var(--muted);font-size:12.5px;font-family:ui-monospace,monospace;}
    .meta b{color:var(--txt);}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;}
    @media(max-width:820px){.grid{grid-template-columns:1fr;}}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:14px 16px;}
    /* Header de cada card: título a la izquierda + status pill a la derecha
       para que el operator vea de un vistazo "activo / pausado por shadow /
       fuera de horario" sin tener que mirar el meta header arriba. */
    .card-head{display:flex;align-items:center;justify-content:space-between;
      gap:10px;margin:0 0 10px;}
    .card h3{margin:0;font-size:11.5px;letter-spacing:.06em;
      text-transform:uppercase;color:var(--muted);font-weight:600;}
    .status-pill{font-size:10px;font-weight:600;padding:2px 8px;border-radius:999px;
      letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;}
    .status-pill.on{background:rgba(61,220,132,.16);color:var(--in);}
    .status-pill.off{background:rgba(255,140,66,.16);color:var(--out);}
    .status-pill.idle{background:rgba(139,148,163,.14);color:var(--muted);}
    /* KPI boxes — mismo tamaño/padding/tipografía para IN/OUT y para el
       total de tráfico exterior. Los colores verde/naranja se aplican
       SOLO a .kpi.in / .kpi.out via .v; los KPI sin clase de color
       heredan el blanco default del card. */
    .kpis{display:flex;gap:12px;margin-bottom:12px;}
    .kpi{flex:1;text-align:center;border-radius:10px;padding:12px 8px;
      background:var(--panel2);border:1px solid var(--line);}
    .kpi .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
    .kpi .v{font-size:46px;font-weight:700;line-height:1.05;font-variant-numeric:tabular-nums;}
    .kpi.in .v{color:var(--in);} .kpi.out .v{color:var(--out);}
    /* Stats compactas (FPS/Tracks/Dets) — viven debajo del stream junto
       con el caption "L | R | disparidad", no en la card de Conteo. */
    .substats{display:flex;gap:18px;color:var(--muted);
      font-size:12.5px;font-family:ui-monospace,monospace;}
    .substats b{color:var(--txt);}
    table{width:100%;border-collapse:collapse;font-size:12px;
      font-family:ui-monospace,monospace;}
    th{text-align:left;color:var(--muted);font-weight:600;padding:5px 6px;
      border-bottom:1px solid var(--line);font-size:10.5px;text-transform:uppercase;}
    td{padding:5px 6px;border-bottom:1px solid #20242c;white-space:nowrap;}
    .tag{font-size:9px;color:var(--muted);}
    .empty{color:var(--muted);text-align:center;padding:16px;}
    .stream{margin-top:14px;background:#000;border:1px solid var(--line);
      border-radius:12px;overflow:hidden;}
    .stream img{display:block;width:100%;}
    .stream .cap{padding:6px 12px;color:var(--muted);font-size:11px;
      font-family:ui-monospace,monospace;border-top:1px solid var(--line);
      display:flex;align-items:center;justify-content:space-between;gap:18px;}
    .off{color:var(--out);}
  </style>
</head>
<body>
  <style>
    .adminbar{margin-top:10px;padding:8px 14px;background:var(--panel);
      border:1px solid var(--line);border-radius:12px;}
    .adm-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
    .adm-label{display:inline-flex;align-items:center;gap:6px;color:var(--muted);
      font-size:12.5px;font-weight:600;margin-right:4px;}
    .adminbar input{background:var(--panel2);border:1px solid var(--line);
      color:var(--txt);border-radius:8px;padding:6px 9px;font-size:13px;min-width:120px;}
    .abtn{display:inline-flex;align-items:center;justify-content:center;gap:6px;
      line-height:1;border:1px solid var(--line);background:var(--panel2);
      color:var(--txt);border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;}
    .ic{font-size:14px;line-height:1;}
    .abtn:hover{border-color:var(--accent);}
    .abtn.warn{border-color:var(--out);color:var(--out);}
    .abtn.danger{border-color:#ff5d5d;color:#ff5d5d;}
    .abtn.ghost{color:var(--muted);}
  </style>
  <div class='wrap'>
    <header>
      <div class='brand'><span class='dot'></span> People Counter</div>
      <div class='site'>
        <div class='name' id='site'>—</div>
        <div class='dev' id='device'>—</div>
      </div>
    </header>
    <div class='adminbar' id='adminbar' style='display:none'>
      <form id='loginForm' class='adm-row'>
        <span class='adm-label'><span class='ic'>🔒</span>Admin</span>
        <input type='password' id='admpw' placeholder='Contraseña' autocomplete='current-password'>
        <button type='submit' class='abtn'>Entrar</button>
      </form>
      <div id='powerPanel' class='adm-row' style='display:none'>
        <span class='adm-label'><span class='ic'>🔓</span>Admin</span>
        <button id='btnReboot' class='abtn warn'><span class='ic'>⟳</span>Reiniciar</button>
        <button id='btnShutdown' class='abtn danger'><span class='ic'>⏻</span>Apagar</button>
        <button id='btnChangePw' class='abtn'>Cambiar contraseña</button>
        <button id='btnLogout' class='abtn ghost'>Salir</button>
      </div>
      <form id='changePwForm' class='adm-row' style='display:none'>
        <input type='password' id='curpw' placeholder='Actual' autocomplete='current-password'>
        <input type='password' id='newpw' placeholder='Nueva (mín 8)' autocomplete='new-password'>
        <input type='password' id='newpw2' placeholder='Repetir' autocomplete='new-password'>
        <button type='submit' class='abtn'>Guardar</button>
        <button type='button' id='btnCancelPw' class='abtn ghost'>Cancelar</button>
      </form>
    </div>
    <div class='meta'>
      <span>Horario <b id='hours'>—</b></span>
      <span style='margin-left:auto'><b id='clock'>—</b></span>
    </div>
    <div class='grid'>
      <div class='card'>
        <div class='card-head'>
          <h3>Conteo (hoy)</h3>
          <span class='status-pill idle' id='counting-status'>—</span>
        </div>
        <div class='kpis'>
          <div class='kpi in'><div class='k'>IN</div><div class='v' id='in'>0</div></div>
          <div class='kpi out'><div class='k'>OUT</div><div class='v' id='out'>0</div></div>
        </div>
        <table>
          <thead><tr><th>Hora</th><th>IN</th><th>OUT</th></tr></thead>
          <tbody id='hourly'><tr><td class='empty' colspan='3'>Sin actividad hoy</td></tr></tbody>
        </table>
      </div>
      <div class='card'>
        <div class='card-head'>
          <h3>Tráfico exterior · WiFi/BLE</h3>
          <span class='status-pill idle' id='external-status'>—</span>
        </div>
        <div class='kpis'>
          <div class='kpi'><div class='k'>Total observados</div><div class='v' id='ext-total'>0</div></div>
        </div>
        <table>
          <thead><tr><th>Hash</th><th>Hora</th><th>RSSI</th></tr></thead>
          <tbody id='ext'><tr><td class='empty' colspan='3'>Sin registros</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class='stream'>
      <img id='stream' src='/stream' alt='live stream' onerror='markStreamBroken()'/>
      <div class='cap'>
        <span>L | R | disparidad</span>
        <span class='substats'>
          <span>FPS <b id='fps'>—</b></span>
          <span>Tracks <b id='tracks'>0</b></span>
          <span>Dets <b id='dets'>0</b></span>
        </span>
      </div>
    </div>
  </div>
  <script>
    const $=id=>document.getElementById(id);
    const fmtT=e=>e?new Date(e*1000).toLocaleTimeString('es-AR',{hour12:false}):'—';
    const fmtR=r=>(r==null)?'—':Math.round(r)+' dBm';
    // Días en orden L → D. La inicial (1 letra) alcanza para el meta
    // header compacto; usamos formato hh-hh sin minutos cuando son ":00".
    const DAY_KEYS=['monday','tuesday','wednesday','thursday','friday','saturday','sunday'];
    const DAY_INITIALS={monday:'L',tuesday:'M',wednesday:'X',thursday:'J',friday:'V',saturday:'S',sunday:'D'};
    function trimHHMM(s){return s.replace(/:00\b/g,'');}
    // Formatea un operating_hours dict como "L-V 10-22 · S-D 10-21".
    // Agrupa runs de días consecutivos con mismo schedule. null/missing = "cerrado".
    function fmtHours(oh){
      if(!oh||typeof oh!=='object')return '—';
      // Build array [{day, sched}] preservando orden L→D.
      const days=DAY_KEYS.map(k=>({day:k,sched:(oh[k]==null?'cerrado':trimHHMM(oh[k]))}));
      if(days.every(d=>d.sched==='cerrado'))return 'cerrado';
      // Agrupar runs.
      const groups=[];
      for(const d of days){
        const last=groups[groups.length-1];
        if(last&&last.sched===d.sched){last.end=d.day;}
        else{groups.push({start:d.day,end:d.day,sched:d.sched});}
      }
      return groups.map(g=>{
        const a=DAY_INITIALS[g.start],b=DAY_INITIALS[g.end];
        const range=(a===b)?a:`${a}-${b}`;
        return g.sched==='cerrado'?`${range} cerrado`:`${range} ${g.sched}`;
      }).join(' · ');
    }
    // El <img> MJPEG no se reconecta solo al reiniciarse el server. Asignar
    // simplemente `img.src = '/stream?t=N'` NO siempre triggerea un fetch
    // nuevo: algunos browsers (Chromium notablemente) mantienen la conexión
    // MJPEG vieja en estado "hung" y el src=newUrl no la cierra. Fix
    // robusto: limpiar src primero (forzar al browser a soltar la conexión
    // vieja), esperar al evento de load del src vacío en el próximo tick
    // del event loop, y entonces asignar la URL nueva con cache-buster.
    let _streamNeedsReload=false;
    let _streamLastReloadAt=0;
    const STREAM_RELOAD_INTERVAL_MS=2000;
    function markStreamBroken(){_streamNeedsReload=true;}
    function maybeReloadStream(){
      if(!_streamNeedsReload)return;
      const now=Date.now();
      if(now-_streamLastReloadAt<STREAM_RELOAD_INTERVAL_MS)return;
      _streamLastReloadAt=now;
      _streamNeedsReload=false;  // re-marca si el nuevo intento también falla
      const img=$('stream');
      // 1x1 transparent GIF — fuerza al browser a cerrar la conexión
      // MJPEG vieja sin generar otro request (data: URLs no van por red).
      img.src='data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
      // Tick para que el browser procese el reset antes del nuevo GET.
      setTimeout(()=>{ img.src='/stream?t='+Date.now(); }, 80);
    }
    setInterval(maybeReloadStream,500);

    // Health probe dedicado para detectar caídas del server rápido (ej.
    // restart del systemd unit). Sin esto, el cliente quedaba esperando
    // al tick(1s) de /stats que podía tardar más en disparar el catch
    // (un fetch que se "cuelga" mid-restart). Poll cada 3s con timeout
    // de 2s vía AbortController garantiza detección y reconnect en <5s.
    let _healthFails=0;
    const HEALTH_TIMEOUT_MS=2000;
    async function healthPoll(){
      const ctrl=new AbortController();
      const tid=setTimeout(()=>ctrl.abort(),HEALTH_TIMEOUT_MS);
      try{
        const r=await fetch('/health?t='+Date.now(),
          {cache:'no-store',signal:ctrl.signal});
        if(!r.ok)throw new Error('non-200');
        if(_healthFails>0){
          // Server volvió después de fallar — recargar stream MJPEG.
          markStreamBroken();
        }
        _healthFails=0;
      }catch(e){_healthFails++;}
      finally{clearTimeout(tid);}
    }
    setInterval(healthPoll,3000);
    // Backup: si el browser detecta el error del <img> (algunos sí lo
    // disparan en stream caído), forzar reload sin esperar al poll.
    $('stream').addEventListener('error',markStreamBroken);

    let _statsOk=true;
    async function tick(){
      try{
        const s=await (await fetch('/stats',{cache:'no-store'})).json();
        if(!_statsOk){_statsOk=true;markStreamBroken();}  // server volvió → reconectar video
        $('site').textContent=s.store_name||'—';
        $('device').textContent=s.device_id||'—';
        $('hours').textContent=fmtHours(s.operating_hours);
        $('clock').textContent=s.device_time||'—';
        // Status pills por card. Texto unificado: "activo" / "pausado" /
        // "fuera de horario". El "pausado" agrupa tanto el toggle del
        // shadow (counting_enabled=false / external_traffic_enabled=false)
        // como cualquier otra forma futura de pausa — desde la perspectiva
        // del operator todas son lo mismo: subsystem no está reportando.
        function setPill(el, cls, txt){
          el.className='status-pill ' + cls;
          el.textContent=txt;
        }
        const cTog=s.counting_toggle_on;
        const wh=s.within_hours;
        if(cTog==null){setPill($('counting-status'),'idle','—');}
        else if(!cTog){setPill($('counting-status'),'off','pausado');}
        else if(!wh){setPill($('counting-status'),'off','fuera de horario');}
        else{setPill($('counting-status'),'on','activo');}
        const eOn=s.external_traffic_on;
        if(eOn==null){setPill($('external-status'),'idle','—');}
        else if(!eOn){setPill($('external-status'),'off','pausado');}
        else{setPill($('external-status'),'on','activo');}
        $('in').textContent=s.total_in??0;
        $('out').textContent=s.total_out??0;
        $('fps').textContent=(typeof s.fps==='number')?s.fps.toFixed(1):'—';
        $('tracks').textContent=s.tracks??0;
        $('dets').textContent=s.dets??0;
        const hr=s.hourly||[];
        $('hourly').innerHTML=hr.length?hr.map(r=>
          `<tr><td>${String(r.hour).padStart(2,'0')}:00</td>`+
          `<td style='color:var(--in)'>${r.in}</td>`+
          `<td style='color:var(--out)'>${r.out}</td></tr>`
        ).join(''):"<tr><td class='empty' colspan='3'>Sin actividad hoy</td></tr>";
        const ext=s.exterior||{};
        // Total = passersby (shoppers ⊆ passersby por invariante del dedup).
        // La categorización passerby/shopper por RSSI vive ahora en la
        // función SQL rssi_class() server-side; el device solo expone el
        // total observado y los registros crudos.
        $('ext-total').textContent=ext.passersby??0;
        const rows=ext.recent||[];
        $('ext').innerHTML=rows.length?rows.map(r=>
          `<tr><td>${r.hash} <span class='tag'>${(r.protocol||'').toUpperCase()}</span></td>`+
          `<td>${fmtT(r.last_seen)}</td><td>${fmtR(r.rssi)}</td></tr>`
        ).join(''):"<tr><td class='empty' colspan='3'>Sin registros</td></tr>";
      }catch(e){_statsOk=false;}
    }
    setInterval(tick,1000);tick();

    // ----- Panel admin: login + power + cambio de contraseña -----
    async function refreshAuth(){
      try{
        const a=await (await fetch('/auth/status',{cache:'no-store'})).json();
        if(!a.power_enabled){$('adminbar').style.display='none';return;}
        $('adminbar').style.display='block';
        $('loginForm').style.display=a.authed?'none':'flex';
        $('powerPanel').style.display=a.authed?'flex':'none';
        $('changePwForm').style.display='none';
      }catch(e){}
    }
    $('loginForm').addEventListener('submit',async ev=>{
      ev.preventDefault();
      const r=await fetch('/login',{method:'POST',
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:'password='+encodeURIComponent($('admpw').value)});
      $('admpw').value='';
      if(r.ok){refreshAuth();}else{alert('Contraseña incorrecta');}
    });
    $('btnLogout').addEventListener('click',async()=>{
      await fetch('/logout',{method:'POST'});refreshAuth();
    });
    async function powerAction(path,verbo){
      if(!confirm('¿Seguro que querés '+verbo+' el dispositivo? Va a quedar fuera de línea.'))return;
      const r=await fetch(path,{method:'POST'});
      if(r.status===403){alert('Sesión expirada — volvé a entrar.');refreshAuth();return;}
      if(r.ok){alert('Listo: '+verbo+' iniciado. El dispositivo se desconecta en unos segundos.');}
      else{alert('No se pudo ('+r.status+').');}
    }
    $('btnReboot').addEventListener('click',()=>powerAction('/reboot','reiniciar'));
    $('btnShutdown').addEventListener('click',()=>powerAction('/shutdown','apagar'));
    $('btnChangePw').addEventListener('click',()=>{
      $('powerPanel').style.display='none';$('changePwForm').style.display='flex';
    });
    $('btnCancelPw').addEventListener('click',()=>{
      $('curpw').value=$('newpw').value=$('newpw2').value='';refreshAuth();
    });
    $('changePwForm').addEventListener('submit',async ev=>{
      ev.preventDefault();
      if($('newpw').value!==$('newpw2').value){alert('La nueva no coincide.');return;}
      const r=await fetch('/change-password',{method:'POST',
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:'current='+encodeURIComponent($('curpw').value)+
             '&new='+encodeURIComponent($('newpw').value)});
      const j=await r.json().catch(()=>({}));
      $('curpw').value=$('newpw').value=$('newpw2').value='';
      if(r.ok){alert('Contraseña actualizada.');refreshAuth();}
      else{alert('No se pudo: '+(j.error||r.status));}
    });
    refreshAuth();
  </script>
</body>
</html>
"""


class _ReusableServer(ThreadingHTTPServer):
    # SO_REUSEADDR así un restart rápido no pega contra un socket TIME_WAIT.
    allow_reuse_address = True
    daemon_threads = True


class WebViewer:
    """Viewer MJPEG live sobre HTTP, aislado del hot path del pipeline.

    Lifecycle:
        viewer = WebViewer(port=80)
        if viewer.start():            # False en falla de bind (puerto ocupado / perm)
            ...
            viewer.push(frame_bgr, {"total_in": ..., ...})
            ...
            viewer.stop()
    """

    def __init__(
        self,
        port: int = 80,
        host: str = "0.0.0.0",
        jpeg_quality: int = 55,
        queue_size: int = 4,
        secret_path: str = ADMIN_SECRET_PATH,
    ) -> None:
        self.port = port
        self.host = host
        # Path del hash de la contraseña admin. Si el archivo existe, los
        # controles de power (Reiniciar/Apagar) se habilitan; si no, no aparecen.
        self._secret_path = secret_path
        self.jpeg_quality = int(jpeg_quality)
        # Queue de input al encoder. Semántica drop-oldest enforced en ``push``.
        self._encode_queue: deque[np.ndarray] = deque(maxlen=queue_size)
        self._encode_lock = threading.Lock()
        self._encode_evt = threading.Event()
        # Output JPEG más reciente, servido a todos los clientes suscriptos.
        self._latest_jpeg: Optional[bytes] = None
        self._latest_id = 0
        self._latest_cond = threading.Condition()
        # Stats publicados por /stats.
        self._stats: dict[str, Any] = {
            "total_in": 0,
            "total_out": 0,
            "fps": 0.0,
            "tracks": 0,
            "dets": 0,
        }
        self._stats_lock = threading.Lock()
        self._server: Optional[_ReusableServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._encode_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        # Contador de subscribers activos al MJPEG stream. El runtime
        # consulta ``has_subscribers`` antes de hacer trabajo que solo
        # sirve al preview (annotation overlay + SGBM-para-panel-depth
        # cada 0.5s) — match al patrón production-grade donde el
        # productor no gasta CPU cuando nadie está mirando. El counting
        # del pipeline NO depende de este flag.
        self._subscribers = 0
        self._subscribers_lock = threading.Lock()
        # Auth: serializa la verificación PBKDF2 + backoff anti brute-force
        # (ver _check_password y las constantes _LOGIN_BACKOFF_*).
        self._auth_lock = threading.Lock()
        self._failed_logins = 0
        # Sesiones admin activas: token random -> expiry epoch (ver
        # _new_session). Protegido por _auth_lock.
        self._sessions: dict[str, float] = {}

    # ----------------------------------------------------------------- API
    @property
    def running(self) -> bool:
        return self._server is not None and not self._stop_evt.is_set()

    def has_subscribers(self) -> bool:
        """¿Hay clientes activos consumiendo el MJPEG stream?

        El main loop usa este flag para gatear trabajo que solo sirve al
        preview (annotate_left, compose_3panel, push al encoder, refresh
        de depth panel cada 0.5s). Cuando no hay clientes, esos costs se
        evitan — el counting del pipeline sigue corriendo idéntico.

        Devuelve False cuando el viewer no arrancó (bind failure) o
        cuando el server está apagado, así los callers pueden gateaer sin
        chequear ``running`` por separado.
        """
        if self._server is None or self._stop_evt.is_set():
            return False
        with self._subscribers_lock:
            return self._subscribers > 0

    def _subscriber_attach(self) -> None:
        with self._subscribers_lock:
            self._subscribers += 1

    def _subscriber_detach(self) -> None:
        with self._subscribers_lock:
            if self._subscribers > 0:
                self._subscribers -= 1

    def start(self) -> bool:
        """Bindea, arranca a servir, devuelve True ante éxito.

        Ante falla de bind (puerto en uso, sin permiso para puerto 80)
        loguea un warning y devuelve False — el caller debería tratar
        al viewer como deshabilitado pero mantener el pipeline corriendo.
        """
        handler_cls = _build_handler(self)
        try:
            self._server = _ReusableServer(
                (self.host, self.port),
                handler_cls,
            )
        except (OSError, PermissionError) as e:
            logger.warning(
                "WebViewer bind to %s:%d failed (%s) - viewer disabled, "
                "pipeline continues.",
                self.host,
                self.port,
                e,
            )
            self._server = None
            return False

        self._stop_evt.clear()
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="web-viewer-http",
            daemon=True,
        )
        self._server_thread.start()
        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            name="web-viewer-encode",
            daemon=True,
        )
        self._encode_thread.start()
        logger.info(
            "WebViewer listening on http://%s:%d/",
            self.host,
            self.port,
        )
        return True

    def stop(self) -> None:
        if self._server is None:
            return
        self._stop_evt.set()
        # Despertar a los waiters así los handlers de /stream salen limpio.
        with self._latest_cond:
            self._latest_cond.notify_all()
        self._encode_evt.set()
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            logger.exception("WebViewer shutdown failed")
        if self._encode_thread is not None:
            self._encode_thread.join(timeout=2.0)
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._server = None

    def update_stats(self, stats: dict) -> None:
        """Actualiza el dict que sirve ``/stats``. Cheap (lock + update).

        Se llama siempre, independiente de si hay clientes mirando el
        MJPEG — alguien puede estar polleando ``/stats`` solo para
        monitorear counts sin abrir el stream.
        """
        if self._server is None:
            return
        with self._stats_lock:
            self._stats.update(stats)

    def push(
        self,
        frame_bgr: np.ndarray,
        stats: Optional[dict] = None,
    ) -> None:
        """Encola un frame para encode JPEG. No bloqueante.

        Stats se actualizan siempre cuando se proveen (via update_stats).
        El frame solo se encolea cuando hay subscribers activos —
        si nadie está mirando, el encode JPEG es trabajo descartado
        y el caller debería haberse salteado el composite igual.
        Mantener el guard interno como defensa en profundidad.
        """
        if self._server is None:
            return
        if stats is not None:
            self.update_stats(stats)
        if not self.has_subscribers():
            return
        with self._encode_lock:
            self._encode_queue.append(frame_bgr)
        self._encode_evt.set()

    # ------------------------------------------------------------- internal
    def _encode_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._encode_evt.wait(timeout=0.5)
            self._encode_evt.clear()
            while not self._stop_evt.is_set():
                with self._encode_lock:
                    if not self._encode_queue:
                        break
                    # Cuando el encoder se queda atrás, saltar adelante
                    # al frame más nuevo y descartar el resto.
                    frame = self._encode_queue.pop()
                    self._encode_queue.clear()
                try:
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                except Exception:
                    logger.exception("JPEG encode failed")
                    continue
                if not ok:
                    continue
                with self._latest_cond:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_id += 1
                    self._latest_cond.notify_all()

    def _wait_next_jpeg(
        self,
        last_seen: int,
        timeout: float = 5.0,
    ) -> tuple[Optional[bytes], int]:
        with self._latest_cond:
            if self._latest_id == last_seen and not self._stop_evt.is_set():
                self._latest_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._latest_id

    def _stats_json(self) -> bytes:
        with self._stats_lock:
            return json.dumps(self._stats).encode()

    # --------------------------------------------------------- admin / power
    def _power_enabled(self) -> bool:
        """True si hay contraseña admin seteada (habilita Reiniciar/Apagar)."""
        return read_secret(self._secret_path) is not None

    def _check_password(self, password: str) -> bool:
        """Verifica la contraseña con serialización + backoff.

        El lock convierte un flood de logins en una cola de a uno:
        ``pbkdf2_hmac`` (200k rounds, ~100ms en la Pi) LIBERA el GIL y
        ``ThreadingHTTPServer`` no capea threads — sin el lock, N requests
        concurrentes saturaban los 4 cores compitiendo con visión/SGBM.
        Tras ``_LOGIN_BACKOFF_AFTER`` fallos consecutivos se agrega un
        sleep DENTRO del lock (todo el flood espera detrás) que baja el
        brute-force online a ~1 guess/s; un login OK resetea el contador.
        """
        with self._auth_lock:
            if self._failed_logins >= _LOGIN_BACKOFF_AFTER:
                time.sleep(_LOGIN_BACKOFF_S)
            ok = verify_password(password, read_secret(self._secret_path))
            if ok:
                self._failed_logins = 0
            else:
                self._failed_logins += 1
            return ok

    def _new_session(self) -> str:
        """Crea una sesión nueva: token random + expiry server-side.

        Reemplaza al token determinístico HMAC(hash) anterior, que era
        idéntico para todos los logins y válido para siempre hasta el
        próximo cambio de contraseña: el Max-Age de la cookie era puramente
        client-side y /logout solo borraba la cookie del browser que lo
        pedía. Con tokens per-sesión, el server expira de verdad
        (``_SESSION_TTL_S``), /logout invalida ESA sesión y el cambio de
        contraseña invalida todas (``_invalidate_sessions``). Un restart
        del proceso borra las sesiones — re-login, aceptable.
        """
        token = secrets.token_hex(32)
        with self._auth_lock:
            self._sessions[token] = time.time() + _SESSION_TTL_S
        return token

    def _cookie_token(self, cookie_header: Optional[str]) -> Optional[str]:
        if not cookie_header:
            return None
        try:
            jar = SimpleCookie()
            jar.load(cookie_header)
            morsel = jar.get("pc_session")
        except Exception:
            return None
        return morsel.value if morsel is not None else None

    def _is_authed(self, cookie_header: Optional[str]) -> bool:
        token = self._cookie_token(cookie_header)
        if not token:
            return False
        now = time.time()
        with self._auth_lock:
            # Purge perezoso de sesiones vencidas (el dict queda acotado al
            # puñado de logins recientes; no hace falta un sweeper aparte).
            expired = [t for t, exp in self._sessions.items() if exp <= now]
            for t in expired:
                del self._sessions[t]
            return self._sessions.get(token, 0.0) > now

    def _drop_session(self, cookie_header: Optional[str]) -> None:
        """Invalida server-side la sesión del cookie (logout real)."""
        token = self._cookie_token(cookie_header)
        if token:
            with self._auth_lock:
                self._sessions.pop(token, None)

    def _invalidate_sessions(self) -> None:
        """Invalida TODAS las sesiones (post cambio de contraseña)."""
        with self._auth_lock:
            self._sessions.clear()

    def _change_password(self, current: str, new: str) -> tuple[bool, str]:
        """(ok, mensaje). Valida la actual + política de la nueva, escribe el hash."""
        if not self._check_password(current):
            return False, "contraseña actual incorrecta"
        if len(new) < MIN_PASSWORD_LEN:
            return False, f"mínimo {MIN_PASSWORD_LEN} caracteres"
        try:
            write_secret(new, self._secret_path)
        except OSError as e:
            logger.exception("admin_password_write_failed")
            return False, f"no se pudo guardar: {e}"
        # Cambio de contraseña invalida toda sesión previa — el caller
        # re-emite una cookie nueva para la sesión actual.
        self._invalidate_sessions()
        logger.warning("admin_password_changed")
        return True, "ok"

    def _trigger_power(self, action: str) -> bool:
        """Dispara reboot/poweroff vía systemd-logind. NO usa sudo: el request
        va a PID1 (ya root) → compatible con NoNewPrivileges. Requiere la regla
        polkit que autoriza a pi (config/polkit/10-people-counter-power.rules)."""
        cmd = {"reboot": "reboot", "shutdown": "poweroff"}.get(action)
        if cmd is None:
            return False
        try:
            subprocess.Popen(
                ["systemctl", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.warning("power_action_triggered action=%s", cmd)
            return True
        except Exception:
            logger.exception("power_action_failed action=%s", action)
            return False


# Cap del body de los POST (login/change-password). Un form real pesa <1KB;
# sin cap, un Content-Length gigante (sin auth — /login lee el body antes de
# verificar nada) acumularía todo el body en un solo bytes en RAM: en la Pi
# de 2GB el OOM killer mata el proceso ENTERO, visión incluida.
_MAX_FORM_BYTES = 8 * 1024


def _build_handler(viewer: WebViewer):
    class Handler(BaseHTTPRequestHandler):
        # Timeout del socket por conexión: un cliente que abre y no manda
        # nada (o declara Content-Length y nunca envía el body) no puede
        # retener el thread indefinidamente. No afecta a /stream, que solo
        # escribe — y un cliente MJPEG colgado también se reapea con esto.
        timeout = 30

        # Silenciar el log default per-request a stderr (spamearía el
        # log con cada byte de MJPEG).
        def log_message(self, fmt, *args) -> None:  # noqa: ARG002
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                _html_bytes = _HTML.encode("utf-8")
                self.send_header("Content-Length", str(len(_html_bytes)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(_html_bytes)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/stats":
                payload = viewer._stats_json()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/stream":
                self._stream_mjpeg()
                return
            if path == "/health":
                # Health probe ultra-liviano. El JS del viewer lo poll-ea
                # cada N segundos para detectar caídas del server (restart
                # del service, network blip) y reconectar el stream MJPEG
                # automáticamente. Sin esto el browser queda con un frame
                # congelado hasta que el operador refresque manualmente.
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path == "/auth/status":
                self._send_json(
                    200,
                    {
                        "power_enabled": viewer._power_enabled(),
                        "authed": viewer._is_authed(self.headers.get("Cookie")),
                    },
                )
                return
            self.send_error(404)

        # --------------------------------------------------------- helpers
        def _send_json(
            self, code: int, obj: dict, cookie: Optional[str] = None
        ) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cookie is not None:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_form(self) -> Optional[dict]:
            """Lee y parsea el form urlencoded del body.

            Devuelve None (y responde 413) si el Content-Length declarado
            excede ``_MAX_FORM_BYTES`` — el caller debe cortar ahí.
            """
            try:
                n = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                n = 0
            if n > _MAX_FORM_BYTES:
                # Cerrar la conexión: el body nunca se lee, y si quedara
                # abierta el handler intentaría parsear esos bytes como el
                # próximo request.
                self.close_connection = True
                self._send_json(413, {"error": "body too large"})
                return None
            raw = self.rfile.read(n).decode("utf-8", "replace") if n > 0 else ""
            return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]

            if path == "/login":
                if not viewer._power_enabled():
                    self._send_json(404, {"error": "disabled"})
                    return
                form = self._read_form()
                if form is None:
                    return
                pw = form.get("password", "")
                if viewer._check_password(pw):
                    cookie = (
                        f"pc_session={viewer._new_session()}; HttpOnly; Path=/; "
                        "SameSite=Strict; Max-Age=86400"
                    )
                    self._send_json(200, {"ok": True}, cookie=cookie)
                else:
                    self._send_json(401, {"error": "contraseña incorrecta"})
                return

            if path == "/logout":
                # Logout REAL: invalida la sesión server-side (antes solo se
                # borraba la cookie del browser que lo pedía y el token —
                # compartido — seguía siendo válido desde otros clientes).
                viewer._drop_session(self.headers.get("Cookie"))
                self._send_json(
                    200,
                    {"ok": True},
                    cookie="pc_session=; HttpOnly; Path=/; Max-Age=0",
                )
                return

            if path == "/change-password":
                if not viewer._is_authed(self.headers.get("Cookie")):
                    self._send_json(403, {"error": "auth required"})
                    return
                form = self._read_form()
                if form is None:
                    return
                ok, msg = viewer._change_password(
                    form.get("current", ""), form.get("new", "")
                )
                if ok:
                    # _change_password invalidó todas las sesiones → emitir
                    # una nueva para que ESTA sesión siga sin re-login.
                    cookie = (
                        f"pc_session={viewer._new_session()}; HttpOnly; Path=/; "
                        "SameSite=Strict; Max-Age=86400"
                    )
                    self._send_json(200, {"ok": True}, cookie=cookie)
                else:
                    self._send_json(400, {"error": msg})
                return

            if path in ("/reboot", "/shutdown"):
                if not viewer._power_enabled():
                    self._send_json(404, {"error": "disabled"})
                    return
                if not viewer._is_authed(self.headers.get("Cookie")):
                    self._send_json(403, {"error": "auth required"})
                    return
                action = "reboot" if path == "/reboot" else "shutdown"
                # Responder ANTES de disparar para que el browser reciba el 200
                # (después de esto el device se va a desconectar).
                self._send_json(200, {"ok": True, "action": action})
                try:
                    self.wfile.flush()
                except Exception:
                    pass
                viewer._trigger_power(action)
                return

            self.send_error(404)

        def _stream_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            # Registrar el cliente para que has_subscribers() devuelva
            # True. detach incondicional en el finally así crashes de
            # render / kills del browser no dejan el contador inflado.
            viewer._subscriber_attach()
            last_id = -1
            try:
                while not viewer._stop_evt.is_set():
                    jpeg, last_id = viewer._wait_next_jpeg(last_id)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Cliente desconectado mid-stream; esperado.
                pass
            finally:
                viewer._subscriber_detach()

    return Handler
