// ENG label app: görüntü yükleme, iş ilerlemesi, bindirme görüntüsü, küçük arayüz yardımcıları

const STATUS = { queued: "kuyrukta", running: "çalışıyor", done: "bitti", failed: "hata", cancelled: "iptal" };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

function uploadOne(url, file, row) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    const bar = row.querySelector("progress");
    const st = row.querySelector(".st");
    xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) bar.value = ev.loaded / ev.total; };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status === 200) {
        bar.value = 1;
        st.textContent = body.warning ? "yüklendi, Label Studio: " + body.warning : "tamam";
        st.style.color = body.warning ? "var(--warn)" : "var(--ok)";
        resolve(body.warning ? "warn" : "ok");
      } else {
        st.textContent = body.error || ("hata " + xhr.status);
        st.style.color = "var(--err)";
        resolve("err");
      }
    };
    xhr.onerror = () => { st.textContent = "bağlantı hatası"; st.style.color = "var(--err)"; resolve("err"); };
    const fd = new FormData();
    fd.append("file", file);
    xhr.open("POST", url);
    xhr.send(fd);
  });
}

async function uploadFiles(files) {
  const drop = document.getElementById("drop");
  const list = document.getElementById("uploads");
  files = [...files];
  if (!files.length) return;
  const rows = files.map((f) => {
    const r = el("div", { class: "upl" }, el("span", { class: "name", title: f.name }, f.name), el("progress", { max: 1, value: 0 }),
      el("span", { class: "st muted" }, "bekliyor"));
    list.append(r);
    return r;
  });
  const res = [];
  for (let i = 0; i < files.length; i++) {           // sırayla: büyük dosyalarda sunucu ve Label Studio boğulmasın
    rows[i].querySelector(".st").textContent = "yükleniyor";
    res.push(await uploadOne(drop.dataset.url, files[i], rows[i]));
  }
  const ok = res.filter((r) => r !== "err").length;
  if (ok && res.every((r) => r === "ok")) {          // sorunsuzsa listeyi tazele, uyarı/hata varsa okunsun diye bekle
    list.append(el("div", { class: "upl summary" }, `${ok} görüntü eklendi, sayfa yenileniyor…`));
    setTimeout(() => location.reload(), 700);
  } else if (ok) {
    list.append(el("div", { class: "upl summary" }, `${ok}/${files.length} görüntü eklendi`,
      el("a", { href: location.pathname }, "sayfayı yenile")));
  }
}

function renderJobs(box, jobs) {
  if (!jobs.length) { box.replaceChildren(el("p", { class: "muted small" }, "Henüz iş yok.")); return; }
  box.replaceChildren(...jobs.slice(0, 6).map((j) => {
    const active = j.status === "queued" || j.status === "running";
    const meta = el("div", { class: "meta" }, el("b", {}, `#${j.id} · ${j.model}`),
      el("span", { class: "badge " + j.status }, STATUS[j.status] || j.status));
    const sub = el("div", { class: "sub" }, el("span", {}, j.runner || "-"),
      el("span", {}, `${j.done}/${j.total}` + (j.failed ? ` · ${j.failed} hata` : "")));
    const kids = [meta, el("progress", { class: j.status, max: j.total || 1, value: j.done + j.failed }), sub];
    if (j.status === "queued" && !j.runner_online) kids.push(el("div", { class: "note" }, "çalıştırıcı çevrimdışı, açılınca başlar"));
    if (j.errors && j.errors.length) kids.push(el("div", { class: "errs" }, j.errors.map((e) => `${e.name}: ${e.error}`).join(" · ")));
    if (j.error) kids.push(el("div", { class: "errs" }, j.error));
    if (active) {
      kids.push(el("form", { method: "post", action: `/jobs/${j.id}/cancel`, class: "inline cancel" },
        el("button", { class: "link danger" }, "iptal et")));
    }
    return el("div", { class: "job" + (active ? " active" : "") }, ...kids);
  }));
}

function watchJobs() {
  const box = document.getElementById("jobs");
  if (!box) return;
  let wasActive = null;
  const tick = async () => {
    try {
      const r = await fetch(box.dataset.url);
      if (r.ok) {
        const { jobs } = await r.json();
        renderJobs(box, jobs);
        const active = jobs.some((j) => j.status === "queued" || j.status === "running");
        if (wasActive && !active) location.reload();     // iş bitti: durum tablosunu tazele
        wasActive = active;
        setTimeout(tick, active ? 2000 : 10000);
        return;
      }
    } catch (_) {}
    setTimeout(tick, 10000);
  };
  tick();
}

// görüntü tablosu: ada göre arama ve durum süzgeci
function initImageFilter() {
  const search = document.getElementById("img-search");
  const seg = document.querySelector(".seg");
  if (!search || !seg) return;
  const rows = [...document.querySelectorAll("table.images tbody tr")];
  const none = document.querySelector(".no-match");
  let state = "all";
  const apply = () => {
    const q = search.value.trim().toLowerCase();       // sunucudaki Python lower() ile aynı
    let shown = 0;
    for (const r of rows) {
      const vis = (state === "all" || r.dataset.state === state) && (!q || r.dataset.name.includes(q));
      r.hidden = !vis;
      shown += vis;
    }
    if (none) none.hidden = shown > 0;
  };
  search.addEventListener("input", apply);
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    state = b.dataset.filter;
    apply();
  });
}

function initDataset() {
  const drop = document.getElementById("drop");
  const input = document.getElementById("files");
  if (drop) {
    input.addEventListener("change", () => { uploadFiles(input.files); input.value = ""; });
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
  }
  initImageFilter();
  watchJobs();
}

function toggleOverlay(btn, url) {
  const row = btn.closest("tr").nextElementSibling;
  const img = row.querySelector("img");
  if (!img.src) img.src = url;
  row.hidden = !row.hidden;
  (btn.querySelector("span") || btn).textContent = row.hidden ? "bindirme" : "gizle";
}

// ---------------------------------------------------------------- model yükleme (ilerleme çubuğuyla)
function mb(n) { return (n / 1048576).toFixed(1) + " MB"; }

function progressUpload(form) {
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const box = form.querySelector(".model-upl");
    const bar = box.querySelector("progress");
    const st = box.querySelector(".st");
    const btn = form.querySelector("button");
    const say = (t, color) => { st.textContent = t; st.style.color = color || ""; };
    box.hidden = false;
    btn.disabled = true;
    bar.value = 0;
    say("yükleme başlıyor…");
    const t0 = Date.now();
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) { say(`yükleniyor: ${mb(e.loaded)}`); return; }
      bar.value = e.loaded / e.total;
      const rate = e.loaded / Math.max((Date.now() - t0) / 1000, 0.1);
      const left = rate > 0 ? Math.round((e.total - e.loaded) / rate) : 0;
      say(`yükleniyor: %${Math.floor(100 * e.loaded / e.total)} · ${mb(e.loaded)} / ${mb(e.total)}` +
          (e.loaded < e.total ? ` · ~${left} s kaldı` : ""));
    };
    xhr.upload.onload = () => { bar.removeAttribute("value"); say("sunucu modeli kontrol ediyor…"); };
    xhr.onload = () => {
      if (xhr.status < 400 && xhr.responseURL) { location.href = xhr.responseURL; return; }
      bar.value = 0;
      btn.disabled = false;
      say(xhr.status === 413 ? "dosya sunucunun izin verdiği boyuttan büyük" : `yükleme başarısız (hata ${xhr.status})`, "var(--err)");
    };
    xhr.onerror = () => { btn.disabled = false; say("bağlantı koptu; yükleme tamamlanmadı, tekrar deneyin", "var(--err)"); };
    window.addEventListener("beforeunload", (e) => { if (btn.disabled && xhr.readyState < 4) e.preventDefault(); });
    xhr.open("POST", form.action);
    xhr.send(new FormData(form));
  });
}

// ---------------------------------------------------------------- her sayfada
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form[data-progress-upload]").forEach(progressUpload);
  // bildirim: kapatılabilir; yenilemede tekrar görünmesin diye msg/err adresten çıkarılır
  document.querySelectorAll(".flash-x").forEach((b) => b.addEventListener("click", () => {
    const f = b.closest(".flash");
    f.classList.add("hide");
    setTimeout(() => f.remove(), 200);
  }));
  const u = new URL(location.href);
  if (u.searchParams.has("msg") || u.searchParams.has("err")) {
    u.searchParams.delete("msg");
    u.searchParams.delete("err");
    history.replaceState(null, "", u.pathname + (u.search || "") + u.hash);
  }
  const ok = document.querySelector(".flash.ok");
  if (ok) setTimeout(() => ok.querySelector(".flash-x")?.click(), 6000);

  // kopyala düğmesi (anahtar ve komutlar)
  document.querySelectorAll("pre.copy").forEach((pre) => {
    const wrap = el("div", { class: "copy-wrap" });
    pre.replaceWith(wrap);
    const btn = el("button", { type: "button", class: "icon-btn copy-btn", title: "Kopyala", "aria-label": "Kopyala" });
    btn.innerHTML = '<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.textContent.trim());
        btn.classList.add("ok");
        btn.title = "Kopyalandı";
        setTimeout(() => { btn.classList.remove("ok"); btn.title = "Kopyala"; }, 1500);
      } catch (_) {                                  // HTTP'de clipboard yok: metni seç
        const r = document.createRange();
        r.selectNodeContents(pre);
        getSelection().removeAllRanges();
        getSelection().addRange(r);
      }
    });
    wrap.append(pre, btn);
  });

  // tıklanabilir tablo satırları (satırdaki bağlantı ya da düğme dışında bir yere tıklanınca)
  document.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", (e) => {
    if (!e.target.closest("a, button, input, form")) location.href = tr.dataset.href;
  }));
});

// ---------------------------------------------------------------- kullanım rehberi: adım adım gösterim
document.addEventListener("DOMContentLoaded", () => {
  const gd = document.getElementById("rehber");
  if (!gd) return;
  const steps = [...gd.querySelectorAll(".gd-step")];
  const links = [...gd.querySelectorAll(".gd-nav a")];
  const pager = document.querySelector(".gd-pager");
  const prev = pager.querySelector(".gd-prev"), next = pager.querySelector(".gd-next");
  let cur = 0;
  const show = (i, scroll) => {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    steps.forEach((s, k) => s.classList.toggle("on", k === cur));
    links.forEach((a, k) => { a.classList.toggle("active", k === cur); if (k < cur) a.classList.add("seen"); });
    gd.querySelector(".gd-cur").textContent = cur + 1;
    gd.querySelector(".gd-fill").style.width = `${(100 * (cur + 1)) / steps.length}%`;
    prev.disabled = cur === 0;
    next.hidden = cur === steps.length - 1;
    history.replaceState(null, "", "#" + steps[cur].id);
    if (scroll) gd.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  gd.classList.add("js");
  pager.hidden = false;
  links.forEach((a, k) => a.addEventListener("click", (e) => { e.preventDefault(); show(k, false); }));
  prev.addEventListener("click", () => show(cur - 1, true));
  next.addEventListener("click", () => show(cur + 1, true));
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowRight") show(cur + 1, false);
    if (e.key === "ArrowLeft") show(cur - 1, false);
  });
  const start = steps.findIndex((s) => "#" + s.id === location.hash);
  show(start < 0 ? 0 : start, false);
});

// ---------------------------------------------------------------- çalıştırıcıların canlı çevrimiçi durumu
document.addEventListener("DOMContentLoaded", () => {
  if (!document.querySelector("[data-runner]")) return;
  const table = document.querySelector("[data-runner-ids]");
  const refresh = async () => {
    if (document.hidden) return;
    let body;
    try {
      const r = await fetch("/api/runners", { headers: { Accept: "application/json" } });
      if (!r.ok) return;
      body = await r.json();
    } catch (_) { return; }
    // çalıştırıcılar sayfasında yeni bağlanan/silinen çalıştırıcı varsa listeyi tazele (yazı yazılmıyorsa)
    if (table && body.runners.map((x) => x.id).join(",") !== table.dataset.runnerIds &&
        !document.activeElement.closest("input, textarea, select")) { location.reload(); return; }
    for (const x of body.runners) {
      const on = x.online ? "on" : "off";
      document.querySelectorAll(`[data-runner="${x.id}"]`).forEach((node) => {
        if (node.tagName === "OPTION") {
          node.textContent = node.dataset.label + (x.online ? "" : " · çevrimdışı");
          return;
        }
        node.querySelectorAll(".r-dot, .r-pill").forEach((d) => { d.classList.remove("on", "off"); d.classList.add(on); });
        node.querySelectorAll(".r-dot").forEach((d) => { d.title = x.online ? "çevrimiçi" : "çevrimdışı"; });
        node.querySelectorAll(".r-txt").forEach((t) => { t.textContent = x.online ? "çevrimiçi" : "çevrimdışı"; });
        node.querySelectorAll(".r-ago").forEach((t) => { t.textContent = x.ago; });
      });
    }
  };
  setInterval(refresh, 5000);
  document.addEventListener("visibilitychange", refresh);
});
