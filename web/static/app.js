// ENG label app: görüntü yükleme, iş ilerlemesi, bindirme görüntüsü

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
        resolve(true);
      } else {
        st.textContent = body.error || ("hata " + xhr.status);
        st.style.color = "var(--err)";
        resolve(false);
      }
    };
    xhr.onerror = () => { st.textContent = "bağlantı hatası"; st.style.color = "var(--err)"; resolve(false); };
    const fd = new FormData();
    fd.append("file", file);
    xhr.open("POST", url);
    xhr.send(fd);
  });
}

async function uploadFiles(files) {
  const drop = document.getElementById("drop");
  const list = document.getElementById("uploads");
  const rows = [...files].map((f) => {
    const r = el("div", { class: "upl" }, el("span", { class: "name" }, f.name), el("progress", { max: 1, value: 0 }),
      el("span", { class: "st muted" }, "bekliyor"));
    list.append(r);
    return r;
  });
  let ok = 0;
  for (let i = 0; i < files.length; i++) {           // sırayla: büyük dosyalarda sunucu ve Label Studio boğulmasın
    rows[i].querySelector(".st").textContent = "yükleniyor";
    if (await uploadOne(drop.dataset.url, files[i], rows[i])) ok++;
  }
  if (ok) list.append(el("div", { class: "upl" }, el("a", { href: location.pathname }, `${ok} görüntü eklendi, sayfayı yenile`)));
}

function renderJobs(box, jobs) {
  box.replaceChildren(...jobs.slice(0, 6).map((j) => {
    const active = j.status === "queued" || j.status === "running";
    const note = j.status === "queued" && !j.runner_online ? " · çalıştırıcı çevrimdışı, açılınca başlar" : "";
    const meta = el("div", { class: "meta" },
      el("span", {}, `#${j.id} ${j.model} · ${j.runner || "-"}`),
      el("span", {}, el("span", { class: "badge " + j.status }, j.status), ` ${j.done}/${j.total}` + (j.failed ? `, ${j.failed} hata` : "") + note));
    const kids = [meta, el("progress", { max: j.total || 1, value: j.done + j.failed })];
    if (j.errors && j.errors.length) kids.push(el("div", { class: "errs" }, j.errors.map((e) => `${e.name}: ${e.error}`).join(" · ")));
    if (j.error) kids.push(el("div", { class: "errs" }, j.error));
    if (active) {
      const f = el("form", { method: "post", action: `/jobs/${j.id}/cancel`, class: "inline" }, el("button", { class: "link danger" }, "iptal"));
      kids.push(f);
    }
    return el("div", { class: "job" }, ...kids);
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

function initDataset() {
  const drop = document.getElementById("drop");
  const input = document.getElementById("files");
  if (drop) {
    input.addEventListener("change", () => uploadFiles(input.files));
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
  }
  watchJobs();
}

function toggleOverlay(btn, url) {
  const row = btn.closest("tr").nextElementSibling;
  const img = row.querySelector("img");
  if (!img.src) img.src = url;
  row.hidden = !row.hidden;
  btn.textContent = row.hidden ? "bindirme" : "gizle";
}
