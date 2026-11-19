"""label-studio-sdk işlemleri: bağlantı, proje, görüntü + tahmin yükleme, görev durumu, etiketleri çekme.

API anahtarı: Label Studio > Account & Settings > Personal Access Token (yeni sürümlerde varsayılan) ya da eski tip
"legacy token"; SDK ikisini de tanır. Görüntüler Label Studio'ya dosya olarak yüklenir (/api/projects/<id>/import),
böylece sunucu başka bir bilgisayarda da olabilir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from label_studio_sdk import LabelStudio

UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")     # Label Studio yüklenen dosya adının önüne 8 karakter ekler


class LSError(RuntimeError):
    pass


def original_name(task_data: dict) -> str:
    """Görev verisindeki görüntü yolundan özgün dosya adı (yükleme önekini atarak)."""
    name = Path(str(task_data.get("image", "")).split("?d=")[-1]).name
    return UPLOAD_PREFIX.sub("", name)


@dataclass
class Status:
    total: int
    labeled: int
    with_predictions: int

    @property
    def pending(self) -> int:
        return self.total - self.labeled


class LSClient:
    def __init__(self, url: str, api_key: str):
        if not api_key:
            raise LSError("API anahtarı boş: Label Studio > Account & Settings > Personal Access Token")
        self.url = url.rstrip("/")
        self.client = LabelStudio(base_url=self.url, api_key=api_key)

    def _call(self, fn, *a, _retry: bool = False, **kw):
        try:
            try:
                return fn(*a, **kw)
            except (httpx.RemoteProtocolError, httpx.ReadError):
                # uWSGI işçisi yenilenince açık bağlantı kapanır; yalnız okuma isteklerini bir kez yinele
                if not _retry:
                    raise
                return fn(*a, **kw)
        except httpx.ConnectError as e:
            raise LSError(f"Label Studio'ya bağlanılamadı ({self.url}); `label-studio start` çalışıyor mu?") from e
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 401:
                raise LSError("API anahtarı geçersiz ya da süresi dolmuş (401)") from e
            if code == 404:
                raise LSError("Label Studio'da bulunamadı (404): proje silinmiş olabilir") from e
            raise

    def check(self) -> str:
        """Bağlantı ve anahtar testi -> kullanıcı e-postası."""
        return self._call(self.client.users.whoami).email

    # -------------------------------------------------------------- proje
    def projects(self) -> list[tuple[int, str]]:
        return [(p.id, p.title) for p in self._call(self.client.projects.list)]

    def get_or_create_project(self, project_id: int | None, title: str, label_config: str) -> int:
        if project_id:
            return self._call(self.client.projects.get, int(project_id)).id
        p = self._call(self.client.projects.create, title=title, label_config=label_config,
                       show_collab_predictions=True)
        return p.id

    def project_link(self, project_id: int) -> str:
        return f"{self.url}/projects/{project_id}/data"

    # -------------------------------------------------------------- görevler
    def tasks(self, project_id: int) -> list:
        return list(self._call(self.client.tasks.list, project=project_id, fields="all"))

    def task_summaries(self, project_id: int) -> list[dict]:
        """Hafif görev listesi: [{id, annotations (iptal edilmemiş), versions}]. Maskeleri (tahmin/etiket sonuçlarını)
        indirmez; yüzlerce görevli projede tasks()'tan çok daha hızlıdır."""
        headers = {k: v for k, v in self.client._client_wrapper.get_headers().items() if k.lower() != "content-type"}
        out, page = [], 1
        while True:
            try:
                r = httpx.get(f"{self.url}/api/tasks", headers=headers, timeout=60, params={
                    "project": project_id, "page": page, "page_size": 1000,
                    "include": "id,total_annotations,predictions_model_versions"})
            except httpx.ConnectError as e:
                raise LSError(f"Label Studio'ya bağlanılamadı ({self.url})") from e
            if r.status_code == 404 and page > 1:
                break
            if r.status_code >= 300:
                raise LSError(f"görev listesi alınamadı ({r.status_code}): {r.text[:200]}")
            body = r.json()
            tasks = body.get("tasks", [])
            for t in tasks:
                v = t.get("predictions_model_versions") or ""
                versions = v if isinstance(v, list) else v.split(",")
                out.append({"id": t["id"], "annotations": int(t.get("total_annotations") or 0),
                            "versions": sorted({str(x).strip() for x in versions if x and str(x).strip()})})
            if not tasks or len(out) >= int(body.get("total", 0)):
                break
            page += 1
        return out

    def task(self, task_id: int):
        return self._call(self.client.tasks.get, task_id, _retry=True)

    def delete_task(self, task_id: int) -> None:
        self._call(self.client.tasks.delete, task_id)

    def delete_project(self, project_id: int) -> None:
        self._call(self.client.projects.delete, project_id)

    def tasks_by_name(self, project_id: int) -> dict[str, object]:
        return {original_name(t.data): t for t in self.tasks(project_id)}

    def upload_image(self, project_id: int, path: Path) -> int:
        """Bir görüntüyü yükleyip oluşan görevin numarasını döndürür."""
        headers = {k: v for k, v in self.client._client_wrapper.get_headers().items() if k.lower() != "content-type"}
        try:
            with open(path, "rb") as fh:
                r = httpx.post(f"{self.url}/api/projects/{project_id}/import", params={"return_task_ids": "true"},
                               headers=headers, files=[("file", (path.name, fh))], timeout=120)
        except httpx.ConnectError as e:
            raise LSError(f"Label Studio'ya bağlanılamadı ({self.url})") from e
        if r.status_code == 401:
            raise LSError("API anahtarı geçersiz ya da süresi dolmuş (401)")
        if r.status_code >= 300:
            raise LSError(f"yükleme başarısız ({r.status_code}): {r.text[:200]}")
        return r.json()["task_ids"][0]

    def upload_file(self, project_id: int, path: Path) -> str:
        """Görev oluşturmadan dosya yükler; Label Studio'daki adresini (/data/upload/...) döndürür."""
        headers = {k: v for k, v in self.client._client_wrapper.get_headers().items() if k.lower() != "content-type"}
        try:
            with open(path, "rb") as fh:
                r = httpx.post(f"{self.url}/api/projects/{project_id}/import", params={"commit_to_project": "false"},
                               headers=headers, files=[("file", (path.name, fh))], timeout=120)
            r.raise_for_status()
            fid = r.json()["file_upload_ids"][0]
            f = httpx.get(f"{self.url}/api/import/file-upload/{fid}", headers=headers, timeout=30)
            f.raise_for_status()
        except httpx.HTTPError as e:
            raise LSError(f"dosya yüklenemedi: {e}") from e
        return "/data/" + f.json()["file"]

    def set_task_image(self, task_id: int, url: str, data: dict | None = None) -> None:
        """Görevin görüntü adresini değiştirir; etiketler ve ön etiketler aynen kalır."""
        data = dict(self.task(task_id).data if data is None else data)
        data["image"] = url
        self._call(self.client.tasks.update, task_id, data=data)

    def add_prediction(self, task_id: int, pred: dict) -> int:
        p = self._call(self.client.predictions.create, task=task_id, result=pred["result"], score=pred["score"],
                       model_version=pred["model_version"])
        return p.id

    def set_model_version(self, project_id: int, version: str) -> None:
        """Projede gösterilecek ön etiket sürümü (Settings > Annotation > Prelabeling)."""
        self._call(self.client.projects.update, project_id, model_version=version, show_collab_predictions=True)

    def status(self, project_id: int) -> Status:
        ts = self.tasks(project_id)
        return Status(len(ts), sum(1 for t in ts if latest_annotation(t) is not None),
                      sum(1 for t in ts if (t.total_predictions or 0) > 0))


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def latest_annotation(task) -> dict | None:
    """Görevin iptal edilmemiş en son insan etiketi (sonuç listesiyle) ya da None."""
    anns = [a for a in (task.annotations or []) if not _get(a, "was_cancelled")]
    if not anns:
        return None
    a = max(anns, key=lambda a: (str(_get(a, "updated_at") or _get(a, "created_at") or ""), _get(a, "id") or 0))
    return {"id": _get(a, "id"), "result": _get(a, "result") or [], "completed_by": _get(a, "completed_by"),
            "updated_at": str(_get(a, "updated_at") or "")}


def prediction_for(task, model_version: str | None = None) -> dict | None:
    """Görevin (verilen sürümdeki) en son model tahmini."""
    ps = [p for p in (task.predictions or []) if model_version is None or _get(p, "model_version") == model_version]
    if not ps:
        return None
    p = max(ps, key=lambda p: _get(p, "id") or 0)
    return {"id": _get(p, "id"), "result": _get(p, "result") or [], "model_version": _get(p, "model_version"),
            "score": _get(p, "score")}
