"""
Generischer Bild-Cache – speichert und liefert Bilder aus verschiedenen Quellen.

Tabellen:  image_cache       – Rohe Bild- und Thumbnail-Daten
           image_reference   – Verknüpfung von Bildern mit beliebigen Entitäten

Alle DB-Operationen verwenden get_connection() und folgen dem bestehenden
try/except + logger.warning()-Muster.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_thumbnail(image_data: bytes, max_size: int = 256) -> bytes | None:
    """Erstellt ein WebP-Vorschaubild mit Pillow.

    Gibt None zurück, wenn Pillow nicht verfügbar ist oder die Konvertierung
    fehlschlägt.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        logger.debug("Pillow nicht verfügbar – kein Thumbnail generiert.")
        return None
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=75)
            return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Thumbnail-Generierung fehlgeschlagen: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def cache_image(
    source: str,
    source_key: str,
    image_data: bytes,
    mime_type: str,
    *,
    thumb_data: bytes | None = None,
    width: int | None = None,
    height: int | None = None,
    alt_text: str | None = None,
    original_url: str | None = None,
    ttl_days: int = 90,
) -> int | None:
    """Speichert oder aktualisiert ein Bild im Cache.

    Gibt die image_id zurück oder None bei Fehler.
    """
    # Thumbnail automatisch erzeugen, falls nicht übergeben
    if thumb_data is None:
        thumb_data = _make_thumbnail(image_data)

    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO image_cache "
                "(source, source_key, mime_type, image_data, thumb_data, "
                " width, height, alt_text, original_url, ttl_days) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "  mime_type    = VALUES(mime_type), "
                "  image_data   = VALUES(image_data), "
                "  thumb_data   = VALUES(thumb_data), "
                "  width        = VALUES(width), "
                "  height       = VALUES(height), "
                "  alt_text     = VALUES(alt_text), "
                "  original_url = VALUES(original_url), "
                "  ttl_days     = VALUES(ttl_days), "
                "  fetched_at   = CURRENT_TIMESTAMP",
                (
                    source, source_key, mime_type, image_data, thumb_data,
                    width, height, alt_text, original_url, ttl_days,
                ),
            )
            conn.commit()
            # Vorhandene ID nach UPSERT ermitteln
            if cursor.lastrowid:
                image_id = cursor.lastrowid
            else:
                cursor.execute(
                    "SELECT id FROM image_cache WHERE source = %s AND source_key = %s",
                    (source, source_key),
                )
                row = cursor.fetchone()
                image_id = row[0] if row else None
            cursor.close()
        return image_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache_image fehlgeschlagen (%s/%s): %s", source, source_key, exc)
        return None


def get_image(image_id: int) -> dict | None:
    """Gibt ein Bild anhand seiner ID zurück und erhöht access_count.

    Rückgabefelder: id, source, source_key, mime_type, image_data, thumb_data,
                    width, height, alt_text, original_url, fetched_at, ttl_days,
                    access_count, last_accessed
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, source, source_key, mime_type, image_data, thumb_data, "
                "       width, height, alt_text, original_url, fetched_at, ttl_days, "
                "       access_count, last_accessed "
                "FROM image_cache WHERE id = %s",
                (image_id,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.close()
                return None
            result = {
                "id": row[0],
                "source": row[1],
                "source_key": row[2],
                "mime_type": row[3],
                "image_data": row[4],
                "thumb_data": row[5],
                "width": row[6],
                "height": row[7],
                "alt_text": row[8],
                "original_url": row[9],
                "fetched_at": row[10],
                "ttl_days": row[11],
                "access_count": row[12],
                "last_accessed": row[13],
            }
            # Zugriffszähler aktualisieren
            cursor.execute(
                "UPDATE image_cache "
                "SET access_count = access_count + 1, last_accessed = %s "
                "WHERE id = %s",
                (datetime.now(timezone.utc), image_id),
            )
            conn.commit()
            cursor.close()
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_image(%s) fehlgeschlagen: %s", image_id, exc)
        return None


def get_image_by_source(source: str, source_key: str) -> dict | None:
    """Gibt ein Bild anhand von Quelle + Schlüssel zurück."""
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM image_cache WHERE source = %s AND source_key = %s",
                (source, source_key),
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        return get_image(row[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_image_by_source(%s/%s) fehlgeschlagen: %s", source, source_key, exc
        )
        return None


def get_images_for_ref(ref_table: str, ref_id: int) -> list[dict]:
    """Gibt alle Bilder zurück, die mit einer bestimmten Entität verknüpft sind.

    Die Bilder werden ohne Rohdaten (image_data/thumb_data) zurückgegeben,
    um den Speicherbedarf gering zu halten.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ic.id, ic.source, ic.source_key, ic.mime_type, "
                "       ic.width, ic.height, ic.alt_text, ic.original_url, "
                "       ic.fetched_at, ir.context "
                "FROM image_reference ir "
                "JOIN image_cache ic ON ic.id = ir.image_id "
                "WHERE ir.ref_table = %s AND ir.ref_id = %s "
                "ORDER BY ir.created_at",
                (ref_table, ref_id),
            )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {
                "id": r[0],
                "source": r[1],
                "source_key": r[2],
                "mime_type": r[3],
                "width": r[4],
                "height": r[5],
                "alt_text": r[6],
                "original_url": r[7],
                "fetched_at": r[8],
                "context": r[9],
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_images_for_ref(%s/%s) fehlgeschlagen: %s", ref_table, ref_id, exc
        )
        return []


def link_image(
    image_id: int,
    ref_table: str,
    ref_id: int,
    context: str | None = None,
) -> None:
    """Verknüpft ein Bild mit einer Entität (dedupliziert).

    Wenn die Verknüpfung bereits existiert, wird sie nicht erneut angelegt.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT IGNORE INTO image_reference (image_id, ref_table, ref_id, context) "
                "VALUES (%s, %s, %s, %s)",
                (image_id, ref_table, ref_id, context),
            )
            conn.commit()
            cursor.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "link_image(%s → %s/%s) fehlgeschlagen: %s", image_id, ref_table, ref_id, exc
        )


def fetch_and_cache_url(
    url: str,
    source: str,
    source_key: str,
    *,
    alt_text: str | None = None,
    ttl_days: int = 90,
    make_thumbnail: bool = True,
) -> int | None:
    """Lädt ein Bild von einer URL herunter, generiert ein Thumbnail und speichert es.

    Gibt die image_id zurück oder None bei Fehler.
    Private IP-Bereiche und localhost sind aus Sicherheitsgründen gesperrt (SSRF-Schutz).
    """
    import ipaddress  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415

    # SSRF-Schutz: private/lokale Adressen sperren
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        if hostname:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                logger.warning(
                    "fetch_and_cache_url: Zugriff auf private Adresse gesperrt: %r", url
                )
                return None
    except ValueError:
        # Kein gültiges IP-Literal – Hostname-Auflösung erfolgt im requests-Layer
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_and_cache_url: URL-Prüfung fehlgeschlagen: %s", exc)

    try:
        import requests  # noqa: PLC0415
        resp = requests.get(url, timeout=15, stream=True, verify=True)
        resp.raise_for_status()
        image_data = resp.content
        mime_type = resp.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_and_cache_url: Download von %r fehlgeschlagen: %s", url, exc)
        return None

    # Bildgröße ermitteln (optional, Pillow)
    width: int | None = None
    height: int | None = None
    try:
        from PIL import Image  # noqa: PLC0415
        with Image.open(io.BytesIO(image_data)) as img:
            width, height = img.size
    except Exception:  # noqa: BLE001
        pass

    thumb_data = _make_thumbnail(image_data) if make_thumbnail else None

    return cache_image(
        source,
        source_key,
        image_data,
        mime_type,
        thumb_data=thumb_data,
        width=width,
        height=height,
        alt_text=alt_text,
        original_url=url,
        ttl_days=ttl_days,
    )


def cleanup_expired() -> int:
    """Löscht Bilder, deren TTL abgelaufen ist.

    Gibt die Anzahl der gelöschten Zeilen zurück.
    """
    try:
        from db.connection import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM image_cache "
                "WHERE ttl_days IS NOT NULL "
                "  AND fetched_at < DATE_SUB(NOW(), INTERVAL ttl_days DAY)"
            )
            deleted = cursor.rowcount
            conn.commit()
            cursor.close()
        logger.info("cleanup_expired: %d abgelaufene Bilder gelöscht.", deleted)
        return deleted
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup_expired fehlgeschlagen: %s", exc)
        return 0
