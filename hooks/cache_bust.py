"""MkDocs hook: cache-busting для локальных extra_javascript / extra_css.

К каждому локальному ассету дописывается ?h=<короткий md5 содержимого>.
Хеш меняется только при изменении файла -> неизменные файлы остаются в кеше
браузера, а отредактированные подтягиваются заново после деплоя. Внешние URL
(MathJax CDN и т.п.) не трогаются.
"""

import hashlib
import os


def _content_hash(abs_path):
    try:
        with open(abs_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return None


def _busted(path, docs_dir):
    if not path or "://" in path:
        return None
    h = _content_hash(os.path.join(docs_dir, path))
    if not h:
        return None
    sep = "&" if "?" in path else "?"
    return path + sep + "h=" + h


def on_config(config):
    docs_dir = config["docs_dir"]

    css = config.get("extra_css") or []
    for i, item in enumerate(css):
        new = _busted(item, docs_dir)
        if new:
            css[i] = new

    js = config.get("extra_javascript") or []
    for i, item in enumerate(js):
        # В MkDocs 1.5+ элементы могут быть ExtraScriptValue (с .path) или строками
        if isinstance(item, str):
            new = _busted(item, docs_dir)
            if new:
                js[i] = new
        else:
            new = _busted(getattr(item, "path", None), docs_dir)
            if new:
                item.path = new

    return config
