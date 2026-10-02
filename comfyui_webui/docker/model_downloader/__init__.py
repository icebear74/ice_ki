"""Image-owned, authenticated replacement for the pinned downloader backend."""
from server import PromptServer

from .backend import ModelDownloader

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

downloader = ModelDownloader()
downloader.register(PromptServer.instance.routes)


async def cleanup_downloader(app):
    await downloader.shutdown()


PromptServer.instance.app.on_cleanup.append(cleanup_downloader)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
