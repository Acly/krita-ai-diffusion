from ai_diffusion.document import KritaDocument

from .conftest import qtapp


@qtapp
async def test_document_active():
    doc = KritaDocument.active()
    assert doc is not None
