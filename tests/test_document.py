from ai_diffusion.document import KritaDocument


def test_document_active(qtapp):
    async def main():
        doc = KritaDocument.active()
        assert doc is not None

    qtapp.run(main())
