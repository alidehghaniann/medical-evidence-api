from fastapi.testclient import TestClient

from app.main import app


def test_demo_uses_unicode_directionality_and_keeps_latin_metadata_isolated():
    with TestClient(app) as client:
        html = client.get("/demo").text

    assert '<html lang="en" dir="ltr">' in html
    assert 'dir="auto"' in html
    assert 'dir="ltr"' in html
    assert "unicode-bidi: isolate" in html
    assert "detectedLanguage(query.value)" in html
    assert "textContent" in html
    assert "reverse()" not in html
    assert "h4>" not in html
    assert "فارسی" in html
