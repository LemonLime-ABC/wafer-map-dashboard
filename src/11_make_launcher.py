# %% [markdown]
# # 11_make_launcher — 배포된 대시보드로 가는 "공유용 파일" 생성
#
# 다른 사람에게 파일 하나만 보내면, 그걸 열었을 때 배포된 사이트로
# 바로 갈 수 있게 하는 것이 목적이다.
#
# 만드는 것 2가지:
#   1. .html — 어디서든 열린다(윈도우·맥·폰). QR 코드도 같이 들어간다.
#              노트북으로 연 사람이 자기 폰으로 스캔할 수 있다.
#   2. .url  — 윈도우 인터넷 바로가기. 더블클릭하면 기본 브라우저로 열린다.
#
# 사용법:
#   python src/11_make_launcher.py
#   (아래 APPS 목록의 주소만 실제 배포 주소로 바꾸면 된다)

# %%
import sys
import base64
from io import BytesIO
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import qrcode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARE = PROJECT_ROOT / "share"
SHARE.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 여기만 고치면 된다 — 배포된 실제 주소로 교체
# ---------------------------------------------------------------
APPS = [
    {
        "key": "wafer",
        "name": "웨이퍼 맵 불량 패턴 분류",
        "desc": "WM-811K · CNN으로 불량 패턴 9종을 분류하고 Grad-CAM으로 판단 근거를 확인",
        "url": "https://wafer-map-dashboard-mhappwnsvbeuuwrfvbbwdhu.streamlit.app/",
        "color": "#A63A20",
    },
    {
        "key": "secom",
        "name": "공정 센서 수율 이상 진단",
        "desc": "SECOM · 센서 590개로 불량 확률을 예측하고 SHAP으로 원인 센서를 지목",
        "url": "https://secom-yield-dashboard-fvkivfpcdqxk7jgx5alxew.streamlit.app/",
        "color": "#1B5E8C",
    },
]

TEAM = "반도체 수율반장 · 오승윤, 안익제 · 서울시립대학교 전자전기컴퓨터공학부"


# ---------------------------------------------------------------
# QR 코드를 PNG -> base64로 만들어 HTML 안에 직접 박는다.
# 이렇게 하면 HTML 파일 하나만 보내도 이미지가 깨지지 않는다
# (외부 파일을 참조하면 파일 하나만 보냈을 때 QR이 안 뜬다).
# ---------------------------------------------------------------
def qr_data_uri(url: str, box: int = 9) -> str:
    qr = qrcode.QRCode(box_size=box, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a1a", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def icon_data_uri() -> str:
    p = PROJECT_ROOT / "assets" / "app_icon.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


live = [a for a in APPS if a["url"].strip()]
if not live:
    print("!! APPS에 주소가 하나도 없습니다. url 값을 채우고 다시 실행하세요.")
    sys.exit(1)

cards = []
for a in live:
    cards.append(f"""
    <section class="card">
      <div class="info">
        <h2 style="color:{a['color']}">{a['name']}</h2>
        <p class="desc">{a['desc']}</p>
        <a class="btn" style="background:{a['color']}" href="{a['url']}"
           target="_blank" rel="noopener">대시보드 열기</a>
        <p class="url">{a['url']}</p>
      </div>
      <div class="qr">
        <img src="{qr_data_uri(a['url'])}" alt="{a['name']} QR 코드">
        <span>휴대폰으로 스캔</span>
      </div>
    </section>""")

icon = icon_data_uri()
icon_tag = f'<img class="appicon" src="{icon}" alt="">' if icon else ""

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>반도체 수율 진단 대시보드</title>
{f'<link rel="icon" href="{icon}">' if icon else ""}
<style>
  :root{{
    --bg:#F5F7F8; --surface:#fff; --ink:#161D20; --muted:#61737A; --line:#E1E8EA;
  }}
  @media (prefers-color-scheme: dark){{
    :root{{ --bg:#0E1416; --surface:#161F22; --ink:#E6ECED; --muted:#93A5AA; --line:#263336; }}
  }}
  *{{box-sizing:border-box}}
  body{{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;
    line-height:1.7; padding:48px 20px;
  }}
  .wrap{{max-width:760px;margin:0 auto}}
  header{{text-align:center;margin-bottom:40px}}
  .appicon{{width:76px;height:76px;margin-bottom:14px}}
  h1{{font-size:1.7rem;margin:0 0 8px;letter-spacing:-.02em}}
  .team{{color:var(--muted);font-size:.9rem;margin:0}}
  .card{{
    background:var(--surface); border:1px solid var(--line); border-radius:14px;
    padding:26px; margin-bottom:20px;
    display:flex; gap:26px; align-items:center; flex-wrap:wrap;
  }}
  .info{{flex:1 1 300px;min-width:260px}}
  h2{{font-size:1.2rem;margin:0 0 8px}}
  .desc{{color:var(--muted);font-size:.93rem;margin:0 0 18px}}
  .btn{{
    display:inline-block;color:#fff;text-decoration:none;font-weight:600;
    padding:11px 26px;border-radius:8px;font-size:1rem;
  }}
  .btn:hover,.btn:focus-visible{{filter:brightness(1.1)}}
  .url{{font-size:.78rem;color:var(--muted);margin:12px 0 0;word-break:break-all}}
  .qr{{text-align:center;flex:0 0 auto}}
  .qr img{{width:132px;height:132px;display:block;border-radius:8px;background:#fff;padding:6px}}
  .qr span{{font-size:.76rem;color:var(--muted);display:block;margin-top:7px}}
  .how{{
    background:var(--surface); border:1px solid var(--line);
    border-radius:14px; padding:0 26px; margin-top:26px;
  }}
  .how summary{{
    cursor:pointer; padding:18px 0; font-weight:600; font-size:.97rem;
    list-style:none;
  }}
  .how summary::-webkit-details-marker{{display:none}}
  .how summary::before{{content:"＋  "; color:var(--muted)}}
  .how[open] summary::before{{content:"−  "}}
  .how-body{{padding-bottom:22px}}
  .how h3{{font-size:.92rem;margin:14px 0 6px}}
  .how ol{{margin:0 0 10px;padding-left:22px}}
  .how li{{margin-bottom:6px;font-size:.9rem}}
  .note{{color:var(--muted);font-size:.84rem}}
  footer{{
    text-align:center;color:var(--muted);font-size:.82rem;
    margin-top:34px;padding-top:20px;border-top:1px solid var(--line);
  }}
  :focus-visible{{outline:2px solid #A63A20;outline-offset:3px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    {icon_tag}
    <h1>반도체 수율 진단 대시보드</h1>
    <p class="team">{TEAM}</p>
  </header>
{"".join(cards)}

  <details class="how">
    <summary>바탕화면·홈 화면에 앱처럼 두는 방법</summary>
    <div class="how-body">
      <h3>PC (Chrome / Edge)</h3>
      <ol>
        <li>위 버튼으로 대시보드를 연다</li>
        <li>주소창 오른쪽 <b>⋮</b> 메뉴 → <b>캐스트, 저장, 공유</b> → <b>페이지를 앱으로 설치</b><br>
            <span class="note">(메뉴 이름은 버전에 따라 "바로가기 만들기"일 수 있다.
            이 경우 <b>창으로 열기</b>에 체크한다)</span></li>
        <li>바탕화면에 아이콘이 생기고, <b>주소창 없는 독립 창</b>으로 열린다</li>
      </ol>
      <h3>휴대폰</h3>
      <ol>
        <li>아이폰(Safari): 하단 <b>공유</b> → <b>홈 화면에 추가</b></li>
        <li>안드로이드(Chrome): 우측 상단 <b>⋮</b> → <b>홈 화면에 추가</b></li>
      </ol>
      <p class="note">
        어느 방식이든 인터넷 연결은 필요하다. 앱을 기기에 설치하는 것이 아니라,
        접속을 편하게 만드는 바로가기다.
      </p>
    </div>
  </details>

  <footer>
    이 파일은 인터넷 연결이 필요합니다. 버튼을 누르거나 QR을 스캔하면 대시보드가 열립니다.
  </footer>
</div>
</body>
</html>"""

html_path = SHARE / "대시보드_바로가기.html"
html_path.write_text(html, encoding="utf-8")
print(f"[1] HTML 런처: {html_path.name}  ({html_path.stat().st_size/1024:.1f} KB)")

# ---------------------------------------------------------------
# 바탕화면 아이콘용 .ico 만들기
#
# 윈도우 바로가기(.url)에 그림 아이콘을 붙이려면 .png가 아니라 .ico가
# 필요하다. .ico는 여러 크기를 한 파일에 담는 형식이라, 바탕화면(48px)과
# 작업표시줄(16px) 등에서 각각 알맞은 크기가 쓰인다.
# ---------------------------------------------------------------
from PIL import Image  # noqa: E402

ICO_PATH = SHARE / "wafer_icon.ico"
src_png = PROJECT_ROOT / "assets" / "app_icon.png"
if src_png.exists():
    im = Image.open(src_png).convert("RGBA")
    im.save(ICO_PATH, format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"[2] 아이콘: {ICO_PATH.name}  ({ICO_PATH.stat().st_size/1024:.1f} KB)")

# ---------------------------------------------------------------
# 윈도우 .url 바로가기 — 내용은 단순한 INI 형식이다.
# 줄바꿈은 반드시 CRLF(\r\n)여야 윈도우가 제대로 인식한다.
#
# 주의: IconFile은 그 PC에 실제로 존재하는 절대 경로여야 한다.
# 그래서 이 아이콘은 "내 바탕화면"에서만 보이고, 남에게 파일을 보내면
# 상대방 PC에는 그 경로가 없어서 기본 아이콘으로 뜬다.
# 남에게 줄 때는 HTML 런처를 쓰는 게 맞다.
# ---------------------------------------------------------------
for a in live:
    # IconFile은 파일명만 쓴다(절대경로 X). 윈도우가 .url 파일이 있는
    # 폴더를 기준으로 찾으므로, .ico를 같은 폴더에 두면 남의 PC에서도
    # 아이콘이 뜬다. 절대경로로 쓰면 그 경로가 없는 PC에서는 안 뜬다.
    lines = ["[InternetShortcut]", f"URL={a['url']}"]
    if ICO_PATH.exists():
        lines += [f"IconFile={ICO_PATH.name}", "IconIndex=0"]
    content = "\r\n".join(lines) + "\r\n"
    p = SHARE / f"{a['name']}.url"
    p.write_text(content, encoding="utf-8", newline="")
    print(f"[3] 윈도우 바로가기: {p.name}")

# QR을 이미지 파일로도 따로 저장 (포스터·발표자료에 넣을 때 씀)
for a in live:
    qr = qrcode.QRCode(box_size=14, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(a["url"])
    qr.make(fit=True)
    p = SHARE / f"QR_{a['key']}.png"
    qr.make_image(fill_color="#1a1a1a", back_color="white").save(p)
    print(f"[4] QR 이미지: {p.name}")

print(f"\n완료 — {SHARE}")
print("주소가 바뀌면 이 파일 위쪽 APPS의 url만 고치고 다시 실행하면 됩니다.")
