/* --- Minimal UX boost (optional) --- */
:root{
  --tap-min: 44px;
}
/* ボタンと入力の最低高さ */
.stButton > button, .stTextInput > div > div > input,
.stTextInput input, .stFileUploader label, .stCameraInput label {
  min-height: var(--tap-min);
  font-weight: 600;
}

/* フォーカスリング（アクセシビリティ） */
:where(button, input, select, textarea):focus-visible {
  outline: 3px solid color-mix(in srgb, var(--mdc-primary) 60%, white);
  outline-offset: 2px;
  border-radius: 12px;
}

/* ダークテーマ準備（OS設定に追従） */
@media (prefers-color-scheme: dark) {
  :root{
    --mdc-bg:#0f1115;
    --mdc-surface:#171a21;
    --mdc-on-primary:#ffffff;
  }
  body{ background: var(--mdc-bg); }
  .md-card{ background: var(--mdc-surface); border-color: rgba(255,255,255,.06); }
}
