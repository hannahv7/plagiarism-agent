import { useState, useRef, useCallback, useEffect } from "react";

// ── Types ──────────────────────────────────────────────────────────────────
type MatchType = "Exact" | "Paraphrased";
interface Match {
  id: number;
  type: MatchType;
  similarity: number;
  suspectedSentence: number;
  originalSentence: number;
  concepts: string[];
  explanation: string;
  suspectedText: string;
  originalText: string;
}

// ── Mock data ──────────────────────────────────────────────────────────────
const MOCK_MATCHES: Match[] = [
  {
    id: 1,
    type: "Exact",
    similarity: 0.97,
    suspectedSentence: 3,
    originalSentence: 1,
    concepts: ["informed consent", "patient rights", "autonomy"],
    explanation:
      "Flagged because the sentence is reproduced verbatim with no structural modification. Core legal terminology and sentence structure are identical.",
    suspectedText:
      "Patients retain the right to refuse treatment under the principle of informed consent, which prioritizes individual autonomy above institutional directive.",
    originalText:
      "Patients retain the right to refuse treatment under the principle of informed consent, which prioritizes individual autonomy above institutional directive.",
  },
  {
    id: 2,
    type: "Paraphrased",
    similarity: 0.89,
    suspectedSentence: 12,
    originalSentence: 8,
    concepts: ["healthcare", "medical ethics", "patient"],
    explanation:
      "Flagged because core concepts overlap with high semantic similarity despite surface-level rewording. Key clinical terms appear in altered order with synonymous substitutions.",
    suspectedText:
      "The ethical obligations of healthcare providers extend to ensuring equitable access to medical interventions for all patient populations regardless of socioeconomic background.",
    originalText:
      "Medical professionals are ethically bound to guarantee that clinical treatments remain accessible to every patient group, irrespective of their financial circumstances.",
  },
  {
    id: 3,
    type: "Exact",
    similarity: 0.99,
    suspectedSentence: 17,
    originalSentence: 14,
    concepts: ["clinical trial", "randomized", "double-blind"],
    explanation:
      "The methodology description is copied without attribution. Statistical framing, experimental design terminology, and sentence structure are reproduced identically.",
    suspectedText:
      "A randomized double-blind clinical trial was conducted across six tertiary care centers to evaluate the efficacy of the proposed intervention over a 24-month follow-up period.",
    originalText:
      "A randomized double-blind clinical trial was conducted across six tertiary care centers to evaluate the efficacy of the proposed intervention over a 24-month follow-up period.",
  },
  {
    id: 4,
    type: "Paraphrased",
    similarity: 0.82,
    suspectedSentence: 21,
    originalSentence: 19,
    concepts: ["biomarker", "diagnostic", "sensitivity"],
    explanation:
      "Flagged due to shared diagnostic framework and identical statistical claims despite different phrasing. The biomarker sensitivity figure and conclusion are lifted directly.",
    suspectedText:
      "The identified biomarker demonstrated diagnostic sensitivity of 94.2% in early-stage detection, making it a clinically valuable screening tool for high-risk populations.",
    originalText:
      "With a diagnostic sensitivity reaching 94.2%, this biomarker serves as a highly effective early detection indicator for individuals in high-risk clinical categories.",
  },
  {
    id: 5,
    type: "Paraphrased",
    similarity: 0.78,
    suspectedSentence: 26,
    originalSentence: 22,
    concepts: ["mortality", "intervention", "outcome"],
    explanation:
      "Structural and thematic overlap detected. Both sentences assert identical causal relationships between early intervention and mortality reduction using equivalent quantitative claims.",
    suspectedText:
      "Early therapeutic intervention was associated with a 37% reduction in all-cause mortality among patients presenting within the first 48 hours of symptom onset.",
    originalText:
      "Patients who received treatment within 48 hours of initial symptom presentation exhibited a 37% decrease in overall mortality compared to delayed-intervention cohorts.",
  },
  {
    id: 6,
    type: "Exact",
    similarity: 0.96,
    suspectedSentence: 29,
    originalSentence: 27,
    concepts: ["peer review", "publication bias", "evidence"],
    explanation:
      "Near-verbatim reproduction of a methodological critique. Only minor punctuation changes detected between source and suspected passages.",
    suspectedText:
      "Publication bias remains a critical limitation in systematic reviews, as studies with null results are disproportionately excluded from peer-reviewed literature.",
    originalText:
      "Publication bias remains a critical limitation in systematic reviews, as studies with null results are disproportionately excluded from peer-reviewed literature.",
  },
];

const HEATMAP_EXACT = new Set([3, 17, 29]);
const HEATMAP_PARA = new Set([12, 21, 26, 6, 9, 15, 18, 24]);
function getHeatmapType(n: number): "exact" | "paraphrased" | "original" {
  if (HEATMAP_EXACT.has(n)) return "exact";
  if (HEATMAP_PARA.has(n)) return "paraphrased";
  return "original";
}

// ── Theme ──────────────────────────────────────────────────────────────────
type ThemeMode = "light" | "dark";
const lightTheme = {
  bg: "#fcfaf8",
  card: "#ffffff",
  cardHover: "#ffffff",
  border: "#e5e7eb",
  borderActive: "#111827",
  borderDashed: "#d1d5db",
  text: "#111827",
  textMuted: "#6b7280",
  textFaint: "#9ca3af",
  textLight: "#ffffff",
  subBg: "#fafaf9",
  track: "#f3f4f6",
  gaugeTrack: "#f3f4f6",
};
const darkTheme = {
  bg: "#0a0a0b",
  card: "#171719",
  cardHover: "#1e1e20",
  border: "#232326",
  borderActive: "#f5f5f5",
  borderDashed: "#3f3f47",
  text: "#f5f5f4",
  textMuted: "#a1a1aa",
  textFaint: "#71717a",
  textLight: "#0a0a0b",
  subBg: "#1c1c1f",
  track: "#27272a",
  gaugeTrack: "#27272a",
};

// ── Sub-components ─────────────────────────────────────────────────────────
function UploadCard({
  label, file, onFile, text, onText, theme
}: {
  label: string; file: File | null; onFile: (f: File) => void; text: string; onText: (t: string) => void; theme: typeof lightTheme;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState<"file" | "paste">("paste");

  const tabStyle = (active: boolean) => ({
    fontFamily: "Inter, sans-serif",
    fontSize: 12,
    fontWeight: 500,
    color: active ? theme.text : theme.textFaint,
    background: "none",
    border: "none",
    borderBottom: active ? `1.5px solid ${theme.text}` : "1.5px solid transparent",
    padding: "6px 2px",
    cursor: "pointer",
  } as React.CSSProperties);

  return (
    <div style={{ flex: 1, minWidth: 0 }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) { setMode("file"); onFile(f); } }}
    >
      <div style={{
        border: `1.5px ${mode === "file" ? "dashed" : "solid"} ${dragging ? theme.borderActive : (file && mode === "file") || (text && mode === "paste") ? theme.borderActive : theme.borderDashed}`,
        borderRadius: 12,
        background: theme.card,
        overflow: "hidden",
        transition: "border-color 0.15s ease, background 0.2s",
      }}>
        <div style={{ padding: "14px 18px 0", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontFamily: "'Fraunces', serif", fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: theme.text }}>{label}</span>
          <div style={{ display: "flex", gap: 14 }}>
            <button style={tabStyle(mode === "paste")} onClick={() => setMode("paste")}>Paste text</button>
            <button style={tabStyle(mode === "file")} onClick={() => setMode("file")}>Upload file</button>
          </div>
        </div>
        {mode === "paste" ? (
          <div style={{ padding: "10px 18px 16px" }}>
            <textarea value={text} onChange={(e) => onText(e.target.value)} placeholder="Paste or type your document text here…"
              style={{ width: "100%", minHeight: 110, resize: "vertical", fontFamily: "Inter, sans-serif", fontSize: 13, color: theme.text, background: "transparent", border: "none", outline: "none", lineHeight: 1.6, padding: 0 }}
            />
            {text && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 6, paddingTop: 8, borderTop: `1px solid ${theme.track}` }}>
                <span style={{ fontFamily: "Inter, sans-serif", fontSize: 11, color: theme.textFaint }}>{text.trim().split(/\s+/).length} words · {text.length} chars</span>
                <button onClick={() => onText("")} style={{ fontFamily: "Inter, sans-serif", fontSize: 11, color: theme.textFaint, background: "none", border: "none", cursor: "pointer", padding: 0 }}>Clear</button>
              </div>
            )}
          </div>
        ) : (
          <div onClick={() => inputRef.current?.click()} style={{ padding: "24px 18px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10, cursor: "pointer", minHeight: 110, justifyContent: "center" }}>
            <div style={{ width: 36, height: 36, borderRadius: "50%", background: theme.text, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke={theme.card} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" /><polyline points="14,2 14,8 20,8" stroke={theme.card} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </div>
            {file ? (
              <span style={{ fontFamily: "Inter, sans-serif", fontSize: 13, color: theme.text, fontWeight: 500 }}>{file.name}</span>
            ) : (
              <span style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.textMuted }}>{dragging ? "Drop to upload" : "Click or drag a file — PDF, TXT up to 200MB"}</span>
            )}
          </div>
        )}
      </div>
      <input ref={inputRef} type="file" accept=".pdf,.txt" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
    </div>
  );
}

function GaugeSVG({ value, theme }: { value: number; theme: typeof lightTheme }) {
  const R = 80; const cx = 110; const cy = 100;
  const pct = value / 100; const angle = Math.PI - pct * Math.PI;
  const trackStart = { x: cx - R, y: cy }; const trackEnd = { x: cx + R, y: cy };
  const arcEndX = cx + R * Math.cos(angle); const arcEndY = cy - R * Math.sin(angle);
  const trackPath = `M ${trackStart.x} ${trackStart.y} A ${R} ${R} 0 0 1 ${trackEnd.x} ${trackEnd.y}`;
  const activePath = `M ${trackStart.x} ${trackStart.y} A ${R} ${R} 0 ${pct > 0.5 ? 1 : 0} 1 ${arcEndX} ${arcEndY}`;
  const needleLength = 72; const needleX = cx + needleLength * Math.cos(angle); const needleY = cy - needleLength * Math.sin(angle);
  const zones = [
    { label: "0%", angle: Math.PI }, { label: "25%", angle: Math.PI * 0.75 },
    { label: "50%", angle: Math.PI * 0.5 }, { label: "75%", angle: Math.PI * 0.25 }, { label: "100%", angle: 0 },
  ];
  return (
    <svg width="220" height="130" viewBox="0 0 220 130">
      <path d={trackPath} fill="none" stroke={theme.gaugeTrack} strokeWidth="12" strokeLinecap="round" />
      <path d={activePath} fill="none" stroke={theme.text} strokeWidth="12" strokeLinecap="round" />
      {zones.map((z) => {
        const ix = cx + (R - 6) * Math.cos(z.angle); const iy = cy - (R - 6) * Math.sin(z.angle);
        const ox = cx + (R + 6) * Math.cos(z.angle); const oy = cy - (R + 6) * Math.sin(z.angle);
        const lx = cx + (R + 18) * Math.cos(z.angle); const ly = cy - (R + 18) * Math.sin(z.angle);
        return (<g key={z.label}><line x1={ix} y1={iy} x2={ox} y2={oy} stroke={theme.textFaint} strokeWidth="1" /><text x={lx} y={ly + 4} textAnchor="middle" fontSize="8" fill={theme.textFaint} fontFamily="Inter, sans-serif">{z.label}</text></g>);
      })}
      <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke={theme.text} strokeWidth="2" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill={theme.text} /><circle cx={cx} cy={cy} r="2.5" fill={theme.card} />
      <text x={cx} y={cy + 22} textAnchor="middle" fontSize="22" fontWeight="700" fill={theme.text} fontFamily="'Fraunces', serif" letterSpacing="-0.5">{value}%</text>
    </svg>
  );
}

function MatchCard({ match, id, theme }: { match: Match; id: string; theme: typeof lightTheme }) {
  return (
    <div id={id} style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, borderLeft: `3px solid ${theme.text}`, boxShadow: "0 1px 2px rgba(0,0,0,0.04)", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 13, fontWeight: 600, color: theme.text }}>{match.type} · {match.similarity} similarity</div>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 500, color: theme.text, border: `1px solid ${theme.border}`, borderRadius: 999, padding: "2px 10px" }}>S{match.suspectedSentence} → O{match.originalSentence}</div>
      </div>
      <div>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 600, color: theme.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Concepts Copied</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {match.concepts.map((c) => (<span key={c} style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.text, border: `1px solid ${theme.text}`, borderRadius: 999, padding: "2px 10px", background: theme.card }}>{c}</span>))}
        </div>
      </div>
      <blockquote style={{ margin: 0, padding: "10px 14px", background: theme.subBg, borderLeft: `2px solid ${theme.border}`, borderRadius: "0 6px 6px 0", fontFamily: "Inter, sans-serif", fontSize: 13, color: theme.textMuted, lineHeight: 1.6 }}>{match.explanation}</blockquote>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.textMuted, fontStyle: "italic", lineHeight: 1.6 }}><span style={{ fontStyle: "normal", fontWeight: 600, color: theme.textFaint, marginRight: 6 }}>Suspected:</span>"{match.suspectedText}"</div>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.textMuted, fontStyle: "italic", lineHeight: 1.6 }}><span style={{ fontStyle: "normal", fontWeight: 600, color: theme.textFaint, marginRight: 6 }}>Original:</span>"{match.originalText}"</div>
      </div>
    </div>
  );
}

function Heatmap({ onCellClick, theme }: { onCellClick: (n: number) => void; theme: typeof lightTheme }) {
  const [tooltip, setTooltip] = useState<{ n: number; x: number; y: number } | null>(null);
  const cells = Array.from({ length: 29 }, (_, i) => i + 1);
  const colorMap = { original: theme.track, paraphrased: theme.textFaint, exact: theme.text };
  const textColorMap = { original: theme.textFaint, paraphrased: theme.card, exact: theme.card };
  const labelMap = { original: "Original", paraphrased: "Paraphrased", exact: "Exact" };
  return (
    <div>
      <div style={{ marginBottom: 12, display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontFamily: "'Fraunces', serif", fontSize: 15, fontWeight: 600, color: theme.text }}>Plagiarism Heatmap</span>
        <span style={{ fontFamily: "Inter, sans-serif", fontSize: 11, color: theme.textMuted }}>Hover a cell to inspect</span>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
        {(["original", "paraphrased", "exact"] as const).map((t) => (
          <div key={t} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: colorMap[t], border: t === "original" ? `1px solid ${theme.border}` : "none" }} />
            <span style={{ fontFamily: "Inter, sans-serif", fontSize: 11, color: theme.textMuted }}>{labelMap[t]}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, position: "relative" }}>
        {cells.map((n) => {
          const type = getHeatmapType(n); const isClickable = type !== "original";
          return (
            <div key={n} onClick={() => isClickable && onCellClick(n)}
              onMouseEnter={(e) => { const rect = (e.target as HTMLElement).getBoundingClientRect(); setTooltip({ n, x: rect.left, y: rect.top }); }}
              onMouseLeave={() => setTooltip(null)}
              style={{ width: 40, height: 40, borderRadius: 8, background: colorMap[type], display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 500, color: textColorMap[type], cursor: isClickable ? "pointer" : "default", border: type === "original" ? `1px solid ${theme.border}` : "none", userSelect: "none" }}
            >{n}</div>
          );
        })}
      </div>
      {tooltip && (
        <div style={{ position: "fixed", left: tooltip.x + 24, top: tooltip.y - 8, background: theme.text, color: theme.card, fontFamily: "Inter, sans-serif", fontSize: 11, padding: "5px 10px", borderRadius: 6, pointerEvents: "none", zIndex: 100, whiteSpace: "nowrap" }}>
          Sentence {tooltip.n}: {labelMap[getHeatmapType(tooltip.n)]}
        </div>
      )}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>("light");
  const theme = themeMode === "light" ? lightTheme : darkTheme;
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [suspectedFile, setSuspectedFile] = useState<File | null>(null);
  const [sourceText, setSourceText] = useState("");
  const [suspectedText, setSuspectedText] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Load saved theme - DEFAULT LIGHT
  useEffect(() => {
    const saved = localStorage.getItem("theme") as ThemeMode | null;
    if (saved) setThemeMode(saved);
    else setThemeMode("light"); // Always light by default, ignore system dark
  }, []);

  const toggleTheme = () => {
    const next = themeMode === "light" ? "dark" : "light";
    setThemeMode(next);
    localStorage.setItem("theme", next);
  };

  const handleAnalyze = useCallback(() => {
    if (analyzing || showResults) return;
    setAnalyzing(true);
    setTimeout(() => { setAnalyzing(false); setShowResults(true); }, 2200);
  }, [analyzing, showResults]);

  useEffect(() => { if (showResults && resultsRef.current) setTimeout(() => { resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }, 100); }, [showResults]);

  const scrollToMatch = useCallback((sentence: number) => {
    const matchIndex = MOCK_MATCHES.findIndex((m) => m.suspectedSentence === sentence);
    if (matchIndex !== -1) {
      const el = document.getElementById(`match-${matchIndex}`);
      if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.style.outline = `2px solid ${theme.text}`; el.style.outlineOffset = "2px"; setTimeout(() => { el.style.outline = "none"; }, 1800); }
    }
  }, [theme]);

  const SCORE = 48.3; const TOTAL = 14; const EXACT = 6; const PARA = 8;

  return (
    <div style={{ minHeight: "100vh", background: theme.bg, fontFamily: "Inter, sans-serif", transition: "background 0.3s, color 0.3s", colorScheme: themeMode }}>
      <header style={{ borderBottom: `1px solid ${theme.border}`, background: theme.bg, position: "sticky", top: 0, zIndex: 50, transition: "background 0.3s, border-color 0.3s" }}>
        <div style={{ maxWidth: 1120, margin: "0 auto", padding: "0 48px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontFamily: "'Fraunces', serif", fontSize: 14, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: theme.text }}>TextField</span>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <button onClick={toggleTheme} aria-label="Toggle theme" style={{
              width: 36, height: 36, borderRadius: 999, border: `1px solid ${theme.border}`, background: theme.card, color: theme.text, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s"
            }}>
              {themeMode === "light" ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
              )}
            </button>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: 1120, margin: "0 auto", padding: "0 48px 96px" }}>
        <div style={{ paddingTop: 56, paddingBottom: 40 }}>
          <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 700, letterSpacing: "-1px", color: theme.text, margin: "0 0 12px", lineHeight: 1.15 }}>TextField</h1>
          <p style={{ fontFamily: "Inter, sans-serif", fontSize: 16, color: theme.textMuted, margin: 0, lineHeight: 1.5 }}>Beyond percentage — we show which concepts were copied, where, and why</p>
        </div>

        <div style={{ display: "flex", gap: 20, marginBottom: 20 }}>
          <UploadCard label="Original Document (Source)" file={sourceFile} onFile={setSourceFile} text={sourceText} onText={setSourceText} theme={theme} />
          <UploadCard label="Suspected Document" file={suspectedFile} onFile={setSuspectedFile} text={suspectedText} onText={setSuspectedText} theme={theme} />
        </div>

        <button onClick={handleAnalyze} disabled={analyzing || showResults} style={{
          width: "100%", height: 48, background: showResults ? theme.textMuted : theme.text, color: theme.card, fontFamily: "Inter, sans-serif", fontSize: 14, fontWeight: 500, border: "none", borderRadius: 12, cursor: analyzing || showResults ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 10, transition: "all 0.15s", opacity: analyzing ? 0.75 : 1, marginBottom: 64
        }}>
          {analyzing ? "Analyzing documents…" : showResults ? "Analysis Complete" : "Analyze Documents"}
        </button>

        {showResults && (
          <div ref={resultsRef}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 32 }}>
              <div style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, padding: "24px 24px 0", overflow: "hidden" }}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 700, color: theme.text }}>{SCORE}%</div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: theme.textMuted, marginBottom: 20 }}>Overall Plagiarism</div>
                <div style={{ height: 3, background: theme.track }}><div style={{ height: 3, background: theme.text, width: `${SCORE}%` }} /></div>
              </div>
              <div style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, padding: "24px" }}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 700, color: theme.text }}>{TOTAL}</div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: theme.textMuted, marginBottom: 8 }}>Explainable Matches</div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.textFaint }}>{EXACT} Exact • {PARA} Paraphrased</div>
              </div>
              <div style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, padding: "24px" }}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 700, color: theme.text }}>Suspicious</div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: theme.textMuted, marginBottom: 8 }}>Document Health</div>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 12, color: theme.textFaint }}>Needs review</div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 32 }}>
              <div style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, padding: "28px 24px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ fontFamily: "Inter, sans-serif", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: theme.textMuted, marginBottom: 20, alignSelf: "flex-start" }}>Similarity Gauge</div>
                <GaugeSVG value={SCORE} theme={theme} />
              </div>
              <div style={{ background: theme.card, borderRadius: 12, border: `1px solid ${theme.border}`, padding: "28px 24px" }}>
                <Heatmap onCellClick={scrollToMatch} theme={theme} />
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {MOCK_MATCHES.map((m, i) => (<MatchCard key={m.id} match={m} id={`match-${i}`} theme={theme} />))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
